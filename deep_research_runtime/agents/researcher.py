"""Researcher agent: executes search tasks and extracts knowledge cards.

Delegates to retrieval utilities for search, cleaning, scoring, and card extraction.
Implements 3-tier error recovery (retry → reformulate → degrade).
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List

import aiohttp

from ..models import KnowledgeCard, SubTask
from ..query_reform import reformulate_queries
from ..recency import recency_weight
from ..retry import compute_backoff_delay
from ..search_service import SearchService
from .base import AgentContext, call_llm_json, call_llm_text, sanitize_path_name


class ResearcherAgent:
    """Handles all search, content cleaning, scoring, and card extraction.

    Wraps the battle-tested retrieval utilities while adding:
    - 3-tier error recovery (Change 7)
    - Adaptive search profiles (Change 6)
    - Knowledge cache integration (Change 5)
    """

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self._retrieval = _RetrievalHelper(ctx)
        # Dual-layer search service: general engines + vertical plugins.
        self._search_service = SearchService(ctx.settings)

    @property
    def settings(self):
        return self.ctx.settings

    # ── Core Research Execution ──

    async def research_section(
        self,
        *,
        task_id: str,
        topic: str,
        section_id: str,
        pending_tasks: List[SubTask],
        existing_cards: List[KnowledgeCard],
        knowledge_cache: Any = None,
    ) -> Dict[str, Any]:
        """Research a single section's pending tasks. Returns new cards and updated tasks."""
        cards_output: List[KnowledgeCard] = []
        task_updates: Dict[str, SubTask] = {}
        existing_count = len(existing_cards)
        semaphore = asyncio.Semaphore(self.settings.search_concurrency)

        async def run_one(task: SubTask, session: aiohttp.ClientSession) -> None:
            query = str(task["query"])
            intent = str(task["intent"])
            task_key = f"{intent}::{query}"
            result: SubTask = {**task, "status": "failed", "result_count": 0, "searched_urls": []}

            rewritten_queries = [str(q).strip() for q in task.get("rewritten_queries", []) if str(q).strip()] if isinstance(task.get("rewritten_queries"), list) else []
            if not rewritten_queries:
                rewritten_queries = [query]
            search_profile = task.get("search_profile", {}) if isinstance(task.get("search_profile"), dict) else {}

            try:
                # Tier 1: Search with automatic retry
                documents = await self._search_with_retry(session, rewritten_queries, search_profile, semaphore, task_id, topic, task)

                if not documents and self.settings.enable_query_reformulation:
                    # Tier 2: Multi-strategy query reformulation. One LLM call
                    # produces up to N alternatives spanning different
                    # strategies (simplify / synonyms / decompose / ...); we
                    # try them in order and early-exit on the first hit.
                    alt_queries = await self._reformulate_queries(query, intent, task_id, topic)
                    for alt in alt_queries:
                        documents = await self._search_with_retry(
                            session, [alt], search_profile, semaphore, task_id, topic, task
                        )
                        if documents:
                            result["reformulated_query"] = alt
                            break

                if not documents:
                    # Tier 3: Graceful degradation
                    result["status"] = "degraded"
                    result["degradation_reason"] = "No documents retrieved after retry and reformulation."
                    task_updates[task_key] = result
                    return

                result["searched_urls"] = [d.get("url", "Unknown") for d in documents]
                documents = self._retrieval.dedupe_and_rank_documents(documents, task)

                # Call model worker for embedding + reranking
                evidence = await self.ctx.worker_caller(rewritten_queries, documents, task_id=task_id, topic=topic, stage="researcher")
                if not evidence:
                    result["status"] = "degraded"
                    result["degradation_reason"] = "No evidence remained after retrieval and reranking."
                    task_updates[task_key] = result
                    return

                # Extract knowledge cards via LLM
                cards = await self._extract_cards(
                    task_id=task_id, topic=topic,
                    section_id=str(task.get("section_id") or section_id),
                    section_title=str(task.get("section_title") or intent),
                    section_goal=str(task.get("section_goal") or ""),
                    query=query, intent=intent,
                    rewritten_queries=rewritten_queries,
                    evidence=evidence,
                    existing_count=existing_count + len(cards_output),
                )

                if not cards:
                    result["status"] = "degraded"
                    result["degradation_reason"] = "No usable cards extracted from evidence."
                    task_updates[task_key] = result
                    return

                # Deduplicate via knowledge cache if available
                if knowledge_cache is not None:
                    cards = knowledge_cache.add_cards(cards)

                cards_output.extend(cards)
                result["status"] = "completed"
                result["result_count"] = len(cards)
                task_updates[task_key] = result

            except Exception as exc:
                result["error"] = str(exc)
                self.ctx.log_task(task_id, "Query failed.", level="error", stage="researcher", query=query, error=str(exc))
                task_updates[task_key] = result

        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(*(run_one(t, session) for t in pending_tasks))

        return {"cards": cards_output, "task_updates": task_updates}

    # ── Tier 1: Retry with backoff ──

    async def _search_with_retry(
        self, session, queries, profile, semaphore, task_id, topic, task
    ) -> List[Dict[str, Any]]:
        """Run searches with exponential backoff + jitter on transient failures.

        Returns the flattened documents on the first attempt that yields any
        results. Empty result sets count as "soft failures" and are also
        retried, since they often mean rate-limiting / transient 429s rather
        than a genuinely empty corpus.
        """
        max_attempts = max(1, int(self.settings.search_retry_attempts))
        backoff_base = float(self.settings.search_retry_backoff_base)

        for attempt in range(max_attempts):
            try:
                async def fetch_one(q):
                    async with semaphore:
                        source_types = task.get("source_types", [])
                        verticals = task.get("verticals", [])
                        return await self._search_service.search(session, q, profile, verticals=verticals, source_types=source_types)

                batches = await asyncio.gather(*(fetch_one(q) for q in queries), return_exceptions=True)
                flat = []
                for q, batch in zip(queries, batches):
                    if isinstance(batch, Exception):
                        continue
                    for doc in batch:
                        d = dict(doc)
                        d["source_query"] = q
                        d["source_queries"] = [q]
                        flat.append(d)
                if flat:
                    return flat
            except Exception:
                # Don't break out — fall through to the shared sleep step below
                # so empty-result and raised-exception paths share the same
                # backoff policy.
                pass
            if attempt < max_attempts - 1:
                delay = compute_backoff_delay(attempt, base=backoff_base, cap=30.0)
                await asyncio.sleep(delay)
        return []

    # ── Tier 2: Multi-strategy query reformulation ──

    async def _reformulate_queries(
        self, query: str, intent: str, task_id: str, topic: str
    ) -> List[str]:
        """Generate strategy-diverse alternative queries via a single LLM call.

        Returns a list with up to ``settings.max_reformulation_attempts``
        candidates, ordered by the strategy priority defined in
        ``query_reform.STRATEGIES``. The caller tries them in order and
        early-exits on the first non-empty search.

        Delegates the prompt construction and JSON parsing to
        ``query_reform.reformulate_queries`` so this module stays focused
        on the search orchestration. The LLM callable is wrapped to bind
        the agent context, which the helper doesn't need to know about.
        """
        max_attempts = max(1, int(self.settings.max_reformulation_attempts))

        async def _bound_llm(prompt: str, **kwargs: Any) -> Any:
            return await call_llm_json(self.ctx, prompt, **kwargs)

        return await reformulate_queries(
            query,
            intent,
            max_attempts=max_attempts,
            call_llm_json=_bound_llm,
            task_id=task_id,
            topic=topic,
            stage="researcher",
        )

    # ── Card Extraction ──

    async def _extract_cards(self, *, task_id, topic, section_id, section_title, section_goal, query, intent, rewritten_queries, evidence, existing_count) -> List[KnowledgeCard]:
        evidence_records = self._retrieval.make_evidence_records(query, intent, evidence)
        prompt = (
            "You are extracting evidence-grounded research units.\n"
            f"Topic: {topic}\nSection: {section_title}\nSection goal: {section_goal}\n"
            f"Research intent: {intent}\nOriginal search intent: {query}\n"
            f"Executed rewritten queries: {rewritten_queries}\n\n"
            "Allowed evidence records:\n"
            f"{self._retrieval.format_evidence_records(evidence_records)}\n\n"
            "Return JSON with this shape:\n"
            '{"cards":[{"claim":"","evidence_summary":"","evidence_ids":["EVIDENCE-ID"],"claim_type":"fact|definition|procedure|trend|comparison|risk|metric","source_type":"primary_source|secondary_source|analysis|community","time_scope":"historical|current|recent|future|timeless","entities":[""],"stance":"supporting|limitation|counter|neutral","evidence_strength":"strong|medium|weak","confidence":"high|medium|low"}]}\n'
            "Rules:\n"
            "- Every card must be fully supported by the listed evidence_ids.\n"
            "- Do not invent unsupported claims.\n"
            "- Prefer fewer, stronger cards over many weak cards.\n"
            "- If evidence is partial, lower confidence.\n"
        )
        raw = await call_llm_json(self.ctx, prompt, task_id=task_id, topic=topic, stage="researcher", name="card_extraction")
        return self._retrieval.normalize_quality_cards(raw, section_id, evidence_records, existing_count)

    # ── Utility Delegates ──

    def group_cards_by_section(self, cards: List[KnowledgeCard]) -> List[Dict[str, Any]]:
        return self._retrieval.group_cards_by_section(cards)


class _RetrievalHelper:
    """Evidence formatting and card normalization helpers.

    Search and scraping live in SearchService. This helper only handles the
    post-search evidence protocol consumed by the model worker and Writer.
    """

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def dedupe_and_rank_documents(self, docs, task):
        # Pull task-level time_scope once; it doesn't change between docs
        # for the same call, and feeding it into ``_score_document`` lets
        # recency decay see what the user actually asked for.
        time_scope = str((task or {}).get("time_scope") or "")
        by_url: Dict[str, Dict[str, Any]] = {}
        for document in docs:
            url = str(document.get("url") or "").split("#", 1)[0].rstrip("/")
            if not url:
                continue
            candidate = dict(document)
            candidate["content"] = str(candidate.get("content") or candidate.get("raw_content") or "").strip()
            if not candidate["content"]:
                continue
            existing = by_url.get(url)
            if existing is None or self._score_document(candidate, time_scope) > self._score_document(existing, time_scope):
                candidate["search_quality_score"] = self._score_document(candidate, time_scope)
                by_url[url] = candidate
        return sorted(by_url.values(), key=lambda item: float(item.get("search_quality_score") or 0.0), reverse=True)

    def _score_document(self, document: Dict[str, Any], time_scope: str = "") -> float:
        """Compute the source-quality score used for ranking and dedup tiebreaks.

        Components, in order of contribution:

        1. ``provider_score`` — whatever the search engine returned.
        2. Body-length bonus — capped at 1.0 to avoid massively long pages
           winning purely on word count.
        3. Layer/kind bonuses — vertical (academic) and primary-format
           (paper/pdf/repo) sources get a fixed boost.
        4. Recency multiplier — applied last, gated on ``time_scope``.
           Returns ≤ 1.0 so it only ever de-emphasises stale content; we
           never *boost* a doc above the additive score on freshness alone.
        """
        content = str(document.get("content") or "")
        score = float(document.get("score", document.get("provider_score", 0.0)) or 0.0)
        score += min(1.0, len(content) / 3000.0)
        if str(document.get("source_layer") or "") == "vertical":
            score += 0.25
        if str(document.get("source_kind") or document.get("page_type") or "") in {"paper", "pdf", "repo"}:
            score += 0.15

        # Recency decay (multiplicative, gated by settings + time_scope).
        # We attach the weight to the document so downstream consumers
        # (writer, fidelity check) can inspect *why* one source ranked
        # higher than another without re-deriving the math.
        settings = getattr(self.ctx, "settings", None)
        if settings is not None and getattr(settings, "recency_weighting_enabled", True) and time_scope:
            published = document.get("published_time") or document.get("year")
            half_lives = {
                "recent": settings.recency_half_life_recent_months,
                "current": settings.recency_half_life_current_months,
                "current_year": settings.recency_half_life_current_months,
            }
            weight = recency_weight(
                published,
                time_scope,
                half_lives_override=half_lives,
                default_half_life_months=float(settings.recency_half_life_default_months),
            )
            document["recency_weight"] = weight
            score *= weight
        return score

    def make_evidence_records(self, query, intent, evidence):
        prefix = f"{self._compact_slug(intent, 'INT')}-{self._compact_slug(query, 'QRY')}"
        records = []
        for index, item in enumerate(evidence, start=1):
            records.append({
                "evidence_id": f"{prefix}-{index:02d}",
                "query": query,
                "intent": intent,
                "url": item.get("url", "Unknown"),
                "title": item.get("title", ""),
                "published_time": item.get("published_time", ""),
                "score": float(item.get("score", 0.0) or 0.0),
                "page_type": str(item.get("page_type") or item.get("source_kind") or "web"),
                "source_type": str(item.get("source_type") or "analysis"),
                "content_quality_score": float(item.get("content_quality_score", 0.0) or 0.0),
                "excerpt": str(item.get("excerpt", "")).strip(),
            })
        return records

    @staticmethod
    def format_evidence_records(records):
        blocks = []
        for item in records:
            blocks.append(
                "\n".join([
                    f"[{item['evidence_id']}]",
                    f"title: {item.get('title', '')}",
                    f"url: {item.get('url', 'Unknown')}",
                    f"published_time: {item.get('published_time', '')}",
                    f"page_type: {item.get('page_type', '')}",
                    f"source_type: {item.get('source_type', '')}",
                    f"score: {item.get('score', 0.0):.4f}",
                    "excerpt:",
                    item.get("excerpt", ""),
                ])
            )
        return "\n\n".join(blocks)

    def normalize_quality_cards(self, raw, section_id, evidence_records, existing_count):
        evidence_map = {item["evidence_id"]: item for item in evidence_records}
        payload = raw.get("cards") if isinstance(raw, dict) else raw
        if not isinstance(payload, list):
            return []

        cards: List[KnowledgeCard] = []
        seen = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            evidence_ids = [
                str(eid).strip()
                for eid in item.get("evidence_ids", [])
                if str(eid).strip() in evidence_map
            ] if isinstance(item.get("evidence_ids"), list) else []
            fallback = str(item.get("evidence_id") or "").strip()
            if not evidence_ids and fallback in evidence_map:
                evidence_ids = [fallback]
            if not evidence_ids:
                continue
            primary = evidence_map[evidence_ids[0]]
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            key = (claim.lower(), primary["evidence_id"])
            if key in seen:
                continue
            seen.add(key)

            confidence = self._normalize_choice(item.get("confidence"), {"high", "medium", "low"}, "medium")
            claim_type = self._normalize_choice(item.get("claim_type"), {"fact", "definition", "procedure", "trend", "comparison", "risk", "metric"}, "fact")
            time_scope = self._normalize_choice(item.get("time_scope"), {"historical", "current", "recent", "future", "timeless"}, "recent" if str(primary.get("published_time") or "").strip() else "timeless")
            stance = self._normalize_choice(item.get("stance"), {"supporting", "limitation", "counter", "neutral"}, "neutral")
            evidence_strength = self._normalize_choice(item.get("evidence_strength"), {"strong", "medium", "weak"}, "medium")
            entities = self._dedupe_text_values([str(entity).strip() for entity in item.get("entities", [])])[:6] if isinstance(item.get("entities"), list) else []

            cards.append({
                "unit_id": f"U{existing_count + len(cards) + 1:03d}",
                "section_id": section_id,
                "claim": claim,
                "evidence_summary": str(item.get("evidence_summary") or primary["excerpt"][:280]).strip(),
                "exact_excerpt": primary["excerpt"],
                "evidence_id": primary["evidence_id"],
                "source": primary["url"],
                "source_title": primary.get("title", ""),
                "source_type": str(item.get("source_type") or primary.get("source_type") or "analysis"),
                "claim_type": claim_type,
                "time_scope": time_scope,
                "entities": entities,
                "stance": stance,
                "evidence_strength": evidence_strength,
                "evidence_score": float(primary.get("score", 0.0) or 0.0),
                "confidence": confidence,
            })
        return cards

    @staticmethod
    def group_cards_by_section(cards):
        grouped: Dict[str, Dict[str, Any]] = {}
        for card in cards:
            section_id = card.get("section_id") or "UNASSIGNED"
            entry = grouped.setdefault(section_id, {"section_id": section_id, "cards": []})
            entry["cards"].append(card)
        return sorted(grouped.values(), key=lambda item: item["section_id"])

    @staticmethod
    def _compact_slug(value: str, fallback: str = "SRC") -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "", value or "").upper()
        return slug[:8] or fallback

    @staticmethod
    def _normalize_choice(value: Any, valid: set[str], fallback: str) -> str:
        parsed = str(value or "").strip().lower()
        return parsed if parsed in valid else fallback

    @staticmethod
    def _dedupe_text_values(values: List[str]) -> List[str]:
        # Shim → :func:`agents.base.dedupe_preserving_order`.
        from .base import dedupe_preserving_order
        return dedupe_preserving_order(values)
