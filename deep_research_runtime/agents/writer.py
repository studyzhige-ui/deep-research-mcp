"""Writer agent: generates the final evidence-grounded report with clickable citations."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from ..citation_grounding import (
    assign_evidence_ids,
    build_audit_summary,
    format_evidence_for_prompt,
    parse_writer_response,
    render_paragraphs_to_markdown,
    validate_paragraphs,
)
from ..models import ResearchState, SectionDigest
from ..quality import QualityMixin
from ..storage import now_iso
from .base import AgentContext, call_llm_json, call_llm_text, infer_user_language


def _language_name(lang: str) -> str:
    return "Chinese" if lang == "zh" else "English"


class WriterAgent:
    """Writes the final research report with citation verification."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self._quality = QualityMixin()

    @property
    def settings(self):
        return self.ctx.settings

    # ── Conflict prompt formatting (Improvement ④) ──

    @staticmethod
    def _format_conflicts_for_section_prompt(records: List[Dict[str, Any]]) -> str:
        """Render conflict records into the writer's section prompt.

        Returns ``""`` (an empty string) when ``records`` is empty so the
        f-string interpolation in the caller cleanly produces no marker
        line at all. This is the "conditional injection" pattern — we do
        NOT emit a ``Conflicts: []`` header for sections without
        disagreement, because that would prime the LLM to invent
        disagreement phrasing.

        For sections that DO have conflicts, the LLM is told both the
        raw pair text and an explicit instruction to surface the
        disagreement instead of arbitrarily choosing one side.
        """
        if not records:
            return ""
        lines = ["Cross-source disagreements detected in this section:"]
        for idx, rec in enumerate(records, start=1):
            severity = rec.get("severity", "moderate")
            topic = rec.get("topic", "")
            summary = rec.get("disagreement_summary") or "Sources disagree."
            claim_a = rec.get("claim_a", "")
            claim_b = rec.get("claim_b", "")
            src_a = rec.get("source_a_title") or rec.get("source_a_url") or "Source A"
            src_b = rec.get("source_b_title") or rec.get("source_b_url") or "Source B"
            lines.append(
                f"  {idx}. [{severity}] {topic}\n"
                f"     - {src_a}: {claim_a}\n"
                f"     - {src_b}: {claim_b}\n"
                f"     Summary: {summary}"
            )
        lines.append(
            "When writing this section, explicitly acknowledge each disagreement. "
            "Phrase it as 'Sources disagree on X: [a] reports ... while [b] reports ...' "
            "and never silently pick one side. For 'strong' severity, foreground the "
            "disagreement; for 'weak', a brief note is sufficient."
        )
        return "\n".join(lines) + "\n\n"

    # ── Digest Formatting ──

    def _format_section_digest_for_writer(
        self, digest: SectionDigest, previous_summaries: List[str],
        report_title: str, user_goal: str, output_language: str,
        evidence_requirements: List[str] | None = None,
    ) -> str:
        lines = [
            f"Report title: {report_title}",
            f"User goal: {user_goal}",
            f"Output language: {_language_name(output_language)}",
            f"Section title: {digest.get('title', '')}",
            f"Section objective: {digest.get('purpose', '')}",
            f"Coverage score: {digest.get('coverage_score', 0)}",
            f"Evidence count score: {digest.get('evidence_count_score', 0)}",
            f"Source diversity score: {digest.get('source_diversity_score', 0)}",
            f"Review reason: {digest.get('review_reason', '')}",
        ]
        questions = digest.get("questions") if isinstance(digest.get("questions"), list) else []
        if questions:
            lines.extend(["Questions to answer:"] + [f"- {q}" for q in questions])
        missing = digest.get("missing_questions") if isinstance(digest.get("missing_questions"), list) else []
        if missing:
            lines.extend(["Coverage gaps:"] + [f"- {q}" for q in missing])
        if previous_summaries:
            lines.extend(["Previous section summaries:"] + [f"- {s}" for s in previous_summaries[-2:]])
        if evidence_requirements:
            lines.extend(["Evidence requirements from outline:"] + [f"- {item}" for item in evidence_requirements if str(item).strip()])

        lines.append("Evidence items:")
        for item in digest.get("items", []) if isinstance(digest.get("items"), list) else []:
            citations = "".join(f"[{n}]" for n in item.get("reference_numbers", []))
            lines.extend([
                f"- claim: {item.get('claim', '')} {citations}".strip(),
                f"  confidence: {item.get('confidence', 'medium')}",
                f"  source_title: {item.get('source_title', '')}",
                f"  evidence_summary: {item.get('evidence_summary', '')}",
                f"  excerpt: {item.get('exact_excerpt', '')}",
            ])
        return "\n".join(lines)

    # ── Citation Verification (Change 4) ──

    async def _verify_section_citations(
        self, section_body: str, digest: SectionDigest, sources: List[Dict[str, Any]],
        task_id: str, topic: str,
    ) -> str:
        """Verify citations are clickable source links or resolvable source numbers."""
        if not self.settings.enable_citation_verification:
            return section_body

        source_map = {s.get("reference_number", 0): s for s in sources if isinstance(s, dict)}
        allowed_urls = {str(s.get("source_url") or "").strip() for s in sources if str(s.get("source_url") or "").strip()}

        def validate_html_link(match: re.Match) -> str:
            url = match.group(1).strip()
            label = match.group(2).strip()
            if url in allowed_urls:
                return match.group(0)
            return label

        section_body = re.sub(r'<a\s+href="(https?://[^"]+)">\s*([^<]+?)\s*</a>', validate_html_link, section_body)

        protected_links: List[str] = []

        def protect_html_link(match: re.Match) -> str:
            token = f"@@DR_LINK_{len(protected_links)}@@"
            protected_links.append(match.group(0))
            return token

        section_body = re.sub(r'<a\s+href="https?://[^"]+">\s*[^<]+?\s*</a>', protect_html_link, section_body)

        citation_links: List[str] = []

        def replace_plain(match: re.Match) -> str:
            try:
                ref_num = int(match.group(1))
            except Exception:
                return ""
            src = source_map.get(ref_num)
            if not src or not str(src.get("source_url") or "").strip():
                return ""
            token = f"@@DR_CIT_{len(citation_links)}@@"
            citation_links.append(self._markdown_citation_link(ref_num, src))
            return token

        section_body = re.sub(r"<sup>\[(\d+)\]</sup>", replace_plain, section_body)
        section_body = re.sub(r"(?<!\])\[(\d+)\](?!\()", replace_plain, section_body)

        def validate_link(match: re.Match) -> str:
            label = match.group(1).strip()
            url = match.group(2).strip()
            if url in allowed_urls:
                return match.group(0)
            return label

        section_body = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", validate_link, section_body)

        for index, link in enumerate(citation_links):
            section_body = section_body.replace(f"@@DR_CIT_{index}@@", link)
        for index, link in enumerate(protected_links):
            section_body = section_body.replace(f"@@DR_LINK_{index}@@", link)

        return section_body

    @staticmethod
    def _markdown_source_link(source: Dict[str, Any]) -> str:
        url = str(source.get("source_url") or "").strip()
        title = str(source.get("source_title") or "").strip() or url
        title = title.replace("[", "").replace("]", "")
        if not url:
            return title
        return f"[{title}]({url})"

    @staticmethod
    def _markdown_citation_link(reference_number: int, source: Dict[str, Any]) -> str:
        url = str(source.get("source_url") or "").strip()
        label = f"[{reference_number}]"
        if not url:
            return label
        return f'<sup><a href="{url}">{label}</a></sup>'

    def _replace_numbered_citations_with_links(self, text: str, sources: List[Dict[str, Any]]) -> str:
        source_map = {int(s.get("reference_number")): s for s in sources if isinstance(s, dict) and s.get("reference_number")}
        citation_links: List[str] = []

        def replace_group(match: re.Match) -> str:
            numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
            links = "".join(self._markdown_citation_link(n, source_map[n]) for n in numbers if n in source_map)
            token = f"@@DR_CIT_{len(citation_links)}@@"
            citation_links.append(links)
            return token

        value = re.sub(r"<sup>(\[(?:\d+\])(?:\[\d+\])*)</sup>", replace_group, str(text or ""))
        value = re.sub(r"(?<!\])(\[(?:\d+\])(?:\[\d+\])*)(?!\()", replace_group, value)
        for index, link in enumerate(citation_links):
            value = value.replace(f"@@DR_CIT_{index}@@", link)
        return value

    # ── Section Outline Builder ──

    @staticmethod
    def _build_section_outline_item(section_plan, digest, section_body):
        body_text = str(section_body or "").strip()
        max_chars = 320
        summary = body_text[:max_chars] + ("..." if len(body_text) > max_chars else "")
        return {
            "section_id": str(section_plan.get("section_id") or digest.get("section_id") or ""),
            "title": str(digest.get("title") or section_plan.get("title") or ""),
            "section_summary": summary,
            "key_claims": digest.get("key_claims", [])[:5],
        }

    def _resolve_section_digest_ids(self, section_plan: Dict[str, Any], digest_map: Dict[str, SectionDigest]) -> List[str]:
        raw_ids = section_plan.get("evidence_digest_ids")
        if not isinstance(raw_ids, list):
            raw_ids = section_plan.get("source_digest_ids") if isinstance(section_plan.get("source_digest_ids"), list) else []
        candidates = [str(value).strip() for value in raw_ids if str(value).strip()]
        section_id = str(section_plan.get("section_id") or "").strip()
        if section_id:
            candidates.append(section_id)
        return [value for value in dict.fromkeys(candidates) if value in digest_map]

    def _merge_digests_for_section(self, section_plan: Dict[str, Any], digests: List[SectionDigest]) -> SectionDigest:
        if len(digests) == 1:
            merged = dict(digests[0])
            merged["title"] = str(section_plan.get("title") or merged.get("title") or "")
            merged["purpose"] = str(section_plan.get("purpose") or merged.get("purpose") or "")
            if isinstance(section_plan.get("questions"), list) and section_plan.get("questions"):
                merged["questions"] = section_plan.get("questions")
            return merged

        max_items = max(3, self.settings.writer_max_cards_per_section)
        merged_items: List[Dict[str, Any]] = []
        key_claims: List[str] = []
        missing_questions: List[str] = []
        questions: List[str] = []
        for digest in digests:
            for item in digest.get("items", []) if isinstance(digest.get("items"), list) else []:
                if isinstance(item, dict):
                    merged_items.append(dict(item))
            if isinstance(digest.get("key_claims"), list):
                key_claims.extend(str(value).strip() for value in digest.get("key_claims", []) if str(value).strip())
            if isinstance(digest.get("missing_questions"), list):
                missing_questions.extend(str(value).strip() for value in digest.get("missing_questions", []) if str(value).strip())
            if isinstance(digest.get("questions"), list):
                questions.extend(str(value).strip() for value in digest.get("questions", []) if str(value).strip())

        merged_items = merged_items[:max_items]
        coverage_scores = [float(d.get("coverage_score") or 0.0) for d in digests]
        evidence_scores = [float(d.get("evidence_count_score") or 0.0) for d in digests]
        diversity_scores = [float(d.get("source_diversity_score") or 0.0) for d in digests]
        avg = lambda values: round(sum(values) / max(1, len(values)), 2)
        return {
            "section_id": str(section_plan.get("section_id") or "+".join(str(d.get("section_id") or "") for d in digests)),
            "title": str(section_plan.get("title") or " / ".join(str(d.get("title") or "") for d in digests if str(d.get("title") or ""))),
            "purpose": str(section_plan.get("purpose") or ""),
            "questions": [str(v).strip() for v in section_plan.get("questions", []) if str(v).strip()] if isinstance(section_plan.get("questions"), list) and section_plan.get("questions") else list(dict.fromkeys(questions))[:6],
            "coverage_score": avg(coverage_scores),
            "evidence_count_score": avg(evidence_scores),
            "source_diversity_score": avg(diversity_scores),
            "is_enough": all(bool(d.get("is_enough")) for d in digests),
            "review_reason": "Merged evidence digest built from compressed section evidence packages.",
            "missing_questions": list(dict.fromkeys(missing_questions))[:6],
            "key_claims": list(dict.fromkeys(key_claims))[:6],
            "items": merged_items,
        }

    @staticmethod
    def _requirement_keywords(requirements: List[str]) -> set[str]:
        text = " ".join(str(item or "") for item in requirements)
        raw = re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "need", "needs", "use",
            "prefer", "include", "evidence", "source", "card", "cards", "exact", "excerpt",
            "优先", "使用", "证据", "来源", "需要", "覆盖", "包含", "原文", "摘录",
        }
        return {item for item in raw if item not in stopwords}

    def _select_raw_cards_for_section(
        self,
        cards: List[Dict[str, Any]],
        digest_ids: List[str],
        digest: SectionDigest,
        evidence_requirements: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        digest_id_set = {str(value).strip() for value in digest_ids if str(value).strip()}
        requirement_keywords = self._requirement_keywords(evidence_requirements or [])
        digest_claims = {
            re.sub(r"\s+", " ", str(item.get("claim") or "").strip()).lower()[:160]
            for item in digest.get("items", []) if isinstance(item, dict)
        }
        candidates: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for card in cards:
            if not isinstance(card, dict):
                continue
            section_id = str(card.get("section_id") or "").strip()
            claim = re.sub(r"\s+", " ", str(card.get("claim") or "").strip())
            claim_key = claim.lower()[:160]
            source = str(card.get("source") or "").strip()
            if digest_id_set and section_id not in digest_id_set and claim_key not in digest_claims:
                continue
            key = (claim_key, source)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(card)

        def score(card: Dict[str, Any]) -> float:
            base = float(card.get("evidence_score") or 0.0)
            confidence = str(card.get("confidence") or "").lower()
            if confidence == "high":
                base += 0.15
            elif confidence == "low":
                base -= 0.1
            if str(card.get("exact_excerpt") or "").strip():
                base += 0.05
            if requirement_keywords:
                haystack = " ".join(
                    str(card.get(key) or "")
                    for key in ("claim", "evidence_summary", "exact_excerpt", "source_title", "claim_type", "source_type", "time_scope")
                ).lower()
                overlap = sum(1 for keyword in requirement_keywords if keyword in haystack)
                base += min(0.45, overlap * 0.15)
            return base

        return sorted(candidates, key=score, reverse=True)[: max(0, self.settings.writer_raw_cards_per_section)]

    def _format_raw_cards_for_writer(
        self,
        cards: List[Dict[str, Any]],
        ref_by_url: Dict[str, int],
        ref_by_domain: Dict[str, int],
    ) -> str:
        if not cards:
            return "Selected raw KnowledgeCards: none"
        lines = ["Selected raw KnowledgeCards for precision:"]
        for index, card in enumerate(cards, start=1):
            source_url = str(card.get("source") or "").strip()
            ref_num = ref_by_url.get(source_url)
            if ref_num is None:
                from urllib.parse import urlparse

                ref_num = ref_by_domain.get((urlparse(source_url).netloc or "").strip().lower())
            citation = f"[{ref_num}]" if ref_num else ""
            excerpt = re.sub(r"\s+", " ", str(card.get("exact_excerpt") or "").strip())[: max(120, self.settings.writer_excerpt_chars)]
            lines.extend([
                f"- raw_card_{index}: {str(card.get('claim') or '').strip()} {citation}".strip(),
                f"  confidence: {str(card.get('confidence') or 'medium')}",
                f"  source_title: {str(card.get('source_title') or '').strip()}",
                f"  evidence_summary: {str(card.get('evidence_summary') or '').strip()}",
                f"  excerpt: {excerpt}",
            ])
        return "\n".join(lines)

    # ── Grounded section writing (Improvement ④ redesigned) ──

    async def _write_section_grounded(
        self,
        *,
        section_plan: Dict[str, Any],
        section_id: str,
        digest: SectionDigest,
        raw_cards: List[Dict[str, Any]],
        ref_by_url: Dict[str, int],
        ref_by_domain: Dict[str, int],
        evidence_requirements: List[str],
        topic: str,
        output_language: str,
        output_language_name: str,
        conflicts_block: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """Write a single section using the structured-grounding path.

        The contract here is different from the legacy free-form writer:
        the LLM is asked to return a JSON ``{paragraphs: [...]}`` instead
        of Markdown. Each paragraph declares which ``evidence_ids`` (from a
        closed set) back its claims. The system then renders the
        Markdown — the LLM never gets to write a raw ``[3]`` itself, so it
        can't invent citation numbers.

        Returns ``{"body": str, "audit": dict, "fallback": bool}``.
        When ``fallback`` is True the caller should fall back to the
        legacy free-form path; the body and audit will be empty.
        """
        # Stamp each raw card with a prompt-friendly evidence_id (E1, E2…).
        # We also stash the reference_number that the writer's existing
        # citation pipeline expects so the rendered Markdown produces the
        # same superscript style as legacy output.
        annotated_cards: List[Dict[str, Any]] = []
        for card in raw_cards:
            copy = dict(card)
            source_url = str(card.get("source") or "").strip()
            ref_num = ref_by_url.get(source_url)
            if ref_num is None:
                from urllib.parse import urlparse
                ref_num = ref_by_domain.get((urlparse(source_url).netloc or "").strip().lower())
            if ref_num is not None:
                copy["reference_number"] = ref_num
            annotated_cards.append(copy)

        annotated, id_to_card = assign_evidence_ids(annotated_cards)
        # Drop cards without a usable reference number — they can't be
        # rendered as a citation anyway, so passing them to the LLM only
        # invites the LLM to cite something that can't be rendered.
        annotated = [c for c in annotated if c.get("reference_number") is not None]
        id_to_card = {c["evidence_id"]: c for c in annotated}
        if not annotated:
            return {"body": "", "audit": {"reason": "no_renderable_cards"}, "fallback": True}

        evidence_block = format_evidence_for_prompt(annotated)
        allowed_ids = ", ".join(c["evidence_id"] for c in annotated)

        prompt = (
            "You are writing one section of a deep research report as STRUCTURED JSON.\n"
            f"Output language: {output_language_name}\nTopic: {topic}\n"
            f"Section title: {section_plan.get('title') or digest.get('title', section_id)}\n\n"
            "Hard rules — violating these makes your output unusable:\n"
            f"- You may ONLY cite evidence IDs from this closed set: [{allowed_ids}]\n"
            "- Each paragraph's evidence_ids array must reference one or more of those IDs.\n"
            "- Never invent IDs, citation numbers, or URLs.\n"
            "- When a paragraph contains a numeric or factual claim (numbers, percentages, dates,\n"
            "  named benchmarks, named people, named products), include a `quote` field with the\n"
            "  verbatim phrase from the cited evidence that grounds the claim.\n"
            "- Do NOT write '[1]' style markers in your `text` field — the system renders citations.\n"
            "- Output strictly valid JSON, no commentary, no markdown fences.\n\n"
            "Soft rules — for report quality:\n"
            "- 4-8 paragraphs total. Begin by directly answering the section's main question.\n"
            "- Organize around 2-4 analytical judgments, not source-by-source summaries.\n"
            "- Preserve important specifics and representative examples.\n"
            "- Be explicit about uncertainty when evidence is thin or mixed.\n"
            "- Avoid generic filler, marketing tone, and weak transitions.\n\n"
            "Return JSON of this exact shape:\n"
            '  {"paragraphs": [\n'
            '     {"text": "...", "evidence_ids": ["E1", "E3"], "quote": "verbatim phrase from E1 or E3"},\n'
            '     {"text": "...", "evidence_ids": ["E2"]},\n'
            '     ...\n'
            '  ]}\n\n'
            f"{conflicts_block}"
            "Evidence catalog (closed set — cite ONLY these IDs):\n"
            f"{evidence_block}\n"
        )

        try:
            raw_response = await call_llm_json(
                self.ctx, prompt,
                task_id=task_id, topic=topic, stage="writer",
                name=f"{section_id}_section_grounded",
            )
        except Exception as exc:
            self.ctx.log_task(
                task_id, "Grounded writer LLM call failed, falling back.",
                level="warning", stage="writer", section_id=section_id, error=str(exc),
            )
            return {"body": "", "audit": {"reason": "llm_error", "error": str(exc)}, "fallback": True}

        paragraphs = parse_writer_response(raw_response)
        if paragraphs is None or not paragraphs:
            return {"body": "", "audit": {"reason": "malformed_response"}, "fallback": True}

        cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
        # If validation ate all paragraphs (e.g. every evidence_id was
        # invented), the section is unsalvageable on this path — fall back.
        if not cleaned:
            return {"body": "", "audit": {"reason": "all_paragraphs_dropped", "details": audit}, "fallback": True}

        body = render_paragraphs_to_markdown(cleaned, id_to_card)
        return {"body": body, "audit": audit, "fallback": False, "paragraphs": cleaned}

    # ── Main Write ──

    async def write_report(self, state: ResearchState, get_task_output_dir) -> Dict[str, Any]:
        """Write the full research report. Returns {'final_report': str}."""
        task_id = state["task_id"]
        topic = state["topic"]
        writer_started_at = time.perf_counter()
        ctx = self.ctx

        await ctx.store.append_task_event(task_id, "writer", "Writing the final evidence-grounded report package.")
        await ctx.store.set_status(task_id, "Stage 4/4: writing the final evidence-grounded report.", lifecycle="running", stage="writer")
        ctx.log_task(task_id, "Writer started.", stage="writer")

        task_dir = get_task_output_dir(task_id, topic)
        report_path = task_dir / f"DeepResearch_Report_{task_id}.md"
        cards_path = task_dir / f"DeepResearch_Cards_{task_id}.json"
        sources_path = task_dir / f"DeepResearch_Sources_{task_id}.json"
        activity_path = task_dir / f"DeepResearch_Activity_{task_id}.json"
        metadata_path = task_dir / f"DeepResearch_Metadata_{task_id}.json"

        cards = state.get("knowledge_cards", [])
        cards_json = json.dumps(cards, ensure_ascii=False, indent=2)
        source_catalog = self._quality._build_source_catalog(cards)
        quality_review = state.get("quality_review", {})
        plan_data = state.get("plan_data") or {}
        output_language = str(plan_data.get("output_language") or infer_user_language(state.get("topic", ""), state.get("approved_plan", "")))
        output_language_name = _language_name(output_language)
        section_digests = state.get("section_digests") if isinstance(state.get("section_digests"), list) else []
        # Map of section_id → list of conflict records produced by the
        # conflict_detector node upstream. Sparse — sections without
        # detected conflicts are absent rather than empty-listed, which
        # lets the prompt builder skip them cleanly.
        section_conflicts: Dict[str, List[Dict[str, Any]]] = (
            state.get("section_conflicts") if isinstance(state.get("section_conflicts"), dict) else {}
        )

        if cards:
            sections = plan_data.get("sections") if isinstance(plan_data.get("sections"), list) else []
            digest_map = {str(d.get("section_id") or ""): d for d in section_digests if isinstance(d, dict)}
            sources = self._quality._filter_report_sources(source_catalog, section_digests)
            sources, ref_by_url, ref_by_domain = self._quality._build_reference_index(sources)
            sources_json = json.dumps(sources, ensure_ascii=False, indent=2)

            section_blocks = []
            section_outlines = []
            final_section_digests = []
            # Citation audit accumulator. One entry per section with the
            # grounding stats (or ``writer_fallback: true`` when we had to
            # use the legacy path). Returned in the writer's final dict so
            # the graph node propagates it into ResearchState.
            citation_audit_per_section: List[Dict[str, Any]] = []

            for index, section_plan in enumerate(sections, start=1):
                section_started_at = time.perf_counter()
                section_id = str(section_plan.get("section_id") or f"S{index:02d}")
                digest_ids = self._resolve_section_digest_ids(section_plan, digest_map)
                if not digest_ids:
                    continue
                evidence_requirements = [str(value).strip() for value in section_plan.get("evidence_requirements", []) if str(value).strip()] if isinstance(section_plan.get("evidence_requirements"), list) else []
                digest = self._merge_digests_for_section(section_plan, [digest_map[digest_id] for digest_id in digest_ids])
                digest = self._quality._attach_reference_numbers_to_digest(digest, ref_by_url, ref_by_domain)
                raw_cards = self._select_raw_cards_for_section(cards, digest_ids, digest, evidence_requirements)
                raw_cards_brief = self._format_raw_cards_for_writer(raw_cards, ref_by_url, ref_by_domain)
                final_section_digests.append(digest)

                previous_summaries = [str(o.get("section_summary") or "").strip() for o in section_outlines[-2:] if str(o.get("section_summary") or "").strip()]
                brief = self._format_section_digest_for_writer(
                    digest=digest, previous_summaries=previous_summaries,
                    report_title=str(plan_data.get("report_title") or topic),
                    user_goal=str(plan_data.get("user_goal") or topic),
                    output_language=output_language,
                    evidence_requirements=evidence_requirements,
                )
                # Conditionally append a "conflicts to surface" block to
                # the section prompt. We do NOT include a "no conflicts:
                # []" header when nothing was detected — that would prime
                # the LLM to invent disagreement language. When the block
                # IS present, the writer is instructed to phrase the
                # disagreement explicitly rather than picking one side.
                conflicts_block = self._format_conflicts_for_section_prompt(
                    section_conflicts.get(section_id, [])
                )
                prompt = (
                    "Write one section of a deep research report.\n"
                    f"Output language: {output_language_name}\nTopic: {topic}\n"
                    f"Section title: {section_plan.get('title') or digest.get('title', section_id)}\n\n"
                    "Use the evidence items below as the full allowed evidence base.\n"
                    "Requirements:\n"
                    "- Begin by directly answering the section's main question in one or two sentences.\n"
                    "- Organize the section around 2-4 analytical judgments, not source-by-source summaries.\n"
                    "- For each major judgment, explain why it matters.\n"
                    "- Preserve important specifics and representative examples.\n"
                    "- Be explicit about uncertainty when evidence is thin or mixed.\n"
                    "- Use numbered citations like [1] or [1][2] only when supported by the evidence items.\n"
                    "- Never invent citations or sources.\n"
                    "- Treat Section digest as the broad outline of available evidence.\n"
                    "- Use Selected raw KnowledgeCards for precise details, quotes, and citations.\n"
                    "- Avoid generic filler, marketing tone, and weak transitions.\n\n"
                    "Data visualization guidelines (Improvement 6):\n"
                    "- When comparing 3+ entities/models/tools with shared attributes, use a Markdown TABLE.\n"
                    "- When showing chronological events or version history, use a Markdown TABLE with date column.\n"
                    "- When illustrating a process/workflow/architecture with 3+ steps, use a Mermaid flowchart:\n"
                    "  ```mermaid\n  graph LR\n    A[Step 1] --> B[Step 2] --> C[Step 3]\n  ```\n"
                    "- When comparing 2 sides (pros/cons, before/after), use a two-column TABLE.\n"
                    "- Only use tables and diagrams when there is REAL DATA from the evidence items.\n"
                    "- Do NOT force tables or diagrams when the content is purely narrative.\n"
                    "- Bold the best/highest value in comparison tables to highlight key takeaways.\n\n"
                    f"{conflicts_block}"
                    f"Section digest:\n{brief}\n"
                    f"\n{raw_cards_brief}\n"
                )
                # ── Grounded-generation path (Improvement ④ redesigned) ──
                # When enabled, ask the LLM for STRUCTURED JSON where each
                # paragraph declares which evidence_ids back it. The system
                # then renders the citation superscripts — the LLM physically
                # cannot invent citation numbers because the system, not
                # the LLM, writes the [N] markers.
                section_audit: Dict[str, Any] = {}
                used_grounded = False
                if getattr(self.settings, "enable_grounded_citations", True):
                    grounded = await self._write_section_grounded(
                        section_plan=section_plan,
                        section_id=section_id,
                        digest=digest,
                        raw_cards=raw_cards,
                        ref_by_url=ref_by_url,
                        ref_by_domain=ref_by_domain,
                        evidence_requirements=evidence_requirements,
                        topic=topic,
                        output_language=output_language,
                        output_language_name=output_language_name,
                        conflicts_block=conflicts_block,
                        task_id=task_id,
                    )
                    if not grounded.get("fallback"):
                        body = grounded["body"]
                        section_audit = grounded["audit"]
                        section_audit["section_id"] = section_id
                        used_grounded = True

                if not used_grounded:
                    # Legacy free-form path. Either grounded mode is
                    # disabled by config, or the structured call failed
                    # and we degraded gracefully. Either way the existing
                    # citation-link rewriter still runs.
                    body = await call_llm_text(
                        ctx, prompt,
                        task_id=task_id, topic=topic, stage="writer",
                        name=f"{section_id}_section_writer",
                    )
                    section_audit = {
                        "section_id": section_id,
                        "writer_fallback": True,
                        "reason": (
                            "grounded_disabled"
                            if not getattr(self.settings, "enable_grounded_citations", True)
                            else "fallback_after_failure"
                        ),
                    }

                # Both paths feed through the same citation-link rewriter
                # so the output style (superscript <sup>[N]</sup> + clickable
                # URL) and references list are identical regardless of path.
                body = self._replace_numbered_citations_with_links(body, sources).strip()

                # Belt-and-suspenders URL validation (Change 4 legacy). The
                # grounded path emits only valid refs by construction, but
                # this keeps the legacy path safe and is a no-op when no
                # invalid URLs are present.
                body = await self._verify_section_citations(body, digest, sources, task_id, topic)
                citation_audit_per_section.append(section_audit)

                section_blocks.append({"section_id": section_id, "title": str(section_plan.get("title") or digest.get("title") or section_id), "body": body})
                section_outlines.append(self._build_section_outline_item(section_plan, digest, body))
                ctx.record_timing(task_id, topic, "writer", f"{section_id}_section_writer_total", section_started_at, section_title=str(digest.get("title") or section_id), item_count=len(digest.get("items", [])), output_chars=len(body))
                ctx.save_probe(task_id, topic, "writer", f"{section_id}_section_digest", {"section_digest": digest, "section_body": body})

            # Report frame
            frame_prompt = (
                "Act as the final editor for a deep research report.\n"
                f"Output language: {output_language_name}\nTopic: {topic}\n"
                f"Report title: {plan_data.get('report_title', topic)}\n"
                f"User goal: {plan_data.get('user_goal', '')}\n"
                f"Final response focus: {json.dumps(quality_review.get('final_response_focus', []), ensure_ascii=False)}\n"
                f"Future outlook focus: {json.dumps(quality_review.get('future_outlook_focus', []), ensure_ascii=False)}\n\n"
                "Generate only these framing parts as JSON:\n"
                '{"introduction":"","direct_answer":"","future_outlook":""}\n\n'
                f"Section outlines:\n{json.dumps(section_outlines, ensure_ascii=False, indent=2)}\n"
            )
            frame = await call_llm_json(ctx, frame_prompt, task_id=task_id, topic=topic, stage="writer", name="report_frame")
            if not isinstance(frame, dict):
                frame = {}
            introduction = self._replace_numbered_citations_with_links(str(frame.get("introduction") or "").strip(), sources)
            direct_answer = self._replace_numbered_citations_with_links(str(frame.get("direct_answer") or "").strip(), sources)
            future_outlook = self._replace_numbered_citations_with_links(str(frame.get("future_outlook") or "").strip(), sources)

            report_lines = [f"# {plan_data.get('report_title', topic)}"]
            if introduction:
                report_lines.extend(["", introduction])
            for block in section_blocks:
                report_lines.extend(["", f"## {block['title']}", "", block["body"].strip()])
            if direct_answer:
                report_lines.extend(["", "## Direct Answer to the User's Goal", "", direct_answer])
            if future_outlook:
                report_lines.extend(["", "## Future Outlook and Judgment", "", future_outlook])
            if sources:
                report_lines.extend(["", "## References", ""])
                for src in sources:
                    report_lines.append(f"{src.get('reference_number', '')}. {self._markdown_source_link(src)}")

            report = "\n".join(line for line in report_lines if line is not None).strip() + "\n"
            ctx.record_timing(task_id, topic, "writer", "writer_total", writer_started_at, section_count=len(section_blocks), card_count=len(cards), report_chars=len(report))
            ctx.save_probe(task_id, topic, "writer", "section_outlines", section_outlines)
            ctx.save_probe(task_id, topic, "writer", "report_frame", frame)
        else:
            sources = self._quality._filter_report_sources(source_catalog, [])
            sources_json = json.dumps(sources, ensure_ascii=False, indent=2)
            report = f"# {topic}\n\nNo evidence-grounded research units were available.\n"
            final_section_digests = []
            ctx.record_timing(task_id, topic, "writer", "writer_total", writer_started_at, section_count=0, card_count=0, report_chars=len(report))

        # Save artifacts
        report_path.write_text(report, encoding="utf-8")
        cards_path.write_text(cards_json, encoding="utf-8")
        sources_path.write_text(sources_json, encoding="utf-8")
        ctx.save_probe(task_id, topic, "writer", "final_report", {"report": report})

        completed_at = now_iso()
        # Aggregate the per-section citation audits into a single summary
        # that downstream MCP tool handlers can surface. We collapse the
        # raw failure lists into counts for the top-level summary but
        # keep the detailed per-section list so users can drill in.
        per_section_audit: List[Dict[str, Any]] = locals().get("citation_audit_per_section", []) or []
        agg = {
            "sections_total": len(per_section_audit),
            "sections_grounded": sum(1 for s in per_section_audit if not s.get("writer_fallback")),
            "sections_fallback": sum(1 for s in per_section_audit if s.get("writer_fallback")),
            "citations_total": sum(int(s.get("citations_total") or 0) for s in per_section_audit),
            "invalid_ids_dropped": sum(len(s.get("invalid_ids_dropped") or []) for s in per_section_audit),
            "ungrounded_paragraphs": sum(int(s.get("ungrounded_paragraphs") or 0) for s in per_section_audit),
            "quote_failures": sum(len(s.get("quote_failures") or []) for s in per_section_audit),
            "numeric_failures": sum(len(s.get("numeric_failures") or []) for s in per_section_audit),
            "per_section": per_section_audit,
        }
        final_meta = {
            "task_id": task_id, "topic": topic,
            "task_dir": str(task_dir), "report_path": str(report_path),
            "cards_path": str(cards_path), "sources_path": str(sources_path),
            "activity_path": str(activity_path), "metadata_path": str(metadata_path),
            "quality_review": quality_review,
            "section_digests": final_section_digests if cards else [],
            "citation_audit": agg,
            "lifecycle": "completed", "stage": "completed", "completed_at": completed_at,
        }
        await ctx.store.save_task_meta(task_id, final_meta)
        await ctx.store.append_task_event(task_id, "writer", "Final report package created.", report_path=str(report_path))
        await ctx.store.append_task_event(task_id, "completed", "Task execution finished.")
        await ctx.store.set_status(task_id, "Task completed.", lifecycle="completed", stage="completed", report_path=str(report_path), completed_at=completed_at)

        final_activity = await ctx.store.recent_task_events(task_id, limit=100)
        activity_path.write_text(json.dumps(final_activity, ensure_ascii=False, indent=2), encoding="utf-8")
        final_metadata = await ctx.store.load_task_meta(task_id)
        metadata_path.write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.log_task(task_id, "Writer finished.", stage="completed", report_path=str(report_path))
        # Propagate citation_audit to the graph state so node_writer
        # can record it in ResearchState and tools.py can surface it.
        return {"final_report": report, "citation_audit": agg}
