import json
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from .models import KnowledgeCard, SectionDigest, SectionDigestItem


WEAK_REFERENCE_DOMAINS = frozenset(
    {
        "zhihu.com",
        "www.zhihu.com",
        "zhuanlan.zhihu.com",
        "medium.com",
        "www.medium.com",
        "wiki.mbalib.com",
        "mbalib.com",
        "www.mbalib.com",
        "chinavalue.net",
        "www.chinavalue.net",
        "cnblogs.com",
        "www.cnblogs.com",
        "woshipm.com",
        "www.woshipm.com",
        "showapi.com",
        "www.showapi.com",
        "233.com",
        "www.233.com",
        "fanruan.com",
        "www.fanruan.com",
        "sina.cn",
        "www.sina.cn",
        "sina.com.cn",
        "www.sina.com.cn",
        "k.sina.com.cn",
        "blog.csdn.net",
        "so.csdn.net",
        "gitcode.csdn.net",
        "aicoding.csdn.net",
        "xingyun3d.csdn.net",
        "allconfs.org",
        "www.allconfs.org",
        "ais.cn",
        "www.ais.cn",
        "aistudio.baidu.com",
        "coaio.com",
        "www.coaio.com",
        "linkedin.com",
        "www.linkedin.com",
    }
)
WEAK_REFERENCE_DOMAIN_SUFFIXES = (".medium.com", ".csdn.net", ".sina.cn", ".sina.com.cn")


class QualityMixin:
    @staticmethod
    def _dedupe_text_items(values: List[str], *, limit: int = 0) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for value in values:
            text = re.sub(r"\s+", " ", str(value or "").strip())
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            output.append(text)
            if limit and len(output) >= limit:
                break
        return output

    @staticmethod
    def _clamp_score(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        return max(0.0, min(1.0, round(parsed, 2)))

    @staticmethod
    def _build_source_catalog(cards: List[KnowledgeCard]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for card in cards:
            source = card.get("source", "")
            if not source:
                continue
            domain = (urlparse(source).netloc or "unknown").strip().lower()
            entry = seen.setdefault(
                source,
                {
                    "source_url": source,
                    "source_title": card.get("source_title", ""),
                    "domain": domain,
                    "unit_ids": [],
                },
            )
            if card.get("unit_id") and card["unit_id"] not in entry["unit_ids"]:
                entry["unit_ids"].append(card["unit_id"])
        values = list(seen.values())
        values.sort(
            key=lambda item: (
                -len(item.get("unit_ids", [])),
                str(item.get("source_title") or ""),
                str(item.get("source_url") or ""),
            )
        )
        return values

    @staticmethod
    def _reference_title_is_generic(title: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(title or "").strip()).lower()
        if not normalized:
            return False
        weak_tokens = [
            "conclusion",
            "recommendation",
            "future research",
            "how to write",
            "before everyone else",
        ]
        return sum(token in normalized for token in weak_tokens) >= 2

    @staticmethod
    def _reference_domain_is_weak(domain: str) -> bool:
        normalized = str(domain or "").strip().lower()
        return normalized in WEAK_REFERENCE_DOMAINS or normalized.endswith(WEAK_REFERENCE_DOMAIN_SUFFIXES)

    def _filter_report_sources(
        self,
        sources: List[Dict[str, Any]],
        section_digests: List[SectionDigest],
    ) -> List[Dict[str, Any]]:
        preferred_urls: set[str] = set()
        preferred_domains: set[str] = set()
        for digest in section_digests:
            if not isinstance(digest, dict):
                continue
            for item in digest.get("items", []) if isinstance(digest.get("items"), list) else []:
                if not isinstance(item, dict):
                    continue
                source_url = str(item.get("source_url") or "").strip()
                if source_url:
                    preferred_urls.add(source_url)
                    preferred_domains.add((urlparse(source_url).netloc or "unknown").strip().lower())

        filtered: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_url = str(source.get("source_url") or "").strip()
            source_title = str(source.get("source_title") or "").strip()
            domain = str(source.get("domain") or (urlparse(source_url).netloc if source_url else "unknown")).strip().lower()
            if preferred_urls and source_url and source_url not in preferred_urls and domain not in preferred_domains:
                continue
            if self._reference_title_is_generic(source_title):
                continue
            if self._reference_domain_is_weak(domain):
                continue
            key = source_url or source_title
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            filtered.append({**source, "domain": domain})

        if filtered:
            return filtered

        fallback: List[Dict[str, Any]] = []
        seen_keys.clear()
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_url = str(source.get("source_url") or "").strip()
            source_title = str(source.get("source_title") or "").strip()
            domain = str(source.get("domain") or (urlparse(source_url).netloc if source_url else "unknown")).strip().lower()
            if self._reference_title_is_generic(source_title):
                continue
            key = source_url or source_title
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            fallback.append({**source, "domain": domain})
        return fallback

    @staticmethod
    def _build_reference_index(sources: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
        numbered_sources: List[Dict[str, Any]] = []
        ref_by_url: Dict[str, int] = {}
        ref_by_domain: Dict[str, int] = {}
        for index, source in enumerate(sources, start=1):
            enriched = dict(source)
            enriched["reference_number"] = index
            numbered_sources.append(enriched)
            source_url = str(enriched.get("source_url") or "").strip()
            if source_url and source_url not in ref_by_url:
                ref_by_url[source_url] = index
            domain = str(enriched.get("domain") or "").strip().lower()
            if domain and domain not in ref_by_domain:
                ref_by_domain[domain] = index
        return numbered_sources, ref_by_url, ref_by_domain

    def _extract_keywords(self, text: str) -> List[str]:
        raw = re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", str(text or "").lower())
        stopwords = {
            "what", "when", "where", "which", "that", "this", "with", "from", "into", "about",
            "question", "questions", "current", "latest", "development", "topic",
        }
        seen = []
        for item in raw:
            if item in stopwords:
                continue
            if item not in seen:
                seen.append(item)
        return seen[:8]

    def _question_is_covered(self, question: str, cards: List[KnowledgeCard]) -> bool:
        keywords = self._extract_keywords(question)
        if not keywords:
            return bool(cards)
        for card in cards:
            haystack = " ".join(
                [
                    str(card.get("claim") or ""),
                    str(card.get("evidence_summary") or ""),
                    str(card.get("exact_excerpt") or ""),
                ]
            ).lower()
            matched = sum(1 for keyword in keywords if keyword in haystack)
            if matched >= max(1, min(2, len(keywords))):
                return True
        return False

    @staticmethod
    def _question_prefers_recent(question: str) -> bool:
        lowered = str(question or "").lower()
        return any(token in lowered for token in ("latest", "recent", "new", "today", "this month", "最新", "最近", "本月", "更新", "发布"))

    @staticmethod
    def _question_prefers_metrics(question: str) -> bool:
        lowered = str(question or "").lower()
        return any(token in lowered for token in ("benchmark", "metric", "metrics", "latency", "throughput", "accuracy", "data", "score", "评测", "评估", "性能", "数据", "指标"))

    @staticmethod
    def _question_prefers_counter_evidence(question: str) -> bool:
        lowered = str(question or "").lower()
        return any(token in lowered for token in ("risk", "limitation", "tradeoff", "challenge", "counter", "对比", "限制", "局限", "争议", "风险", "挑战"))

    @staticmethod
    def _question_prefers_definition(question: str) -> bool:
        lowered = str(question or "").lower()
        return any(token in lowered for token in ("what is", "definition", "overview", "concept", "是什么", "定义", "概念", "介绍"))

    @staticmethod
    def _question_prefers_detail(question: str) -> bool:
        lowered = str(question or "").lower()
        return any(token in lowered for token in ("how it works", "mechanism", "implementation", "architecture", "workflow", "原理", "机制", "实现", "架构", "流程"))

    @staticmethod
    def _merge_unique_str_lists(*value_lists: List[str], limit: int = 0) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for values in value_lists:
            for value in values:
                text = re.sub(r"\s+", " ", str(value or "").strip())
                if not text:
                    continue
                lowered = text.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                output.append(text)
                if limit and len(output) >= limit:
                    return output
        return output

    def _build_follow_up_requests(
        self,
        *,
        section: Dict[str, Any],
        missing_questions: List[str],
        gap_types: List[str],
        required_source_types: List[str],
        required_evidence_types: List[str],
        reason: str,
    ) -> List[Dict[str, Any]]:
        fallback_questions = [
            str(item).strip()
            for item in (section.get("query_hints") or section.get("questions") or [])
            if str(item).strip()
        ]
        candidate_questions = self._merge_unique_str_lists(missing_questions, fallback_questions, limit=3)
        if not candidate_questions:
            return []

        requests: List[Dict[str, Any]] = []
        normalized_gap_types = gap_types or ["missing_primary_source"]
        for index, question in enumerate(candidate_questions, start=1):
            gap_type = normalized_gap_types[min(index - 1, len(normalized_gap_types) - 1)]
            requests.append(
                {
                    "query": question,
                    "gap_type": gap_type,
                    "evidence_goal": reason,
                    "required_source_types": required_source_types[:4],
                    "required_evidence_types": required_evidence_types[:4],
                    "time_scope": "month" if self._question_prefers_recent(question) else "",
                }
            )
        return requests

    def _rule_based_section_review(
        self,
        section_id: str,
        title: str,
        section: Dict[str, Any],
        section_cards: List[KnowledgeCard],
    ) -> Dict[str, Any]:
        questions = [str(item).strip() for item in section.get("questions", []) if str(item).strip()] if isinstance(section.get("questions"), list) else []
        if not section_cards:
            missing_questions = questions[:2] or [str(item).strip() for item in section.get("query_hints", []) if str(item).strip()][:2]
            gap_types = ["missing_primary_source"]
            required_source_types = ["primary_source", "documentation", "report"]
            required_evidence_types = ["fact", "definition"]
            return {
                "section_id": section_id,
                "section_title": title,
                "is_enough": False,
                "reason": "This section does not yet have enough evidence-grounded material.",
                "missing_questions": missing_questions,
                "gap_types": gap_types,
                "required_source_types": required_source_types,
                "required_evidence_types": required_evidence_types,
                "follow_up_requests": self._build_follow_up_requests(
                    section=section,
                    missing_questions=missing_questions,
                    gap_types=gap_types,
                    required_source_types=required_source_types,
                    required_evidence_types=required_evidence_types,
                    reason="Collect primary evidence for this section before drafting.",
                ),
                "coverage_score": 0.0,
                "evidence_count_score": 0.0,
                "source_diversity_score": 0.0,
                "primary_source_score": 0.0,
                "claim_type_diversity_score": 0.0,
                "card_count": 0,
            }

        covered_questions = [question for question in questions if self._question_is_covered(question, section_cards)]
        coverage_score = round(len(covered_questions) / max(1, len(questions)), 2) if questions else 1.0
        card_count = len(section_cards)
        unique_sources = {str(card.get("source") or "").strip() for card in section_cards if str(card.get("source") or "").strip()}
        unique_domains = {urlparse(url).netloc.strip().lower() for url in unique_sources if url}
        evidence_count_score = round(min(1.0, card_count / 4.0), 2)
        source_diversity_score = round(min(1.0, len(unique_domains) / 3.0), 2)
        primary_source_count = sum(1 for card in section_cards if str(card.get("source_type") or "").strip().lower() == "primary_source")
        primary_source_score = round(primary_source_count / max(1, card_count), 2)
        claim_type_count = len({str(card.get("claim_type") or "").strip().lower() for card in section_cards if str(card.get("claim_type") or "").strip()})
        claim_type_diversity_score = round(min(1.0, claim_type_count / 4.0), 2)
        missing_questions = [question for question in questions if question not in covered_questions][:3]

        gap_types: List[str] = []
        required_source_types: List[str] = []
        required_evidence_types: List[str] = []

        if coverage_score < (0.7 if questions else 1.0):
            if any(self._question_prefers_definition(question) for question in missing_questions):
                gap_types.append("missing_definition")
                required_source_types.extend(["primary_source", "documentation"])
                required_evidence_types.extend(["definition", "mechanism"])
            elif any(self._question_prefers_detail(question) for question in missing_questions):
                gap_types.append("missing_primary_detail")
                required_source_types.extend(["primary_source", "documentation", "repository"])
                required_evidence_types.extend(["implementation_detail", "workflow", "architecture"])
            else:
                gap_types.append("missing_primary_source")
                required_source_types.extend(["primary_source", "documentation", "paper"])
                required_evidence_types.extend(["fact", "first_hand_statement"])

        if any(self._question_prefers_recent(question) for question in questions) and not any(
            str(card.get("time_scope") or "").strip().lower() in {"current", "recent"} for card in section_cards
        ):
            gap_types.append("missing_recent_update")
            required_source_types.extend(["primary_source", "announcement", "news"])
            required_evidence_types.extend(["recent_change", "release_event"])

        if any(self._question_prefers_metrics(question) for question in questions) and not any(
            str(card.get("claim_type") or "").strip().lower() == "metric" for card in section_cards
        ):
            gap_types.append("missing_quantitative_support")
            required_source_types.extend(["paper", "benchmark", "report"])
            required_evidence_types.extend(["metric", "benchmark_result", "table"])

        if any(self._question_prefers_counter_evidence(question) for question in questions) and not any(
            str(card.get("stance") or "").strip().lower() in {"limitation", "counter"} for card in section_cards
        ):
            gap_types.append("missing_counter_evidence")
            required_source_types.extend(["analysis", "community", "paper"])
            required_evidence_types.extend(["limitation", "tradeoff", "counter_example"])

        if primary_source_score < 0.34:
            gap_types.append("missing_primary_source")
            required_source_types.extend(["primary_source", "documentation"])
            required_evidence_types.extend(["first_hand_statement"])

        is_enough = (
            coverage_score >= (0.7 if questions else 1.0)
            and card_count >= 3
            and len(unique_domains) >= min(2, card_count)
            and primary_source_score >= 0.25
        )
        reason = (
            "The current evidence covers the section questions and has enough independent sources to support drafting."
            if is_enough
            else "The current evidence still lacks coverage or independent source support, so another targeted evidence pass is needed."
        )
        gap_types = self._dedupe_text_items(gap_types, limit=4) if not is_enough else []
        required_source_types = self._dedupe_text_items(required_source_types, limit=4)
        required_evidence_types = self._dedupe_text_items(required_evidence_types, limit=4)
        return {
            "section_id": section_id,
            "section_title": title,
            "is_enough": is_enough,
            "reason": reason,
            "missing_questions": missing_questions,
            "gap_types": gap_types,
            "required_source_types": required_source_types,
            "required_evidence_types": required_evidence_types,
            "follow_up_requests": (
                self._build_follow_up_requests(
                    section=section,
                    missing_questions=missing_questions,
                    gap_types=gap_types,
                    required_source_types=required_source_types,
                    required_evidence_types=required_evidence_types,
                    reason=reason,
                )
                if not is_enough
                else []
            ),
            "coverage_score": coverage_score,
            "evidence_count_score": evidence_count_score,
            "source_diversity_score": source_diversity_score,
            "primary_source_score": primary_source_score,
            "claim_type_diversity_score": claim_type_diversity_score,
            "card_count": card_count,
        }

    def _summarize_cards_for_review(self, section_cards: List[KnowledgeCard]) -> List[Dict[str, Any]]:
        cards_sorted = sorted(section_cards, key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        card_limit = max(4, min(8, self.settings.writer_max_cards_per_section))
        summarized: List[Dict[str, Any]] = []
        for index, card in enumerate(cards_sorted[:card_limit], start=1):
            summarized.append(
                {
                    "card_id": str(card.get("unit_id") or f"CARD-{index:02d}"),
                    "claim": str(card.get("claim") or "").strip(),
                    "evidence_summary": str(card.get("evidence_summary") or "").strip(),
                    "exact_excerpt": re.sub(r"\s+", " ", str(card.get("exact_excerpt") or "").strip())[:280],
                    "source_url": str(card.get("source") or "").strip(),
                    "source_title": str(card.get("source_title") or "").strip(),
                    "source_type": str(card.get("source_type") or "").strip(),
                    "claim_type": str(card.get("claim_type") or "").strip(),
                    "time_scope": str(card.get("time_scope") or "").strip(),
                    "stance": str(card.get("stance") or "").strip(),
                    "evidence_strength": str(card.get("evidence_strength") or "").strip(),
                    "confidence": str(card.get("confidence") or "medium").strip().lower(),
                    "evidence_score": round(float(card.get("evidence_score") or 0.0), 4),
                }
            )
        return summarized

    async def _llm_section_review(
        self,
        *,
        task_id: str,
        topic: str,
        section_id: str,
        title: str,
        section: Dict[str, Any],
        section_cards: List[KnowledgeCard],
        rule_review: Dict[str, Any],
    ) -> Dict[str, Any]:
        questions = self._dedupe_text_items(
            [str(item).strip() for item in section.get("questions", [])] if isinstance(section.get("questions"), list) else [],
            limit=6,
        )
        card_summaries = self._summarize_cards_for_review(section_cards)
        if not card_summaries:
            return {
                "review_available": False,
                "is_semantically_enough": False,
                "semantic_coverage_score": 0.0,
                "support_score": 0.0,
                "conflict_score": 0.0,
                "missing_questions": questions[:3],
                "weak_claims": [],
                "reason": "No evidence cards are available for semantic review.",
            }

        prompt = (
            "You are reviewing whether a research section is truly ready for drafting.\n"
            "You must be conservative and evidence-grounded.\n\n"
            f"Overall topic: {topic}\n"
            f"Section id: {section_id}\n"
            f"Section title: {title}\n"
            f"Section purpose: {str(section.get('purpose') or '').strip()}\n"
            f"Section questions: {json.dumps(questions, ensure_ascii=False)}\n"
            "Rule review summary:\n"
            f"{json.dumps({'is_enough': bool(rule_review.get('is_enough')), 'coverage_score': rule_review.get('coverage_score', 0.0), 'evidence_count_score': rule_review.get('evidence_count_score', 0.0), 'source_diversity_score': rule_review.get('source_diversity_score', 0.0), 'primary_source_score': rule_review.get('primary_source_score', 0.0), 'missing_questions': rule_review.get('missing_questions', []), 'gap_types': rule_review.get('gap_types', [])}, ensure_ascii=False)}\n\n"
            "Evidence cards for this section:\n"
            f"{json.dumps(card_summaries, ensure_ascii=False)}\n\n"
            "Return JSON with this exact shape:\n"
            '{"is_semantically_enough":true,"semantic_coverage_score":0.0,"support_score":0.0,"conflict_score":0.0,"missing_questions":[""],"weak_claims":[""],"gap_types":["missing_primary_source"],"required_source_types":["primary_source"],"required_evidence_types":["fact"],"follow_up_focuses":[""],"reason":""}\n'
            "Review rules:\n"
            "- semantic_coverage_score judges whether the section questions are actually answered, not just keyword-matched.\n"
            "- support_score judges whether the current claims are sufficiently backed by the evidence cards.\n"
            "- conflict_score judges whether the cards contain unresolved conflicts or contradictory conclusions.\n"
            "- missing_questions should contain only the most important unanswered questions.\n"
            "- weak_claims should name claims that seem insufficiently supported.\n"
            "- gap_types should describe what kind of evidence is still missing.\n"
            "- required_source_types should describe what source types are needed next, for example primary_source, documentation, paper, benchmark, report, news, community, analysis.\n"
            "- required_evidence_types should describe what evidence forms are needed next, for example definition, metric, implementation_detail, recent_change, limitation, tradeoff.\n"
            "- follow_up_focuses should be short phrases that clarify what the next search pass should focus on.\n"
            "- If the section is not ready, be explicit and conservative.\n"
            "- Do not invent new facts beyond the evidence cards."
        )

        try:
            payload = await self.call_llm_json(
                prompt,
                task_id=task_id,
                topic=topic,
                stage="reflector",
                name=f"{section_id}_semantic_review_llm",
            )
        except Exception as exc:
            self.log_task(
                task_id,
                "Semantic review fallback to rule-only mode.",
                level="warning",
                stage="reflector",
                section_id=section_id,
                error=str(exc),
            )
            return {
                "review_available": False,
                "is_semantically_enough": bool(rule_review.get("is_enough")),
                "semantic_coverage_score": self._clamp_score(rule_review.get("coverage_score"), default=0.0),
                "support_score": self._clamp_score(rule_review.get("evidence_count_score"), default=0.0),
                "conflict_score": 0.0,
                "missing_questions": self._dedupe_text_items(
                    list(rule_review.get("missing_questions", [])) if isinstance(rule_review.get("missing_questions"), list) else [],
                    limit=3,
                ),
                "weak_claims": [],
                "gap_types": self._dedupe_text_items(
                    list(rule_review.get("gap_types", [])) if isinstance(rule_review.get("gap_types"), list) else [],
                    limit=4,
                ),
                "required_source_types": self._dedupe_text_items(
                    list(rule_review.get("required_source_types", [])) if isinstance(rule_review.get("required_source_types"), list) else [],
                    limit=4,
                ),
                "required_evidence_types": self._dedupe_text_items(
                    list(rule_review.get("required_evidence_types", [])) if isinstance(rule_review.get("required_evidence_types"), list) else [],
                    limit=4,
                ),
                "follow_up_focuses": [],
                "reason": "Semantic review was unavailable, so the system used the rule-based review only.",
            }

        payload = payload if isinstance(payload, dict) else {}
        semantic_coverage_score = self._clamp_score(payload.get("semantic_coverage_score"), default=self._clamp_score(rule_review.get("coverage_score"), default=0.0))
        support_score = self._clamp_score(payload.get("support_score"), default=self._clamp_score(rule_review.get("evidence_count_score"), default=0.0))
        conflict_score = self._clamp_score(payload.get("conflict_score"), default=0.0)
        llm_missing_questions = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("missing_questions", [])] if isinstance(payload.get("missing_questions"), list) else [],
            limit=3,
        )
        weak_claims = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("weak_claims", [])] if isinstance(payload.get("weak_claims"), list) else [],
            limit=4,
        )
        gap_types = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("gap_types", [])] if isinstance(payload.get("gap_types"), list) else [],
            limit=4,
        )
        required_source_types = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("required_source_types", [])] if isinstance(payload.get("required_source_types"), list) else [],
            limit=4,
        )
        required_evidence_types = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("required_evidence_types", [])] if isinstance(payload.get("required_evidence_types"), list) else [],
            limit=4,
        )
        follow_up_focuses = self._dedupe_text_items(
            [str(item).strip() for item in payload.get("follow_up_focuses", [])] if isinstance(payload.get("follow_up_focuses"), list) else [],
            limit=4,
        )
        reason = re.sub(r"\s+", " ", str(payload.get("reason") or "").strip())
        is_semantically_enough = bool(payload.get("is_semantically_enough"))
        if not isinstance(payload.get("is_semantically_enough"), bool):
            is_semantically_enough = semantic_coverage_score >= 0.7 and support_score >= 0.6 and conflict_score <= 0.4 and not llm_missing_questions
        return {
            "review_available": True,
            "is_semantically_enough": is_semantically_enough,
            "semantic_coverage_score": semantic_coverage_score,
            "support_score": support_score,
            "conflict_score": conflict_score,
            "missing_questions": llm_missing_questions,
            "weak_claims": weak_claims,
            "gap_types": gap_types,
            "required_source_types": required_source_types,
            "required_evidence_types": required_evidence_types,
            "follow_up_focuses": follow_up_focuses,
            "reason": reason or "Semantic review completed.",
        }

    def _merge_section_review(
        self,
        section: Dict[str, Any],
        rule_review: Dict[str, Any],
        llm_review: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(rule_review)
        llm_available = bool(llm_review.get("review_available"))
        llm_requires_follow_up = False
        if llm_available:
            llm_requires_follow_up = (
                not bool(llm_review.get("is_semantically_enough"))
                or float(llm_review.get("semantic_coverage_score") or 0.0) < 0.7
                or float(llm_review.get("support_score") or 0.0) < 0.6
                or float(llm_review.get("conflict_score") or 0.0) > 0.4
            )

        final_is_enough = bool(rule_review.get("is_enough"))
        if final_is_enough and llm_requires_follow_up:
            final_is_enough = False

        final_missing_questions = self._dedupe_text_items(
            (
                list(rule_review.get("missing_questions", [])) if isinstance(rule_review.get("missing_questions"), list) else []
            )
            + (
                list(llm_review.get("missing_questions", [])) if isinstance(llm_review.get("missing_questions"), list) else []
            ),
            limit=3,
        )
        final_gap_types = self._dedupe_text_items(
            (list(rule_review.get("gap_types", [])) if isinstance(rule_review.get("gap_types"), list) else [])
            + (list(llm_review.get("gap_types", [])) if isinstance(llm_review.get("gap_types"), list) else []),
            limit=4,
        )
        final_required_source_types = self._dedupe_text_items(
            (list(rule_review.get("required_source_types", [])) if isinstance(rule_review.get("required_source_types"), list) else [])
            + (list(llm_review.get("required_source_types", [])) if isinstance(llm_review.get("required_source_types"), list) else []),
            limit=4,
        )
        final_required_evidence_types = self._dedupe_text_items(
            (list(rule_review.get("required_evidence_types", [])) if isinstance(rule_review.get("required_evidence_types"), list) else [])
            + (list(llm_review.get("required_evidence_types", [])) if isinstance(llm_review.get("required_evidence_types"), list) else []),
            limit=4,
        )
        follow_up_focuses = self._dedupe_text_items(
            list(llm_review.get("follow_up_focuses", [])) if isinstance(llm_review.get("follow_up_focuses"), list) else [],
            limit=4,
        )
        if not final_is_enough and not final_missing_questions:
            fallback_queries = section.get("query_hints") or section.get("questions") or []
            final_missing_questions = self._dedupe_text_items(
                [str(item).strip() for item in fallback_queries if str(item).strip()],
                limit=3,
            )

        final_reason = str(rule_review.get("reason") or "").strip()
        final_decision_source = "rule_review"
        if llm_available and llm_requires_follow_up:
            final_reason = str(llm_review.get("reason") or final_reason).strip()
            final_decision_source = "rule_plus_llm"
        elif llm_available and final_is_enough:
            final_reason = str(llm_review.get("reason") or final_reason).strip()
            final_decision_source = "rule_plus_llm"

        follow_up_requests = (
            self._build_follow_up_requests(
                section=section,
                missing_questions=self._merge_unique_str_lists(final_missing_questions, follow_up_focuses, limit=3),
                gap_types=final_gap_types,
                required_source_types=final_required_source_types,
                required_evidence_types=final_required_evidence_types,
                reason=final_reason,
            )
            if not final_is_enough
            else []
        )

        merged.update(
            {
                "is_enough": final_is_enough,
                "reason": final_reason,
                "missing_questions": final_missing_questions,
                "gap_types": final_gap_types,
                "required_source_types": final_required_source_types,
                "required_evidence_types": final_required_evidence_types,
                "follow_up_requests": follow_up_requests,
                "rule_review": dict(rule_review),
                "llm_review": dict(llm_review),
                "semantic_coverage_score": float(llm_review.get("semantic_coverage_score") or 0.0) if llm_available else None,
                "support_score": float(llm_review.get("support_score") or 0.0) if llm_available else None,
                "conflict_score": float(llm_review.get("conflict_score") or 0.0) if llm_available else None,
                "weak_claims": list(llm_review.get("weak_claims", [])) if isinstance(llm_review.get("weak_claims"), list) else [],
                "final_decision_source": final_decision_source,
            }
        )
        return merged

    @staticmethod
    def _normalize_digest_claim(value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        normalized = re.sub(r"[^\w\u4e00-\u9fff ]+", "", normalized)
        return normalized[:140]

    def _build_section_digest(
        self,
        section: Dict[str, Any],
        section_cards: List[KnowledgeCard],
        section_review: Dict[str, Any],
    ) -> SectionDigest:
        item_limit = max(6, self.settings.section_digest_max_cards_per_section)
        cards_sorted = sorted(section_cards, key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        seen_claims: set[str] = set()
        items: List[SectionDigestItem] = []
        for card in cards_sorted:
            claim_key = self._normalize_digest_claim(str(card.get("claim") or ""))
            if claim_key and claim_key in seen_claims:
                continue
            if claim_key:
                seen_claims.add(claim_key)
            source_url = str(card.get("source") or "").strip()
            items.append(
                {
                    "item_id": f"{section.get('section_id', 'SXX')}-D{len(items) + 1:02d}",
                    "claim": str(card.get("claim") or "").strip(),
                    "evidence_summary": str(card.get("evidence_summary") or "").strip(),
                    "exact_excerpt": re.sub(r"\s+", " ", str(card.get("exact_excerpt") or "").strip())[: max(self.settings.section_digest_excerpt_chars, 120)],
                    "confidence": str(card.get("confidence") or "medium"),
                    "source_title": str(card.get("source_title") or "").strip(),
                    "source_url": source_url,
                    "reference_numbers": [],
                }
            )
            if len(items) >= item_limit:
                break

        key_claims = [str(item.get("claim") or "").strip() for item in items if str(item.get("claim") or "").strip()][:6]
        return {
            "section_id": str(section.get("section_id") or ""),
            "title": str(section.get("title") or ""),
            "purpose": str(section.get("purpose") or ""),
            "questions": list(section.get("questions", [])),
            "coverage_score": float(section_review.get("coverage_score") or 0.0),
            "evidence_count_score": float(section_review.get("evidence_count_score") or 0.0),
            "source_diversity_score": float(section_review.get("source_diversity_score") or 0.0),
            "is_enough": bool(section_review.get("is_enough")),
            "review_reason": str(section_review.get("reason") or "").strip(),
            "missing_questions": list(section_review.get("missing_questions", [])) if isinstance(section_review.get("missing_questions"), list) else [],
            "key_claims": key_claims,
            "items": items,
        }

    def _attach_reference_numbers_to_digest(
        self,
        digest: SectionDigest,
        ref_by_url: Dict[str, int],
        ref_by_domain: Dict[str, int],
    ) -> SectionDigest:
        enriched = dict(digest)
        updated_items: List[SectionDigestItem] = []
        updated_claims: List[str] = []
        for item in digest.get("items", []) if isinstance(digest.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            item_copy = dict(item)
            numbers: List[int] = []
            source_url = str(item_copy.get("source_url") or "").strip()
            if source_url and source_url in ref_by_url:
                numbers.append(ref_by_url[source_url])
            else:
                source_domain = (urlparse(source_url).netloc or "unknown").strip().lower()
                if source_domain and source_domain in ref_by_domain:
                    numbers.append(ref_by_domain[source_domain])
            item_copy["reference_numbers"] = sorted(set(numbers))
            updated_items.append(item_copy)
            citation = "".join(f"[{number}]" for number in item_copy["reference_numbers"])
            claim = str(item_copy.get("claim") or "").strip()
            if claim:
                updated_claims.append(f"{claim} {citation}".strip())
        enriched["items"] = updated_items
        enriched["key_claims"] = updated_claims[:6]
        return enriched

    def _build_section_outline_item(
        self,
        section: Dict[str, Any],
        digest: SectionDigest,
        section_body: str,
    ) -> Dict[str, Any]:
        return {
            "section_id": str(section.get("section_id") or digest.get("section_id") or ""),
            "title": str(section.get("title") or digest.get("title") or ""),
            "purpose": str(section.get("purpose") or digest.get("purpose") or ""),
            "review_reason": str(digest.get("review_reason") or ""),
            "key_findings": list(digest.get("key_claims", []))[:4],
            "section_summary": re.sub(r"\s+", " ", section_body.strip())[: self.settings.writer_outline_summary_chars],
            "section_body_preview": section_body[:600],
        }
