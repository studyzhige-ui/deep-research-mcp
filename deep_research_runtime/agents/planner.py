"""Planner agent: turns user intent into research execution strategy and outlines."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..models import Document, EvidenceOutline, ReconnaissanceResult, ResearchExecutionPlan, SectionPlan, SubTask
from .base import AgentContext, call_llm_json, pick_text


# ── Domain Constants ──

DEFAULT_EXCLUDE_DOMAINS = [
    "medium.com",
    "dev.to",
    "csdn.net",
    "juejin.cn",
    "cnblogs.com",
    "geeksforgeeks.org",
    "w3schools.com",
]

RECENT_TERMS = (
    "latest", "recent", "new", "today", "this month", "this week",
    "announcement", "release", "update", "breaking", "latest progress",
    "recent progress", "最新", "最近", "本月", "本周", "发布", "更新", "进展", "财报",
)
COMPARISON_TERMS = ("vs", "versus", "compare", "comparison", "difference", "tradeoff", "区别", "对比", "差异")
QUANT_TERMS = (
    "benchmark", "metrics", "latency", "throughput", "accuracy", "score",
    "evaluate", "evaluation", "experiment", "performance", "data",
    "指标", "性能", "数据", "评测", "评估", "实验",
)
PRIMARY_DETAIL_TERMS = (
    "mechanism", "design", "workflow", "implementation", "architecture",
    "how it works", "integration", "原理", "机制", "流程", "架构", "实现", "设计", "接入",
)
DEFINITION_TERMS = (
    "what is", "overview", "introduction", "basic concept", "definition",
    "概念", "定义", "背景", "介绍", "是什么",
)
COUNTER_EVIDENCE_TERMS = (
    "risk", "limitation", "challenge", "tradeoff", "drawback",
    "criticism", "controversy", "风险", "限制", "局限", "挑战", "争议", "取舍",
)
QUERY_NOISE_PATTERNS = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bhow to\b", r"\bhow does\b",
    r"\bplease\b", r"\bexplain\b", r"\btell me\b", r"\bcan you\b",
    r"\bi want to know\b", r"\u4ec0\u4e48", r"\u600e\u4e48",
    r"\u8bf7\u95ee", r"\u544a\u8bc9\u6211", r"\u4ecb\u7ecd\u4e00\u4e0b",
]
VERSION_PATTERN = re.compile(r"\bv?\d+(?:\.\d+){0,2}\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"(20\d{2}(?:[-/年](?:0?[1-9]|1[0-2]))?(?:[-/月](?:0?[1-9]|[12]\d|3[01]))?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d{2}|"
    r"20\d{2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)",
    re.IGNORECASE,
)
VALID_GAP_TYPES = {
    "missing_definition", "missing_primary_source", "missing_recent_update",
    "missing_quantitative_support", "missing_counter_evidence",
    "missing_entity_coverage", "missing_primary_detail",
}
VALID_SEARCH_GOALS = {
    "broad_scan", "definition_clarification", "primary_detail",
    "recent_update", "quant_support", "counter_evidence", "comparison_fill",
}


class PlannerAgent:
    """Handles research plan creation, query rewriting, and search profile generation."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    @property
    def settings(self):
        return self.ctx.settings

    # ── Execution Plan Drafting ──

    async def generate_seed_queries(
        self,
        *,
        topic: str,
        background_intent: str,
        output_language: str,
        task_id: str = "",
    ) -> List[str]:
        prompt = (
            "You are preparing a light reconnaissance search for a deep research task.\n"
            f"Topic: {topic}\nUser intent and constraints: {background_intent}\n\n"
            "Return JSON only:\n"
            '{"queries":[""]}\n'
            "Rules:\n"
            f"- Return {max(1, self.settings.draft_max_queries)} concise web-search queries.\n"
            "- Queries should discover the source landscape, not pre-write a report outline.\n"
            "- Include academic terms only when the user asks for papers, journals, conferences, citations, or literature.\n"
        )
        try:
            payload = await call_llm_json(
                self.ctx, prompt, role="planner", task_id=task_id,
                topic=topic, stage="draft", name="seed_queries",
            )
            queries = payload.get("queries") if isinstance(payload, dict) else []
        except Exception:
            queries = []
        cleaned = self._dedupe_text_values([str(q).strip() for q in queries if str(q).strip()])
        if cleaned:
            return cleaned[: max(1, self.settings.draft_max_queries)]
        return self._fallback_seed_queries(topic, background_intent)[: max(1, self.settings.draft_max_queries)]

    @staticmethod
    def _fallback_seed_queries(topic: str, background_intent: str) -> List[str]:
        text = f"{topic} {background_intent}".strip()
        if not text:
            return ["latest research overview"]
        if re.search(r"论文|文献|学术|paper|papers|journal|conference|ieee|acm", text, re.I):
            return [text, f"{topic} survey papers", f"{topic} recent research"]
        return [text, f"{topic} overview", f"{topic} latest developments"]

    def build_reconnaissance_result(
        self,
        *,
        seed_queries: List[str],
        documents: List[Document],
    ) -> ReconnaissanceResult:
        source_counts: Dict[str, int] = {}
        layers: Dict[str, int] = {}
        for document in documents:
            source = str(document.get("source_name") or "unknown")
            layer = str(document.get("source_layer") or "general")
            source_counts[source] = source_counts.get(source, 0) + 1
            layers[layer] = layers.get(layer, 0) + 1
        landscape = [f"{name}: {count}" for name, count in sorted(source_counts.items())]
        if layers:
            landscape.insert(0, "layers: " + ", ".join(f"{k}={v}" for k, v in sorted(layers.items())))
        ambiguities = []
        if not documents:
            ambiguities.append("The reconnaissance search returned no usable documents; the formal phase should broaden queries and sources.")
        return {
            "seed_queries": seed_queries,
            "documents": documents,
            "source_landscape": landscape,
            "ambiguities": ambiguities,
            "suggested_strategy": [
                "Start with broad web discovery, then add vertical search only for domain-specific evidence needs.",
                "Extract claims only from clickable source URLs and preserve source provenance for every citation.",
            ],
        }

    async def draft_execution_plan(
        self,
        *,
        topic: str,
        background_intent: str,
        reconnaissance: ReconnaissanceResult,
        output_language: str,
        task_id: str = "",
    ) -> ResearchExecutionPlan:
        docs_brief = []
        for doc in reconnaissance.get("documents", [])[:8]:
            docs_brief.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source_name": doc.get("source_name", ""),
                "source_layer": doc.get("source_layer", ""),
                "source_kind": doc.get("source_kind", ""),
                "year": doc.get("year"),
            })
        language = "Chinese" if output_language == "zh" else "English"
        prompt = (
            "You are drafting a human-confirmable execution strategy for deep research.\n"
            "Do NOT produce a report outline. Produce a plan for how to research.\n\n"
            f"Topic: {topic}\nUser intent and constraints: {background_intent}\n"
            f"Reconnaissance seed queries: {reconnaissance.get('seed_queries', [])}\n"
            f"Reconnaissance source landscape: {reconnaissance.get('source_landscape', [])}\n"
            f"Sample discovered documents: {docs_brief}\n\n"
            "Return JSON only:\n"
            '{"task_type":"","user_goal":"","scope":"","source_strategy":[{"layer":"general|vertical","name":"","when_to_use":"","reason":""}],"query_strategy":[{"research_goal":"","queries":[""],"verticals":[""],"priority":"high|medium|low","source_types":[""],"extraction_fields":[""]}],"screening_rules":[""],"extraction_schema":[""],"quality_rules":[""],"expected_deliverable":"","reconnaissance_summary":""}\n'
            "Rules:\n"
            "- User-facing strings must be in "
            f"{language}.\n"
            "- source_strategy must describe a two-layer strategy: general web search first, vertical search only when needed.\n"
            "- query_strategy items are executable search tasks, not report sections.\n"
            "- Every citation in the final report must be clickable and traceable to a source URL.\n"
        )
        try:
            payload = await call_llm_json(
                self.ctx, prompt, role="planner", task_id=task_id,
                topic=topic, stage="draft", name="execution_plan",
            )
        except Exception:
            payload = {}
        return self.normalize_execution_plan(payload, topic=topic, background_intent=background_intent, reconnaissance=reconnaissance, output_language=output_language)

    def normalize_execution_plan(
        self,
        payload: Any,
        *,
        topic: str,
        background_intent: str,
        reconnaissance: ReconnaissanceResult | None = None,
        output_language: str = "en",
    ) -> ResearchExecutionPlan:
        if not isinstance(payload, dict):
            payload = {}
        query_strategy_raw = payload.get("query_strategy") if isinstance(payload.get("query_strategy"), list) else []
        query_strategy: List[Dict[str, Any]] = []
        for index, item in enumerate(query_strategy_raw, start=1):
            if not isinstance(item, dict):
                continue
            queries = item.get("queries") if isinstance(item.get("queries"), list) else []
            cleaned_queries = self._dedupe_text_values([str(q).strip() for q in queries if str(q).strip()])
            if not cleaned_queries:
                continue
            verticals = item.get("verticals") if isinstance(item.get("verticals"), list) else []
            source_types = item.get("source_types") if isinstance(item.get("source_types"), list) else []
            extraction_fields = item.get("extraction_fields") if isinstance(item.get("extraction_fields"), list) else []
            query_strategy.append({
                "task_id": f"Q{index:02d}",
                "research_goal": str(item.get("research_goal") or cleaned_queries[0]).strip(),
                "queries": cleaned_queries[: self.settings.get_search_profile(str(item.get("priority") or "medium")).get("max_queries", self.settings.max_queries_per_intent)],
                "verticals": self._dedupe_text_values([str(v).strip().lower() for v in verticals if str(v).strip()]),
                "priority": str(item.get("priority") or "medium").strip() or "medium",
                "source_types": self._dedupe_text_values([str(v).strip() for v in source_types if str(v).strip()]),
                "extraction_fields": self._dedupe_text_values([str(v).strip() for v in extraction_fields if str(v).strip()]),
            })

        if not query_strategy:
            seed_queries = list((reconnaissance or {}).get("seed_queries", [])) if isinstance(reconnaissance, dict) else []
            if not seed_queries:
                seed_queries = self._fallback_seed_queries(topic, background_intent)
            verticals = ["academic"] if re.search(r"论文|文献|学术|paper|papers|journal|conference|ieee|acm", f"{topic} {background_intent}", re.I) else []
            query_strategy = [{
                "task_id": "Q01",
                "research_goal": background_intent or topic,
                "queries": seed_queries[: self.settings.max_queries_per_intent],
                "verticals": verticals,
                "priority": "high",
                "source_types": ["paper"] if verticals else [],
                "extraction_fields": ["title", "source", "claim", "evidence", "date"],
            }]

        source_strategy = payload.get("source_strategy") if isinstance(payload.get("source_strategy"), list) else []
        if not source_strategy:
            source_strategy = [
                {"layer": "general", "name": "general_search", "when_to_use": "always", "reason": "广覆盖发现网页、报告和官方资料。"},
                {"layer": "vertical", "name": "academic", "when_to_use": "when scholarly papers or literature are requested", "reason": "补充论文和学术元数据。"},
            ]

        return {
            "task_type": str(payload.get("task_type") or "deep_research").strip(),
            "user_goal": str(payload.get("user_goal") or background_intent or topic).strip(),
            "scope": str(payload.get("scope") or topic).strip(),
            "source_strategy": source_strategy,
            "query_strategy": query_strategy,
            "screening_rules": self._dedupe_text_values([str(v).strip() for v in payload.get("screening_rules", [])]) if isinstance(payload.get("screening_rules"), list) else [],
            "extraction_schema": self._dedupe_text_values([str(v).strip() for v in payload.get("extraction_schema", [])]) if isinstance(payload.get("extraction_schema"), list) else ["claim", "evidence", "source_url", "source_title"],
            "quality_rules": self._dedupe_text_values([str(v).strip() for v in payload.get("quality_rules", [])]) if isinstance(payload.get("quality_rules"), list) else ["Every claim must cite a clickable source URL."],
            "expected_deliverable": str(payload.get("expected_deliverable") or "Evidence-grounded research report").strip(),
            "output_language": output_language,
            "reconnaissance_summary": str(payload.get("reconnaissance_summary") or "").strip(),
        }

    @staticmethod
    def render_execution_plan(plan: ResearchExecutionPlan, reconnaissance: ReconnaissanceResult | None = None, output_language: str = "en") -> str:
        zh = output_language == "zh"
        lines = [
            "调研执行计划" if zh else "Research Execution Plan",
            "",
            ("研究目标：" if zh else "Research goal: ") + str(plan.get("user_goal", "")),
            ("研究范围：" if zh else "Scope: ") + str(plan.get("scope", "")),
        ]
        if reconnaissance:
            lines.extend(["", "初步侦察结果：" if zh else "Reconnaissance findings:"])
            for item in reconnaissance.get("source_landscape", [])[:6]:
                lines.append(f"- {item}")
            for item in reconnaissance.get("ambiguities", [])[:4]:
                lines.append(f"- {item}")
        lines.extend(["", "搜索路径：" if zh else "Search path:"])
        for item in plan.get("source_strategy", []):
            if isinstance(item, dict):
                lines.append(f"- [{item.get('layer', '')}] {item.get('name', '')}: {item.get('reason', '')}")
        lines.extend(["", "执行查询策略：" if zh else "Executable query strategy:"])
        for index, item in enumerate(plan.get("query_strategy", []), start=1):
            if not isinstance(item, dict):
                continue
            verticals = ", ".join(item.get("verticals", [])) if isinstance(item.get("verticals"), list) else ""
            lines.append(f"{index}. {item.get('research_goal', '')}")
            queries = item.get("queries") if isinstance(item.get("queries"), list) else []
            for query in queries:
                lines.append(f"   - {query}")
            if verticals:
                lines.append(f"   - vertical: {verticals}")
        if plan.get("screening_rules"):
            lines.extend(["", "筛选标准：" if zh else "Screening rules:"])
            lines.extend(f"- {item}" for item in plan.get("screening_rules", []))
        if plan.get("extraction_schema"):
            lines.extend(["", "信息抽取字段：" if zh else "Extraction fields:"])
            lines.extend(f"- {item}" for item in plan.get("extraction_schema", []))
        if plan.get("quality_rules"):
            lines.extend(["", "质量规则：" if zh else "Quality rules:"])
            lines.extend(f"- {item}" for item in plan.get("quality_rules", []))
        return "\n".join(lines).strip()

    def build_research_subtasks(self, execution_plan: ResearchExecutionPlan) -> List[SubTask]:
        sub_tasks: List[SubTask] = []
        seen = set()
        for order, item in enumerate(execution_plan.get("query_strategy", []), start=1):
            if not isinstance(item, dict):
                continue
            track_id = str(item.get("task_id") or f"Q{order:02d}")
            goal = str(item.get("research_goal") or f"Research track {order}")
            priority = str(item.get("priority") or "medium")
            queries = item.get("queries") if isinstance(item.get("queries"), list) else []
            for query in queries:
                query = str(query).strip()
                key = (track_id, query)
                if not query or key in seen:
                    continue
                seen.add(key)
                sub_tasks.append({
                    "query": query,
                    "intent": goal,
                    "section_id": track_id,
                    "section_title": goal,
                    "section_goal": goal,
                    "section_order": order,
                    "priority": priority,
                    "source_strategy": list(item.get("verticals", [])) if isinstance(item.get("verticals"), list) else [],
                    "source_types": list(item.get("source_types", [])) if isinstance(item.get("source_types"), list) else [],
                    "extraction_fields": list(item.get("extraction_fields", [])) if isinstance(item.get("extraction_fields"), list) else [],
                    "status": "pending",
                })
        return sub_tasks

    @staticmethod
    def research_tracks_as_sections(execution_plan: ResearchExecutionPlan) -> List[SectionPlan]:
        sections: List[SectionPlan] = []
        for order, item in enumerate(execution_plan.get("query_strategy", []), start=1):
            if not isinstance(item, dict):
                continue
            track_id = str(item.get("task_id") or f"Q{order:02d}")
            goal = str(item.get("research_goal") or f"Research track {order}")
            sections.append({
                "section_id": track_id,
                "title": goal,
                "purpose": goal,
                "priority": str(item.get("priority") or "medium"),
                "questions": [str(q).strip() for q in item.get("queries", []) if str(q).strip()] if isinstance(item.get("queries"), list) else [goal],
                "query_hints": [str(q).strip() for q in item.get("queries", []) if str(q).strip()] if isinstance(item.get("queries"), list) else [],
                "depends_on": [],
                "evidence_digest_ids": [track_id],
                "evidence_requirements": [
                    "Use the most relevant high-confidence evidence cards from this evidence digest.",
                    "Prefer cards with exact excerpts and traceable source URLs.",
                ],
            })
        return sections

    async def build_evidence_outline(
        self,
        *,
        topic: str,
        execution_plan: ResearchExecutionPlan,
        section_digests: List[Dict[str, Any]],
        output_language: str,
        task_id: str = "",
    ) -> EvidenceOutline:
        if not section_digests:
            return self._fallback_evidence_outline(topic, execution_plan, output_language, section_digests=[])
        digest_summaries = [
            {
                "section_id": d.get("section_id", ""),
                "title": d.get("title", ""),
                "purpose": d.get("purpose", ""),
                "questions": d.get("questions", [])[:4] if isinstance(d.get("questions"), list) else [],
                "coverage_score": d.get("coverage_score", 0),
                "evidence_count_score": d.get("evidence_count_score", 0),
                "source_diversity_score": d.get("source_diversity_score", 0),
                "is_enough": bool(d.get("is_enough")),
                "key_claims": d.get("key_claims", [])[:5] if isinstance(d.get("key_claims"), list) else [],
                "evidence_items": [
                    {
                        "claim": item.get("claim", ""),
                        "confidence": item.get("confidence", "medium"),
                        "source_title": item.get("source_title", ""),
                    }
                    for item in (d.get("items", []) if isinstance(d.get("items"), list) else [])[:5]
                    if isinstance(item, dict)
                ],
            }
            for d in section_digests
            if isinstance(d, dict)
        ]
        prompt = (
            "Build the final report outline only from compressed section evidence digests.\n"
            f"Topic: {topic}\nUser goal: {execution_plan.get('user_goal', '')}\n"
            f"Available section evidence digests: {digest_summaries}\n\n"
            "Return JSON only:\n"
            '{"report_title":"","sections":[{"section_id":"S01","title":"","purpose":"","priority":"high|medium|low","questions":[],"query_hints":[],"depends_on":[],"evidence_digest_ids":["Q01"],"evidence_requirements":[""]}],"outline_notes":""}\n'
            "Rules:\n"
            "- This is the report outline, generated AFTER evidence collection.\n"
            "- Use only the compressed evidence digests above; do not ask for raw KnowledgeCards.\n"
            "- Every report section must cite one or more existing evidence_digest_ids from the available section_id values.\n"
            "- evidence_requirements should describe what kind of evidence Writer should prioritize, not exact card IDs.\n"
            "- Good evidence_requirements mention source type, metric, comparison, time scope, uncertainty, or exact excerpt needs.\n"
            "- Do not invent evidence_digest_ids.\n"
            f"- Use {'Chinese' if output_language == 'zh' else 'English'}."
        )
        try:
            payload = await call_llm_json(self.ctx, prompt, role="planner", task_id=task_id, topic=topic, stage="outline_builder", name="evidence_outline")
        except Exception:
            payload = {}
        outline = self._normalize_evidence_outline(payload, topic, execution_plan, section_digests, output_language)
        return outline

    def _normalize_evidence_outline(self, payload: Any, topic: str, execution_plan: ResearchExecutionPlan, section_digests: List[Dict[str, Any]], output_language: str) -> EvidenceOutline:
        normalized = self.normalize_section_payload(payload if isinstance(payload, dict) else {}, output_language=output_language)
        if not normalized.get("sections"):
            return self._fallback_evidence_outline(topic, execution_plan, output_language, section_digests=section_digests)
        digest_ids = {str(digest.get("section_id") or "") for digest in section_digests if str(digest.get("section_id") or "")}
        valid_sections: List[SectionPlan] = []
        for section in normalized.get("sections", []):
            evidence_ids = [str(v).strip() for v in section.get("evidence_digest_ids", []) if str(v).strip()] if isinstance(section.get("evidence_digest_ids"), list) else []
            if not evidence_ids and str(section.get("section_id") or "") in digest_ids:
                evidence_ids = [str(section.get("section_id"))]
            evidence_ids = [value for value in dict.fromkeys(evidence_ids) if value in digest_ids]
            if not evidence_ids:
                continue
            item = dict(section)
            item["evidence_digest_ids"] = evidence_ids
            requirements = item.get("evidence_requirements") if isinstance(item.get("evidence_requirements"), list) else []
            item["evidence_requirements"] = [str(value).strip() for value in requirements if str(value).strip()]
            valid_sections.append(item)
        if not valid_sections:
            return self._fallback_evidence_outline(topic, execution_plan, output_language, section_digests=section_digests)
        return {
            "report_title": str(normalized.get("report_title") or topic),
            "user_goal": str(execution_plan.get("user_goal") or topic),
            "sections": valid_sections,
            "outline_notes": str((payload or {}).get("outline_notes") or ""),
        }

    def _fallback_evidence_outline(self, topic: str, execution_plan: ResearchExecutionPlan, output_language: str, *, section_digests: List[Dict[str, Any]] | None = None) -> EvidenceOutline:
        digest_sections: List[SectionPlan] = []
        for order, digest in enumerate(section_digests or [], start=1):
            if not isinstance(digest, dict):
                continue
            digest_id = str(digest.get("section_id") or f"Q{order:02d}").strip() or f"Q{order:02d}"
            title = str(digest.get("title") or "").strip() or pick_text(output_language, f"证据主题 {order}", f"Evidence theme {order}")
            digest_sections.append({
                "section_id": digest_id,
                "title": title,
                "purpose": str(digest.get("purpose") or title),
                "priority": "high" if bool(digest.get("is_enough")) else "medium",
                "questions": [str(v).strip() for v in digest.get("questions", []) if str(v).strip()] if isinstance(digest.get("questions"), list) else [title],
                "query_hints": [],
                "depends_on": [],
                "evidence_digest_ids": [digest_id],
                "evidence_requirements": [
                    pick_text(output_language, "优先使用高置信度、带原文摘录和可点击来源的知识卡片。", "Prefer high-confidence cards with exact excerpts and clickable sources."),
                    pick_text(output_language, "覆盖该证据包中的主要观点，同时保留重要不确定性。", "Cover the main claims in this digest while preserving important uncertainty."),
                ],
            })
        track_sections = self.research_tracks_as_sections(execution_plan)
        if digest_sections:
            sections = digest_sections
        elif track_sections:
            sections = track_sections
        else:
            sections = [{
                "section_id": "S01",
                "title": pick_text(output_language, "主要发现", "Main findings"),
                "purpose": str(execution_plan.get("user_goal") or topic),
                "priority": "high",
                "questions": [str(execution_plan.get("user_goal") or topic)],
                "query_hints": [],
                "depends_on": [],
                "evidence_digest_ids": ["S01"],
                "evidence_requirements": [
                    pick_text(output_language, "优先使用高置信度、带原文摘录和可点击来源的知识卡片。", "Prefer high-confidence cards with exact excerpts and clickable sources."),
                ],
            }]
        return {
            "report_title": topic,
            "user_goal": str(execution_plan.get("user_goal") or topic),
            "sections": sections,
            "outline_notes": "Fallback evidence outline built from research tracks.",
        }

    # ── Plan Normalization ──

    @staticmethod
    def normalize_section_payload(payload: Any, output_language: str = "en") -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}

        task_type = str(payload.get("task_type") or "broad_exploration").strip() or "broad_exploration"
        user_goal = str(payload.get("user_goal") or "").strip()
        report_title = str(payload.get("report_title") or "").strip() or f"{user_goal or 'Research'} Report"

        sections_raw = payload.get("sections") if isinstance(payload.get("sections"), list) else []
        sections: List[SectionPlan] = []
        for index, item in enumerate(sections_raw, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            questions = item.get("questions") if isinstance(item.get("questions"), list) else []
            query_hints = item.get("query_hints") if isinstance(item.get("query_hints"), list) else []
            depends_on = item.get("depends_on") if isinstance(item.get("depends_on"), list) else []
            evidence_digest_ids_raw = item.get("evidence_digest_ids")
            if not isinstance(evidence_digest_ids_raw, list):
                evidence_digest_ids_raw = item.get("source_digest_ids") if isinstance(item.get("source_digest_ids"), list) else []
            evidence_requirements_raw = item.get("evidence_requirements")
            if not isinstance(evidence_requirements_raw, list):
                evidence_requirements_raw = item.get("evidence_needs") if isinstance(item.get("evidence_needs"), list) else []
            section_id = str(item.get("section_id") or f"S{index:02d}").strip() or f"S{index:02d}"
            sections.append({
                "section_id": section_id,
                "title": title,
                "purpose": str(item.get("purpose") or "").strip(),
                "priority": str(item.get("priority") or "medium").strip() or "medium",
                "questions": [str(v).strip() for v in questions if str(v).strip()],
                "query_hints": [str(v).strip() for v in query_hints if str(v).strip()],
                "depends_on": [str(v).strip() for v in depends_on if str(v).strip()],
                "evidence_digest_ids": [str(v).strip() for v in evidence_digest_ids_raw if str(v).strip()],
                "evidence_requirements": [str(v).strip() for v in evidence_requirements_raw if str(v).strip()],
            })

        if not sections:
            sections = [
                {
                    "section_id": "S01",
                    "title": "Basic concepts and current landscape",
                    "purpose": "Give the reader enough background to understand the topic.",
                    "priority": "high",
                    "questions": ["What basic concepts and background should be understood first?"],
                    "query_hints": [], "depends_on": [], "evidence_digest_ids": [], "evidence_requirements": [],
                },
                {
                    "section_id": "S02",
                    "title": "Main research question",
                    "purpose": "Directly address the user's primary need.",
                    "priority": "high",
                    "questions": [user_goal or "What is the main problem the user wants solved?"],
                    "query_hints": [], "depends_on": ["S01"], "evidence_digest_ids": [], "evidence_requirements": [],
                },
                {
                    "section_id": "S03",
                    "title": "Limits, risks, and forward-looking judgment",
                    "purpose": "Summarize limitations, uncertainty, and future direction.",
                    "priority": "medium",
                    "questions": ["What are the limits, risks, and likely next steps?"],
                    "query_hints": [], "depends_on": ["S02"], "evidence_digest_ids": [], "evidence_requirements": [],
                },
            ]

        return {
            "task_type": task_type,
            "user_goal": user_goal,
            "report_title": report_title,
            "sections": sections,
            "final_response_goal": str(payload.get("final_response_goal") or user_goal).strip(),
            "planner_notes": str(payload.get("planner_notes") or "").strip(),
            "output_language": output_language,
        }

    @staticmethod
    def render_user_plan(plan_data: Dict[str, Any], output_language: str = "en") -> str:
        title = plan_data.get("report_title", "Research Report")
        lines = [title, "Research Plan"]
        for index, section in enumerate(plan_data.get("sections", []), start=1):
            lines.append(f"({index}) {section.get('title', '')}")
            purpose = section.get("purpose", "")
            if purpose:
                lines.append(f"Goal: {purpose}")
            questions = section.get("questions", [])
            if len(questions) == 1:
                lines.append(f"Key question: {questions[0]}")
            elif questions:
                for offset, question in enumerate(questions, start=1):
                    marker = chr(ord("a") + offset - 1)
                    lines.append(f"({marker}) {question}")
        return "\n".join(lines)

    # ── Sub-task Generation ──

    def build_section_subtasks(self, plan_data: Dict[str, Any]) -> List[SubTask]:
        sub_tasks: List[SubTask] = []
        seen = set()
        for order, section in enumerate(plan_data.get("sections", []), start=1):
            section_id = str(section.get("section_id") or f"S{order:02d}")
            title = str(section.get("title") or f"Section {order}")
            purpose = str(section.get("purpose") or "").strip()
            priority = str(section.get("priority") or "medium").strip()
            hints = section.get("query_hints") if isinstance(section.get("query_hints"), list) else []
            questions = section.get("questions") if isinstance(section.get("questions"), list) else []
            candidates = [str(v).strip() for v in hints + questions if str(v).strip()]
            if not candidates:
                candidates = [title]

            # Include perspective queries if available (Improvement 5)
            perspective_queries = section.get("perspective_queries") if isinstance(section.get("perspective_queries"), list) else []
            candidates.extend([str(q).strip() for q in perspective_queries if str(q).strip()])

            # Adaptive search depth: use priority to determine query limit
            profile = self.settings.get_search_profile(priority)
            max_queries = profile.get("max_queries", self.settings.max_queries_per_intent)

            for query in candidates[:max_queries]:
                key = (section_id, query)
                if key in seen:
                    continue
                seen.add(key)
                sub_tasks.append({
                    "query": query,
                    "intent": title,
                    "section_id": section_id,
                    "section_title": title,
                    "section_goal": purpose,
                    "section_order": order,
                    "status": "pending",
                })
        return sub_tasks

    # ── Multi-Perspective Discovery (Improvement 5: STORM-style) ──

    async def inject_perspectives(
        self,
        plan_data: Dict[str, Any],
        *,
        topic: str = "",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Discover expert perspectives and inject supplemental queries into sections.

        For each high/medium priority section, asks the LLM to propose 2-3
        expert viewpoints (e.g., industry practitioner, researcher, critic)
        and one concrete search query per perspective.

        These queries are appended to the section's `perspective_queries` list,
        which build_section_subtasks will pick up alongside normal queries.

        Returns the mutated plan_data with perspective_queries injected.
        """
        sections = plan_data.get("sections", [])
        if not sections:
            return plan_data

        # Only generate perspectives for high/medium priority sections
        eligible = [
            s for s in sections
            if str(s.get("priority", "medium")).strip() in {"high", "medium"}
        ]
        if not eligible:
            return plan_data

        section_briefs = "\n".join(
            f"  {s.get('section_id', '')}: {s.get('title', '')} — {s.get('purpose', '')}"
            for s in eligible
        )

        prompt = (
            "You are a research perspective advisor.\n\n"
            f"Topic: {topic}\n\n"
            "The following sections will be researched:\n"
            f"{section_briefs}\n\n"
            "For EACH section, propose 2-3 expert perspectives that would enrich the research.\n"
            "Each perspective should be a different stakeholder, critic, or domain expert\n"
            "who would ask a different question about the same topic.\n\n"
            "Examples of good perspectives:\n"
            "- An industry practitioner asking about real-world deployment challenges\n"
            "- An academic researcher asking about theoretical foundations\n"
            "- A policy maker asking about regulatory implications\n"
            "- A critic asking about risks and limitations\n"
            "- A user/consumer asking about practical impact\n\n"
            "Return JSON:\n"
            '{"sections":[{"section_id":"S01","perspectives":[{"role":"industry practitioner","query":"concrete search query from this perspective"}]}]}\n\n'
            "Rules:\n"
            "- Each query must be specific and searchable (not vague).\n"
            "- Queries should NOT overlap with the section's existing questions.\n"
            "- Keep queries concise (under 15 words).\n"
            "- Use the same language as the section titles.\n"
        )

        try:
            result = await call_llm_json(
                self.ctx, prompt, role="planner",
                task_id=task_id, topic=topic,
                stage="supervisor", name="perspective_discovery",
            )
        except Exception:
            # If perspective discovery fails, proceed without it
            return plan_data

        if not isinstance(result, dict):
            return plan_data

        # Merge perspective queries into plan sections
        perspective_sections = result.get("sections", [])
        if not isinstance(perspective_sections, list):
            return plan_data

        section_map = {str(s.get("section_id", "")): s for s in sections}

        for ps in perspective_sections:
            if not isinstance(ps, dict):
                continue
            sid = str(ps.get("section_id", ""))
            section = section_map.get(sid)
            if not section:
                continue

            perspectives = ps.get("perspectives", [])
            if not isinstance(perspectives, list):
                continue

            queries = []
            for p in perspectives:
                if not isinstance(p, dict):
                    continue
                query = str(p.get("query", "")).strip()
                role = str(p.get("role", "")).strip()
                if query:
                    queries.append(query)

            if queries:
                existing = section.get("perspective_queries", [])
                if not isinstance(existing, list):
                    existing = []
                section["perspective_queries"] = existing + queries

        return plan_data

    # ── Query Planning Utilities ──

    @staticmethod
    def _dedupe_text_values(values: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in values:
            cleaned = str(item or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _dedupe_limited(self, values: List[str], limit: int) -> List[str]:
        return self._dedupe_text_values(values)[:max(1, limit)]

    @staticmethod
    def _strip_query_surface(text: str) -> str:
        cleaned = str(text or "").strip()
        for pattern in QUERY_NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\"'`]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:!?")
        return cleaned

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}|[\u4e00-\u9fff]{2,}", str(text or ""))

    @staticmethod
    def _detect_time_scope(text: str) -> str:
        lowered = str(text or "").lower()
        if "today" in lowered or "今日" in lowered or "今天" in lowered:
            return "day"
        if "this week" in lowered or "本周" in lowered:
            return "week"
        if any(token in lowered for token in RECENT_TERMS) or DATE_PATTERN.search(text):
            return "month"
        if "year" in lowered or "年度" in lowered:
            return "year"
        return ""

    @staticmethod
    def _infer_gap_type(text: str) -> str:
        lowered = str(text or "").lower()
        if any(t in lowered for t in RECENT_TERMS):
            return "missing_recent_update"
        if any(t in lowered for t in QUANT_TERMS):
            return "missing_quantitative_support"
        if any(t in lowered for t in COUNTER_EVIDENCE_TERMS):
            return "missing_counter_evidence"
        if any(t in lowered for t in DEFINITION_TERMS):
            return "missing_definition"
        if any(t in lowered for t in PRIMARY_DETAIL_TERMS):
            return "missing_primary_detail"
        if any(t in lowered for t in COMPARISON_TERMS):
            return "missing_counter_evidence"
        return "missing_primary_source"

    @staticmethod
    def _gap_type_to_search_goal(gap_type: str, text: str) -> str:
        lowered = str(text or "").lower()
        if any(t in lowered for t in COMPARISON_TERMS):
            return "comparison_fill"
        mapping = {
            "missing_definition": "definition_clarification",
            "missing_primary_source": "broad_scan",
            "missing_recent_update": "recent_update",
            "missing_quantitative_support": "quant_support",
            "missing_counter_evidence": "counter_evidence",
            "missing_entity_coverage": "broad_scan",
            "missing_primary_detail": "primary_detail",
        }
        return mapping.get(gap_type, "broad_scan")

    @staticmethod
    def _required_source_types_for_gap(gap_type: str) -> List[str]:
        mapping = {
            "missing_definition": ["primary_source", "documentation", "specification"],
            "missing_primary_source": ["primary_source", "documentation", "paper", "repository"],
            "missing_recent_update": ["primary_source", "announcement", "news"],
            "missing_quantitative_support": ["paper", "benchmark", "report", "primary_source"],
            "missing_counter_evidence": ["analysis", "community", "paper", "report"],
            "missing_entity_coverage": ["primary_source", "documentation", "report"],
            "missing_primary_detail": ["primary_source", "documentation", "repository", "specification"],
        }
        return list(mapping.get(gap_type, ["primary_source", "report"]))

    @staticmethod
    def _required_evidence_types_for_gap(gap_type: str) -> List[str]:
        mapping = {
            "missing_definition": ["definition", "mechanism"],
            "missing_primary_source": ["first_hand_statement", "implementation_detail"],
            "missing_recent_update": ["recent_change", "release_event"],
            "missing_quantitative_support": ["metric", "benchmark_result", "table"],
            "missing_counter_evidence": ["limitation", "tradeoff", "counter_example"],
            "missing_entity_coverage": ["entity_fact", "representative_example"],
            "missing_primary_detail": ["workflow", "implementation_detail", "architecture"],
        }
        return list(mapping.get(gap_type, ["fact"]))

    @staticmethod
    def _build_evidence_goal(gap_type: str, normalized_topic: str) -> str:
        mapping = {
            "missing_definition": f"Collect definitions, scope, and core mechanisms for {normalized_topic}.",
            "missing_primary_source": f"Collect primary-source evidence and original details for {normalized_topic}.",
            "missing_recent_update": f"Collect recent updates, releases, or changes for {normalized_topic}.",
            "missing_quantitative_support": f"Collect metrics, benchmarks, tables, or other quantitative evidence for {normalized_topic}.",
            "missing_counter_evidence": f"Collect limitations, tradeoffs, disagreements, or counter-evidence for {normalized_topic}.",
            "missing_entity_coverage": f"Collect representative entities, cases, and coverage for {normalized_topic}.",
            "missing_primary_detail": f"Collect implementation details, workflows, and engineering design evidence for {normalized_topic}.",
        }
        return mapping.get(gap_type, f"Collect the missing evidence needed for {normalized_topic}.")

    def _rule_query_plan(self, task: SubTask) -> Dict[str, Any]:
        raw_query = str(task.get("query") or "").strip()
        intent = str(task.get("intent") or "").strip()
        section_goal = str(task.get("section_goal") or "").strip()
        combined = " ".join(part for part in [raw_query, intent, section_goal] if part).strip()
        normalized_topic = self._strip_query_surface(raw_query or combined)
        if not normalized_topic:
            normalized_topic = self._strip_query_surface(intent or raw_query)

        task_gap_type = str(task.get("gap_type") or "").strip()
        gap_type = task_gap_type if task_gap_type in VALID_GAP_TYPES else self._infer_gap_type(combined or raw_query)
        search_goal = self._gap_type_to_search_goal(gap_type, combined)
        must_terms = self._dedupe_limited(self._extract_terms(normalized_topic), limit=6)
        version_terms = self._dedupe_limited(VERSION_PATTERN.findall(combined), limit=2)
        date_terms = self._dedupe_limited([m.group(0) for m in DATE_PATTERN.finditer(combined)], limit=2)
        time_scope = str(task.get("time_scope") or "").strip() or self._detect_time_scope(combined)
        required_source_types = self._dedupe_limited(
            ([str(i).strip() for i in task.get("required_source_types", [])] if isinstance(task.get("required_source_types"), list) else [])
            + self._required_source_types_for_gap(gap_type),
            limit=4,
        )
        required_evidence_types = self._dedupe_limited(
            ([str(i).strip() for i in task.get("required_evidence_types", [])] if isinstance(task.get("required_evidence_types"), list) else [])
            + self._required_evidence_types_for_gap(gap_type),
            limit=4,
        )
        evidence_goal = str(task.get("evidence_goal") or "").strip() or self._build_evidence_goal(gap_type, normalized_topic or raw_query)
        return {
            "gap_type": gap_type, "search_goal": search_goal, "evidence_goal": evidence_goal,
            "normalized_topic": normalized_topic, "must_terms": must_terms,
            "version_terms": version_terms, "date_terms": date_terms,
            "time_scope": time_scope, "required_source_types": required_source_types,
            "required_evidence_types": required_evidence_types,
        }

    async def _model_query_plans_batch(
        self,
        payload_items: List[Dict[str, Any]],
        task_id: str = "",
        topic: str = "",
        stage: str = "supervisor",
    ) -> Dict[int, Dict[str, Any]]:
        if not payload_items:
            return {}

        prompt = (
            "You are refining an internal search plan for a deep research engine.\n"
            "Return JSON only with this exact shape:\n"
            '{"items":[{"index":0,"gap_type":"missing_definition|missing_primary_source|missing_recent_update|missing_quantitative_support|missing_counter_evidence|missing_entity_coverage|missing_primary_detail","search_goal":"broad_scan|definition_clarification|primary_detail|recent_update|quant_support|counter_evidence|comparison_fill","normalized_topic":"","must_terms":[""],"required_source_types":[""],"required_evidence_types":[""],"time_scope":"day|week|month|year|"}]}\n'
            "Rules:\n"
            "- Do not generate full search queries.\n"
            "- Focus on what evidence is missing and what search action should be taken next.\n"
            "- Keep must_terms short and high-signal.\n"
            "- Use recent_update only when the query truly needs freshness.\n"
            "- Use quant_support when the user needs data, metrics, benchmarks, or evaluation.\n"
            "- Use counter_evidence when the section likely needs limits, tradeoffs, or disagreements.\n\n"
            f"Items: {payload_items}\n"
        )

        try:
            result = await call_llm_json(self.ctx, prompt, task_id=task_id, topic=topic or "query_plans", stage=stage, name=f"search_intent_batch_{len(payload_items)}")
            items = result.get("items") if isinstance(result, dict) else []
            if not isinstance(items, list):
                return {}
            plans: Dict[int, Dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index"))
                except Exception:
                    continue
                plans[index] = item
            return plans
        except Exception:
            return {}

    def _merge_query_plan(self, rule_plan: Dict[str, Any], model_plan: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(rule_plan)
        if not isinstance(model_plan, dict):
            return merged

        gap_type = str(model_plan.get("gap_type") or "").strip()
        if gap_type in VALID_GAP_TYPES:
            merged["gap_type"] = gap_type
        search_goal = str(model_plan.get("search_goal") or "").strip()
        if search_goal in VALID_SEARCH_GOALS:
            merged["search_goal"] = search_goal
        normalized_topic = str(model_plan.get("normalized_topic") or "").strip()
        if normalized_topic:
            merged["normalized_topic"] = normalized_topic
        time_scope = str(model_plan.get("time_scope") or "").strip()
        if time_scope in {"day", "week", "month", "year"}:
            merged["time_scope"] = time_scope
        must_terms = model_plan.get("must_terms")
        if isinstance(must_terms, list):
            merged["must_terms"] = self._dedupe_limited([str(i).strip() for i in must_terms if str(i).strip()], limit=6)
        source_types = model_plan.get("required_source_types")
        if isinstance(source_types, list):
            merged["required_source_types"] = self._dedupe_limited(
                list(merged.get("required_source_types", [])) + [str(i).strip() for i in source_types if str(i).strip()], limit=4,
            )
        evidence_types = model_plan.get("required_evidence_types")
        if isinstance(evidence_types, list):
            merged["required_evidence_types"] = self._dedupe_limited(
                list(merged.get("required_evidence_types", [])) + [str(i).strip() for i in evidence_types if str(i).strip()], limit=4,
            )
        return merged

    def _build_rewritten_queries(self, raw_query: str, plan: Dict[str, Any]) -> List[str]:
        search_goal = str(plan.get("search_goal") or "broad_scan")
        normalized_topic = str(plan.get("normalized_topic") or "").strip() or self._strip_query_surface(raw_query)
        must_terms = list(plan.get("must_terms", [])) if isinstance(plan.get("must_terms"), list) else []
        version_terms = list(plan.get("version_terms", [])) if isinstance(plan.get("version_terms"), list) else []
        date_terms = list(plan.get("date_terms", [])) if isinstance(plan.get("date_terms"), list) else []
        short_topic = " ".join((must_terms or self._extract_terms(normalized_topic))[:6]) or normalized_topic or raw_query
        date_suffix = " ".join(date_terms).strip()
        version_suffix = " ".join(version_terms).strip()
        gap_type = str(plan.get("gap_type") or "missing_primary_source")

        queries: List[str] = []
        goal_templates = {
            "definition_clarification": [f"{short_topic} definition", f"{short_topic} overview", f"{short_topic} core concepts"],
            "primary_detail": [f"{short_topic} mechanism design", f"{short_topic} implementation details", f"{short_topic} workflow architecture"],
            "recent_update": [f"{short_topic} latest update {date_suffix}", f"{short_topic} recent progress {date_suffix}", f"{short_topic} release announcement {date_suffix}"],
            "quant_support": [f"{short_topic} benchmark metrics", f"{short_topic} evaluation data", f"{short_topic} latency throughput accuracy"],
            "counter_evidence": [f"{short_topic} limitations challenges", f"{short_topic} risks tradeoffs", f"{short_topic} criticism issues"],
            "comparison_fill": [f"{short_topic} comparison", f"{short_topic} differences", f"{short_topic} tradeoffs"],
        }
        queries.extend(goal_templates.get(search_goal, [short_topic, f"{short_topic} {version_suffix}", f"{short_topic} {date_suffix}"]))
        queries.append(self._strip_query_surface(raw_query))

        gap_extras = {
            "missing_primary_source": f"{short_topic} official documentation source",
            "missing_recent_update": f"{short_topic} recent announcement",
            "missing_quantitative_support": f"{short_topic} benchmark report",
            "missing_counter_evidence": f"{short_topic} limitations comparison",
        }
        if gap_type in gap_extras:
            queries.append(gap_extras[gap_type])

        clean = self._dedupe_text_values([re.sub(r"\s+", " ", q).strip() for q in queries])
        return clean[:max(2, self.settings.search_rewrite_query_count)]

    def _build_search_profile(self, plan: Dict[str, Any], priority: str = "medium") -> Dict[str, Any]:
        search_goal = str(plan.get("search_goal") or "broad_scan")
        required_source_types = list(plan.get("required_source_types", [])) if isinstance(plan.get("required_source_types"), list) else []
        required_evidence_types = list(plan.get("required_evidence_types", [])) if isinstance(plan.get("required_evidence_types"), list) else []

        # Adaptive depth: scale max_results by section priority
        priority_profile = self.settings.get_search_profile(priority)

        profile: Dict[str, Any] = {
            "topic": "general",
            "search_depth": "advanced" if search_goal in {"primary_detail", "quant_support", "counter_evidence", "comparison_fill"} else "basic",
            "max_results": priority_profile.get("max_results", self.settings.search_max_results),
            "include_raw_content": True,
            "exclude_domains": DEFAULT_EXCLUDE_DOMAINS[:],
            "time_range": "",
            "required_source_types": required_source_types,
            "required_evidence_types": required_evidence_types,
            "cleaning_mode": "full",
        }
        if search_goal == "recent_update":
            profile["topic"] = "news"
            profile["search_depth"] = "advanced"
            profile["time_range"] = str(plan.get("time_scope") or "month")
        return profile

    async def prepare_subtasks_for_search(
        self,
        sub_tasks: List[SubTask],
        task_id: str = "",
        topic: str = "",
        stage: str = "supervisor",
    ) -> List[SubTask]:
        if not sub_tasks:
            return []

        rule_plans: List[Dict[str, Any]] = []
        batch_payload: List[Dict[str, Any]] = []
        for index, task in enumerate(sub_tasks):
            rule_plan = self._rule_query_plan(task)
            rule_plans.append(rule_plan)
            batch_payload.append({
                "index": index,
                "raw_query": str(task.get("query") or "").strip(),
                "intent": str(task.get("intent") or "").strip(),
                "section_goal": str(task.get("section_goal") or "").strip(),
                "rule_plan": rule_plan,
            })

        model_plans = await self._model_query_plans_batch(batch_payload, task_id=task_id, topic=topic, stage=stage)
        enriched: List[SubTask] = []
        for index, task in enumerate(sub_tasks):
            merged_plan = self._merge_query_plan(rule_plans[index], model_plans.get(index, {}))
            rewritten_queries = self._build_rewritten_queries(str(task.get("query") or ""), merged_plan)
            priority = str(task.get("priority") or "medium").strip() or "medium"
            search_profile = self._build_search_profile(merged_plan, priority=priority)
            enriched.append({
                **task,
                "gap_type": str(merged_plan.get("gap_type") or "missing_primary_source"),
                "search_goal": str(merged_plan.get("search_goal") or "broad_scan"),
                "evidence_goal": str(merged_plan.get("evidence_goal") or ""),
                "required_source_types": list(merged_plan.get("required_source_types", [])) if isinstance(merged_plan.get("required_source_types"), list) else [],
                "required_evidence_types": list(merged_plan.get("required_evidence_types", [])) if isinstance(merged_plan.get("required_evidence_types"), list) else [],
                "rewritten_queries": rewritten_queries or [str(task.get("query") or "").strip()],
                "time_scope": str(merged_plan.get("time_scope") or ""),
                "search_profile": search_profile,
            })
        return enriched

    # ── Follow-up Research Planning (Improvement 4) ──

    async def plan_follow_up(
        self,
        *,
        question: str,
        existing_plan: Dict[str, Any],
        existing_cards: list,
        task_id: str = "",
        topic: str = "",
        output_language: str = "en",
    ) -> Dict[str, Any]:
        """Generate incremental research plan for a follow-up question.

        Analyzes the follow-up question against existing research to determine:
        1. Which existing sections need expansion
        2. Whether new sections are needed
        3. What new sub-tasks to create

        Returns dict with 'new_sub_tasks' and 'updated_plan'.
        """
        existing_sections = existing_plan.get("sections", [])
        section_summaries = "\n".join(
            f"  {s.get('section_id', '')}: {s.get('title', '')} — {s.get('purpose', '')}"
            for s in existing_sections
        )
        card_summary = f"{len(existing_cards)} knowledge cards already collected"

        prompt = (
            "You are planning incremental follow-up research.\n\n"
            f"Original topic: {topic}\n"
            f"Follow-up question: {question}\n\n"
            f"Existing sections:\n{section_summaries}\n\n"
            f"Existing evidence: {card_summary}\n\n"
            "Decide how to handle this follow-up:\n"
            "- If the question deepens an existing section, add sub-tasks to that section.\n"
            "- If the question is a new area, create a new section.\n"
            "- Do NOT duplicate research already done.\n\n"
            f"All user-facing text must be in {'Chinese' if output_language == 'zh' else 'English'}.\n\n"
            "Return JSON:\n"
            '{"mode":"deepen|new_section|cross_validate",'
            '"target_sections":["S01"],'
            '"new_sections":[{"section_id":"S06","title":"","purpose":"","priority":"high","questions":[],"query_hints":[]}],'
            '"change_summary":"Brief description of what changed"}'
        )

        result = await call_llm_json(
            self.ctx, prompt, role="planner",
            task_id=task_id, topic=topic, stage="follow_up", name="plan_follow_up",
        )
        if not isinstance(result, dict):
            return {}

        mode = str(result.get("mode", "deepen"))
        change_summary = str(result.get("change_summary", f"Follow-up: {question}"))

        # Build updated plan
        updated_plan = dict(existing_plan)
        new_sections = result.get("new_sections", [])
        if isinstance(new_sections, list):
            for ns in new_sections:
                if isinstance(ns, dict) and ns.get("title"):
                    updated_plan.setdefault("sections", []).append(ns)

        # Build sub-tasks for the follow-up
        target_section_ids = result.get("target_sections", [])
        if not isinstance(target_section_ids, list):
            target_section_ids = []

        # Generate sub-tasks from target sections and new sections
        sub_tasks_source = []
        for section in updated_plan.get("sections", []):
            sid = section.get("section_id", "")
            if sid in target_section_ids or section in new_sections:
                sub_tasks_source.append(section)

        # If no sections matched, create a generic section for the follow-up
        if not sub_tasks_source:
            next_id = f"S{len(updated_plan.get('sections', [])) + 1:02d}"
            fallback = {
                "section_id": next_id,
                "title": question[:80],
                "purpose": f"Follow-up research: {question}",
                "priority": "high",
                "questions": [question],
                "query_hints": [],
                "depends_on": [],
            }
            updated_plan.setdefault("sections", []).append(fallback)
            sub_tasks_source = [fallback]

        new_sub_tasks = self.build_section_subtasks({"sections": sub_tasks_source})
        new_sub_tasks = await self.prepare_subtasks_for_search(
            new_sub_tasks, task_id=task_id, topic=topic, stage="follow_up",
        )

        return {
            "new_sub_tasks": new_sub_tasks,
            "updated_plan": updated_plan,
            "mode": mode,
            "change_summary": change_summary,
        }
