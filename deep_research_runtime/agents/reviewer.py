"""Reviewer agent: evaluates research quality and computes semantic saturation.

Wraps the existing QualityMixin for section review logic and adds:
- Semantic saturation scoring (Change 1)
- Degraded task handling (Change 7)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models import KnowledgeCard, SectionDigest
from ..quality import QualityMixin
from .base import AgentContext, call_llm_json


class ReviewerAgent:
    """Evaluates section quality, computes saturation, and generates follow-up tasks."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self._quality = QualityMixin()

    @property
    def settings(self):
        return self.ctx.settings

    # ── Section Review (delegates to QualityMixin) ──

    def rule_based_section_review(self, section_id, title, section, cards):
        return self._quality._rule_based_section_review(section_id, title, section, cards)

    async def llm_section_review(self, *, task_id, topic, section_id, title, section, section_cards, rule_review):
        # Inject call_llm_json into QualityMixin temporarily
        self._quality.call_llm_json = lambda prompt, **kw: call_llm_json(self.ctx, prompt, **kw)
        self._quality.settings = self.settings
        return await self._quality._llm_section_review(
            task_id=task_id, topic=topic, section_id=section_id,
            title=title, section=section, section_cards=section_cards,
            rule_review=rule_review,
        )

    def merge_section_review(self, section, rule_review, llm_review):
        return self._quality._merge_section_review(section, rule_review, llm_review)

    def build_section_digest(self, section, section_cards, section_review):
        return self._quality._build_section_digest(section, section_cards, section_review)

    # ── Semantic Saturation (Change 1) ──

    def compute_saturation_score(
        self,
        section_reviews: List[Dict[str, Any]],
        previous_coverage: float,
        cards_before: int,
        cards_after: int,
    ) -> Tuple[float, float]:
        """Compute how 'saturated' the research is.

        Returns (saturation_score, current_coverage) where:
        - saturation_score ∈ [0, 1]: 1.0 = fully saturated, no new info gained
        - current_coverage: aggregate coverage across all sections

        The system should stop researching when saturation_score >= threshold.
        """
        # Aggregate coverage from section reviews
        if not section_reviews:
            return 0.0, 0.0

        coverage_values = [float(r.get("coverage_score", 0.0) or 0.0) for r in section_reviews]
        current_coverage = sum(coverage_values) / max(1, len(coverage_values))

        # Coverage delta: how much did coverage improve this loop?
        coverage_delta = max(0.0, current_coverage - previous_coverage)

        # Marginal card gain: what fraction of cards are new?
        new_cards = max(0, cards_after - cards_before)
        marginal_gain = new_cards / max(cards_after, 1)

        # Saturation: high when both coverage_delta and marginal_gain are low
        # When coverage_delta = 0 and marginal_gain = 0 → saturation = 1.0
        saturation = 1.0 - (coverage_delta * 0.6 + marginal_gain * 0.4)
        saturation = max(0.0, min(1.0, saturation))

        return round(saturation, 3), round(current_coverage, 3)

    def should_stop_early(
        self,
        saturation_score: float,
        loop_count: int,
        has_follow_ups: bool,
    ) -> Tuple[bool, str]:
        """Decide whether to stop the research loop early.

        Returns (should_stop, reason).
        """
        if loop_count < self.settings.min_loops_before_early_stop:
            return False, ""

        if saturation_score >= self.settings.saturation_threshold:
            return True, f"Semantic saturation reached ({saturation_score:.2f} >= {self.settings.saturation_threshold})"

        if not has_follow_ups:
            return True, "No follow-up tasks available"

        if loop_count >= self.settings.max_reflection_loops:
            return True, f"Maximum reflection loops reached ({loop_count})"

        return False, ""

    # ── Degraded Task Assessment (Change 7) ──

    @staticmethod
    def assess_degraded_impact(
        sub_tasks: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess the impact of degraded sub-tasks on report quality.

        Returns a summary of which sections are affected and severity.
        """
        degraded = [t for t in sub_tasks if t.get("status") == "degraded"]
        if not degraded:
            return {"has_degraded": False, "critical_gaps": [], "supplementary_gaps": []}

        section_priority = {s.get("section_id", ""): s.get("priority", "medium") for s in sections}
        critical_gaps = []
        supplementary_gaps = []

        for task in degraded:
            sid = task.get("section_id", "")
            priority = section_priority.get(sid, "medium")
            gap_info = {
                "section_id": sid,
                "query": task.get("query", ""),
                "reason": task.get("degradation_reason", "Unknown"),
            }
            if priority == "high":
                critical_gaps.append(gap_info)
            else:
                supplementary_gaps.append(gap_info)

        return {
            "has_degraded": True,
            "critical_gaps": critical_gaps,
            "supplementary_gaps": supplementary_gaps,
        }
