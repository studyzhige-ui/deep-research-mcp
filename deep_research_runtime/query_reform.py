"""Multi-strategy query reformulation for zero-result searches.

The previous implementation issued one LLM call asking for a simpler
rewrite, tried it once, and degraded if that failed. That's a single-shot
strategy and misses recall in the common cases where the original query is
not just too specific but also uses the wrong vocabulary (CN ↔ EN), is
compound (should be decomposed), or is over-constrained by a time qualifier
that has no exact-match results.

This module asks for ``K`` reformulations *in a single LLM call*, each
explicitly using a different strategy, then lets the caller try them in
order until one returns results. That gets us the recall benefit of multiple
attempts at the LLM cost of a single one.

Design notes for callers:

* Strategies are listed in the prompt by ``StrategyName`` so a sufficiently
  capable LLM can tag each output. We don't *require* the LLM to label its
  outputs — we just want them diverse, and the strategy list is a forcing
  function for diversity.
* The function returns a list of up to ``max_attempts`` queries with
  duplicates removed and the original query filtered out. The caller treats
  an empty list as "give up, degrade the task."
* Failures (LLM error, malformed JSON) return an empty list rather than
  raising. The caller's degradation path is already designed for that case.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, List, Optional


# The strategy catalog is intentionally small and explicit. Each one
# encodes a distinct hypothesis about *why* the original query returned no
# hits, so the LLM is biased toward genuinely different rewrites instead of
# producing three near-paraphrases.
STRATEGIES = [
    ("SIMPLIFY", "Remove specific qualifiers and use more general wording. "
                 "Goal: broaden the search to anything related to the core topic."),
    ("SYNONYMS", "Swap key terminology for a synonym, technical-vs-colloquial form, "
                 "or the other language (EN ↔ CN). Goal: hit indexes that use different vocabulary."),
    ("DECOMPOSE", "Split the query into 2 narrower sub-queries (join with ' | '). "
                  "Goal: find sources that cover the parts even when no source covers the whole."),
    ("BROADEN_TIME", "Remove or relax the time/date qualifier. "
                     "Goal: find evergreen sources when fresh ones don't exist."),
    ("SPECIFY", "Add a more specific entity, product name, or technical term that the "
                "vague original might be hiding. Goal: convert a fuzzy query into an exact one."),
]


def _build_prompt(query: str, intent: str, max_attempts: int) -> str:
    """Compose the single prompt that asks the LLM for ``max_attempts`` rewrites.

    The prompt explicitly numbers the strategies and asks for exactly N
    outputs in order so the response is positionally interpretable even if
    the LLM omits strategy labels. ``intent`` is what the planner already
    annotates on the sub-task — passing it through gives the LLM context
    about *why* the user wanted this search in the first place.
    """
    strategies = STRATEGIES[:max_attempts]
    bullet_lines = "\n".join(
        f"{i + 1}. {name}: {desc}" for i, (name, desc) in enumerate(strategies)
    )
    return (
        "A web search query returned zero results. Generate alternative queries "
        "to retry. Each alternative must use a DIFFERENT strategy from the list "
        "below, in the listed order.\n\n"
        f"Original query: {query!r}\n"
        f"Research intent: {intent!r}\n\n"
        f"Strategies (use exactly these {len(strategies)} in order):\n"
        f"{bullet_lines}\n\n"
        "Return ONLY a JSON object with this shape:\n"
        '  {"queries": ["alt 1 using strategy 1", "alt 2 using strategy 2", ...]}\n'
        "Rules:\n"
        "- One alternative per strategy, in the listed order.\n"
        "- Do NOT repeat the original query.\n"
        "- Do NOT include explanations, strategy labels, or numbering inside the strings.\n"
        "- Each alternative is a complete standalone search query (3-12 words ideal).\n"
    )


def _parse_response(raw: Any) -> List[str]:
    """Extract ``queries`` from the LLM JSON response.

    Tolerates:

    * A dict already shaped like ``{"queries": [...]}``.
    * A bare list of strings (some LLMs ignore the wrapper).
    * A code-fenced or stringified JSON blob that needs another parse pass.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        queries = raw.get("queries", [])
        if isinstance(queries, list):
            return [str(item).strip() for item in queries if str(item).strip()]
    if isinstance(raw, str):
        # Strip common code-fence wrappers and try again.
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            return _parse_response(json.loads(text))
        except (ValueError, TypeError):
            return []
    return []


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    """Drop case-insensitive duplicates while keeping the first occurrence."""
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


async def reformulate_queries(
    query: str,
    intent: str,
    *,
    max_attempts: int,
    call_llm_json: Callable[..., Awaitable[Any]],
    task_id: str = "",
    topic: str = "",
    stage: str = "researcher",
) -> List[str]:
    """Generate up to ``max_attempts`` strategy-diverse alternatives for a failed query.

    Parameters
    ----------
    query, intent:
        The original search query and the sub-task's research intent. Both
        get inlined into the prompt so the LLM has context.
    max_attempts:
        How many alternatives to ask for. Capped at the length of
        :data:`STRATEGIES`. ``<= 0`` short-circuits to an empty list (the
        caller should treat that as "reformulation disabled").
    call_llm_json:
        Injected LLM callable. The tests pass a fake; production passes
        :func:`deep_research_runtime.agents.base.call_llm_json`.
    task_id, topic, stage:
        Tracing context propagated to ``call_llm_json``. Empty defaults
        keep the function unit-testable without a live agent context.
    """
    if max_attempts <= 0:
        return []
    capped = min(max_attempts, len(STRATEGIES))
    prompt = _build_prompt(query, intent, capped)
    try:
        raw = await call_llm_json(
            prompt,
            task_id=task_id,
            topic=topic,
            stage=stage,
            name="query_reformulation",
        )
    except Exception:
        # The caller treats an empty list as "give up" — that's already
        # the right behavior here, no need to surface the error to the
        # researcher's hot path.
        return []
    candidates = _parse_response(raw)
    # Filter out the original query (case-insensitive) and anything empty.
    original_lower = query.strip().lower()
    candidates = [c for c in candidates if c and c.lower() != original_lower]
    candidates = _dedupe_preserving_order(candidates)
    return candidates[:capped]


__all__ = ["reformulate_queries", "STRATEGIES"]
