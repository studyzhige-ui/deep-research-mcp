"""Cross-source conflict detection within a single research section.

When two knowledge cards in the same section make incompatible claims about
the same entity (e.g. "Model X scored 95% on benchmark Y" vs "Model X scored
87% on benchmark Y"), the writer should surface that disagreement explicitly
rather than silently pick one. This module produces the structured input
the writer needs.

Pipeline (per section, in order — each step trims the candidate pool):

1. **Gate** — sections with fewer than ``min_cards`` cards are skipped.
   Conflict surfacing requires multiple independent corroborations to be
   meaningful; one or two cards isn't enough signal.
2. **Coarse pair generation** — every unordered pair of cards within the
   section, then prune:

   * same source URL (a single article can't conflict with itself);
   * ``time_scope`` mismatch with no overlap (2020 data vs 2024 data on
     unemployment isn't a conflict, it's a time series);
   * stance combination — only ``supporting × counter`` pairs *or* any
     pair where both cards reference the same entity with different
     numbers. The other combinations (neutral, limitation) are typically
     complementary, not contradictory.
3. **Fine filter** — claim text token similarity above a small threshold
   so unrelated topics within the same section don't pair up.
4. **Rank + cap** — sort by confidence product (high × high first) and
   keep at most ``max_pairs_per_section``. This is what gets sent to the
   LLM.
5. **LLM judge** — a single batched call per section: "for each pair,
   classify ``COMPATIBLE / PARTIAL / CONTRADICTORY``". The judge only
   asserts *existence* of a conflict; severity comes from deterministic
   math (step 6).
6. **Severity scoring** — computed from card confidences and the
   magnitude of any numeric divergence. Bucketed into ``weak / moderate /
   strong`` so the writer can adjust phrasing.

The output is a ``{section_id: [ConflictRecord, ...]}`` dict that the
writer consumes; sections without conflicts are simply absent rather than
mapped to ``[]`` to keep state JSON-compact and to support a strict
"if section in conflicts" check in the writer prompt builder.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

# Card type is loose at the module boundary — TypedDicts don't validate at
# runtime and we want test fakes to pass plain dicts.
Card = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
#  Constants & helpers
# ──────────────────────────────────────────────────────────────────────

# Stance pairs we treat as candidate conflicts. Anything else is dropped
# during the coarse filter. ``neutral`` is intentionally NOT included
# because a neutral card next to a supporting/counter card usually adds
# context, not disagreement.
_CONFLICT_STANCE_PAIRS = frozenset({
    ("supporting", "counter"),
    ("counter", "supporting"),
})

# Confidence → numeric weight. We deliberately keep the spread modest
# (low ≠ 0) so a low-confidence card can still combine with a high-
# confidence one and produce a non-trivial severity.
_CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}

# Pattern for extracting numbers + optional units from claim text. Matches
# integers and decimals, with optional %, $, M, B, 万, 亿 suffixes. Tuned
# to favor recall over precision — we're using these matches to *detect*
# numeric divergence, not to format anything.
_NUMBER_RE = re.compile(
    r"-?\d+(?:\.\d+)?\s*"
    r"(?:%|[$￥€£]|million|billion|trillion|万|亿|千|百|k|m|b|tb|gb|mb)?",
    re.IGNORECASE,
)

# Token splitter for the claim-similarity fine filter. Whitespace + simple
# CJK char chunking; not perfect but good enough for "are these claims
# about the same thing" triage.
_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def _tokens(text: str) -> set:
    """Lower-case word + CJK character tokens. Empty input yields empty set."""
    if not text:
        return set()
    return set(_TOKEN_RE.findall(str(text).lower()))


def _claim_similarity(a: str, b: str) -> float:
    """Jaccard over claim tokens. Used as a topic-relatedness filter.

    Jaccard is appropriate *here* (unlike for citation fidelity, where we
    use containment) because we're asking "are these two claims about the
    same thing" — a symmetric question that wants both directions of
    overlap.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _entities_overlap(card_a: Card, card_b: Card) -> bool:
    """Cards share at least one named entity (case-insensitive)."""
    ea = {str(e).strip().lower() for e in (card_a.get("entities") or []) if str(e).strip()}
    eb = {str(e).strip().lower() for e in (card_b.get("entities") or []) if str(e).strip()}
    return bool(ea & eb)


def _time_scopes_compatible(card_a: Card, card_b: Card) -> bool:
    """True when two cards could be about the same time period.

    We're generous here — only reject when one card is clearly historical
    and the other clearly recent. Cards without explicit time_scope are
    assumed compatible with anything (otherwise we'd lose too many real
    conflicts to missing metadata).
    """
    sa = str(card_a.get("time_scope") or "").lower()
    sb = str(card_b.get("time_scope") or "").lower()
    if not sa or not sb:
        return True
    incompatible_pairs = {
        ("historical", "recent"), ("recent", "historical"),
        ("historical", "current"), ("current", "historical"),
        ("historical", "current_year"), ("current_year", "historical"),
        ("historical", "future"), ("future", "historical"),
    }
    return (sa, sb) not in incompatible_pairs


def _extract_numbers(text: str) -> List[float]:
    """Return numbers found in a string. Used for numeric divergence detection.

    We strip non-digit characters from each match before parsing, which
    means ``"95%"`` becomes ``95.0``. That's fine for our purposes —
    divergence is computed in relative terms and the unit just has to be
    consistent across the two compared claims.
    """
    out: List[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0)
        digits = re.sub(r"[^\d.\-]", "", raw)
        if not digits or digits in {".", "-", "-."}:
            continue
        try:
            out.append(float(digits))
        except ValueError:
            continue
    return out


def _numeric_divergence(a_text: str, b_text: str) -> Optional[float]:
    """Return relative numeric difference if both claims have at least one number.

    The metric is ``|a - b| / max(|a|, |b|, 1)`` of the first matched
    number in each claim. We use the first match because pairing all
    numbers cross-product blows up combinatorially and most claims have
    one dominant figure.

    Returns ``None`` when at least one claim has no numbers — the caller
    treats that as "non-numeric claim", which falls back to a default
    severity factor.
    """
    a_nums = _extract_numbers(a_text)
    b_nums = _extract_numbers(b_text)
    if not a_nums or not b_nums:
        return None
    a, b = a_nums[0], b_nums[0]
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _confidence_weight(value: Any) -> float:
    return _CONFIDENCE_WEIGHT.get(str(value or "").strip().lower(), 0.5)


# ──────────────────────────────────────────────────────────────────────
#  Candidate-pair generation
# ──────────────────────────────────────────────────────────────────────


def _generate_candidate_pairs(
    cards: List[Card],
    *,
    min_similarity: float = 0.3,
) -> List[Tuple[int, int]]:
    """Run every unordered pair through the coarse + fine filters.

    Returns indices into ``cards``, sorted by confidence product so the
    top-quality pairs survive the subsequent cap.
    """
    candidates: List[Tuple[int, int, float, float]] = []
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cards[i], cards[j]

            # Same source can't conflict with itself.
            if str(a.get("source") or "") == str(b.get("source") or "") and a.get("source"):
                continue

            # Time-scope sanity check — drop only the unambiguous
            # historical-vs-recent kind.
            if not _time_scopes_compatible(a, b):
                continue

            stance_pair = (str(a.get("stance") or "").lower(), str(b.get("stance") or "").lower())
            same_entity = _entities_overlap(a, b)
            numeric_div = _numeric_divergence(str(a.get("claim") or ""), str(b.get("claim") or ""))
            has_numeric_disagreement = same_entity and numeric_div is not None and numeric_div > 0.02

            # Keep pairs that either:
            #   (a) have opposing stances (supporting × counter), or
            #   (b) refer to the same entity with different numbers.
            if stance_pair not in _CONFLICT_STANCE_PAIRS and not has_numeric_disagreement:
                continue

            # Fine filter — claims should be talking about related things.
            sim = _claim_similarity(str(a.get("claim") or ""), str(b.get("claim") or ""))
            if sim < min_similarity and not has_numeric_disagreement:
                continue

            conf_product = _confidence_weight(a.get("confidence")) * _confidence_weight(b.get("confidence"))
            candidates.append((i, j, conf_product, sim))

    # Highest confidence × similarity first.
    candidates.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return [(i, j) for i, j, _, _ in candidates]


# ──────────────────────────────────────────────────────────────────────
#  Severity (deterministic — never asked from the LLM)
# ──────────────────────────────────────────────────────────────────────


def _severity(card_a: Card, card_b: Card) -> Tuple[str, Dict[str, Any]]:
    """Return ``(label, details)`` for a confirmed conflict.

    The label drives writer phrasing ("Sources strongly disagree" vs
    "Some sources report otherwise"). ``details`` carries the raw numeric
    bits in case downstream tools want to display them.
    """
    conf = _confidence_weight(card_a.get("confidence")) * _confidence_weight(card_b.get("confidence"))
    numeric_div = _numeric_divergence(
        str(card_a.get("claim") or ""), str(card_b.get("claim") or "")
    )
    if numeric_div is None:
        # Non-numeric claims: rely on confidence alone with a middle
        # divergence factor. Confidence-only severity rarely hits "strong"
        # without strong corroboration on both sides.
        divergence_factor = 0.5
        numeric_payload: Optional[Dict[str, Any]] = None
    elif numeric_div > 0.20:
        divergence_factor = 1.0
        numeric_payload = {"divergence_pct": round(numeric_div * 100, 2)}
    elif numeric_div > 0.05:
        divergence_factor = 0.6
        numeric_payload = {"divergence_pct": round(numeric_div * 100, 2)}
    else:
        divergence_factor = 0.3
        numeric_payload = {"divergence_pct": round(numeric_div * 100, 2)}

    score = conf * divergence_factor
    if score >= 0.7:
        label = "strong"
    elif score >= 0.4:
        label = "moderate"
    else:
        label = "weak"
    return label, {
        "confidence_product": round(conf, 3),
        "divergence_factor": divergence_factor,
        "score": round(score, 3),
        "numeric": numeric_payload,
    }


# ──────────────────────────────────────────────────────────────────────
#  LLM judge (single batched call per section)
# ──────────────────────────────────────────────────────────────────────


def _build_judge_prompt(section_title: str, pairs: List[Tuple[Card, Card]]) -> str:
    """Compose the single-call prompt asking the LLM to label each pair.

    The pairs are numbered so the LLM's JSON response can be mapped back
    even if it omits any. We ask only for COMPATIBLE / PARTIAL /
    CONTRADICTORY plus a one-line summary on contradictory verdicts —
    not severity (we compute that ourselves) and not "which one is right"
    (we don't want the LLM to silently take sides).
    """
    pair_blocks = []
    for idx, (a, b) in enumerate(pairs):
        pair_blocks.append(
            f"Pair {idx}:\n"
            f"  A) {a.get('claim', '')!r}  (source: {a.get('source_title') or a.get('source') or 'unknown'})\n"
            f"  B) {b.get('claim', '')!r}  (source: {b.get('source_title') or b.get('source') or 'unknown'})"
        )
    return (
        "You are auditing a research section for cross-source disagreement.\n"
        f"Section: {section_title!r}\n\n"
        "For each numbered pair of claims below, classify the relationship:\n"
        "- COMPATIBLE: the claims do not conflict, or describe complementary facets.\n"
        "- PARTIAL: the claims are partly compatible but disagree on some aspect or precondition.\n"
        "- CONTRADICTORY: the claims directly disagree about the same fact.\n\n"
        "Pairs:\n"
        + "\n".join(pair_blocks)
        + "\n\nReturn ONLY a JSON object:\n"
        '  {"pairs": [{"id": 0, "verdict": "COMPATIBLE|PARTIAL|CONTRADICTORY", "summary": "one-line description if not COMPATIBLE"}, ...]}\n'
        "If you cannot judge a pair, mark it COMPATIBLE — false positives are worse than false negatives here."
    )


def _parse_judge_response(raw: Any, n_pairs: int) -> Dict[int, Tuple[str, str]]:
    """Turn the LLM JSON into ``{pair_id: (verdict, summary)}``.

    Defensive: malformed input or missing entries default to COMPATIBLE so
    no fake conflict slips into the report. The contract documented in
    ``_build_judge_prompt`` makes this the safe default.
    """
    out: Dict[int, Tuple[str, str]] = {i: ("COMPATIBLE", "") for i in range(n_pairs)}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return out
    if not isinstance(raw, dict):
        return out
    pairs = raw.get("pairs", [])
    if not isinstance(pairs, list):
        return out
    for entry in pairs:
        if not isinstance(entry, dict):
            continue
        try:
            pid = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if pid not in out:
            continue
        verdict = str(entry.get("verdict") or "").upper().strip()
        if verdict not in {"COMPATIBLE", "PARTIAL", "CONTRADICTORY"}:
            continue
        summary = str(entry.get("summary") or "").strip()
        out[pid] = (verdict, summary)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────


async def detect_section_conflicts(
    section_id: str,
    section_title: str,
    cards: List[Card],
    *,
    call_llm_json: Callable[..., Awaitable[Any]],
    min_cards: int = 3,
    max_pairs: int = 6,
    min_similarity: float = 0.3,
    task_id: str = "",
    topic: str = "",
    stage: str = "conflict_detector",
) -> List[Dict[str, Any]]:
    """Return a list of confirmed conflict records for one section.

    Skips sections too small to bother (``len(cards) < min_cards``) or
    where the candidate-pair filter eliminates everything. The return
    shape is ready to JSON-serialize into ``ResearchState["section_conflicts"]``.
    """
    if len(cards) < min_cards:
        return []
    pair_indices = _generate_candidate_pairs(cards, min_similarity=min_similarity)
    if not pair_indices:
        return []
    pair_indices = pair_indices[:max_pairs]
    pairs = [(cards[i], cards[j]) for i, j in pair_indices]
    try:
        raw = await call_llm_json(
            _build_judge_prompt(section_title, pairs),
            task_id=task_id, topic=topic, stage=stage, name="conflict_judge",
        )
    except Exception:
        # Same philosophy as elsewhere — a flaky judge produces an empty
        # conflict list, not an exception. The downstream writer simply
        # skips the conflict-surfacing branch.
        return []
    verdicts = _parse_judge_response(raw, len(pairs))

    records: List[Dict[str, Any]] = []
    for pair_id, ((i, j), (card_a, card_b)) in enumerate(zip(pair_indices, pairs)):
        verdict, summary = verdicts.get(pair_id, ("COMPATIBLE", ""))
        if verdict not in {"PARTIAL", "CONTRADICTORY"}:
            continue
        severity_label, severity_details = _severity(card_a, card_b)
        records.append({
            "section_id": section_id,
            "topic": _infer_pair_topic(card_a, card_b),
            "verdict": verdict,
            "claim_a": str(card_a.get("claim") or ""),
            "claim_b": str(card_b.get("claim") or ""),
            "source_a_url": str(card_a.get("source") or ""),
            "source_b_url": str(card_b.get("source") or ""),
            "source_a_title": str(card_a.get("source_title") or ""),
            "source_b_title": str(card_b.get("source_title") or ""),
            "confidence_a": str(card_a.get("confidence") or "medium"),
            "confidence_b": str(card_b.get("confidence") or "medium"),
            "disagreement_summary": summary or _default_summary(card_a, card_b),
            "severity": severity_label,
            "severity_details": severity_details,
        })
    return records


def _infer_pair_topic(card_a: Card, card_b: Card) -> str:
    """Pick a short topic label from shared entities, falling back to a snippet."""
    ea = [str(e).strip() for e in (card_a.get("entities") or []) if str(e).strip()]
    eb = [str(e).strip() for e in (card_b.get("entities") or []) if str(e).strip()]
    common = [e for e in ea if e.lower() in {x.lower() for x in eb}]
    if common:
        return common[0]
    text = str(card_a.get("claim") or "").strip()
    return text[:60] + ("…" if len(text) > 60 else "")


def _default_summary(card_a: Card, card_b: Card) -> str:
    """Fallback when the LLM didn't provide a summary string."""
    return f"Sources disagree on this point."


async def detect_conflicts_for_state(
    knowledge_cards: Iterable[Card],
    section_digests: Iterable[Dict[str, Any]],
    *,
    call_llm_json: Callable[..., Awaitable[Any]],
    min_cards: int = 3,
    max_pairs: int = 6,
    task_id: str = "",
    topic: str = "",
) -> Dict[str, List[Dict[str, Any]]]:
    """Orchestrate section-by-section conflict detection across a whole task.

    Groups cards by ``section_id``, runs :func:`detect_section_conflicts`
    once per section, and returns a sparse mapping (sections with no
    conflicts are absent so the writer can use a clean ``in`` check).

    The function isn't tied to LangGraph — it accepts plain iterables so
    tests can drive it without spinning up a graph.
    """
    by_section: Dict[str, List[Card]] = {}
    for card in knowledge_cards:
        sid = str(card.get("section_id") or "")
        if not sid:
            continue
        by_section.setdefault(sid, []).append(card)

    titles = {str(d.get("section_id") or ""): str(d.get("title") or "") for d in section_digests}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sid, cards in by_section.items():
        records = await detect_section_conflicts(
            sid, titles.get(sid, sid), cards,
            call_llm_json=call_llm_json,
            min_cards=min_cards,
            max_pairs=max_pairs,
            task_id=task_id, topic=topic,
        )
        if records:
            out[sid] = records
    return out


__all__ = [
    "detect_section_conflicts",
    "detect_conflicts_for_state",
]
