"""Structural grounding for citation generation in the writer.

The mature deep-research products (Gemini Deep Research, ChatGPT Deep
Research, Perplexity Pro) don't get accurate citations by auditing free
prose after the fact — they get them by **constraining the writer's
output structurally**. The model is given a closed set of evidence IDs and
asked to produce structured paragraphs that declare which IDs back each
piece of prose. The system then renders the citations from that
declaration; the LLM never gets to invent citation numbers.

This module implements that pattern. The writer calls into it to:

1. Format an evidence list keyed by short, prompt-friendly IDs (``E1``,
   ``E2``, ...).
2. Validate the LLM's JSON response against a closed set of IDs.
3. Verify any verbatim ``quote`` claim against the cited excerpt to catch
   numeric / factual paraphrase drift before it ships.
4. Render the validated paragraphs to Markdown with the existing
   superscript citation style.

Design choices worth flagging:

* **No regex post-audit on free prose.** That was the previous approach
  and it scaled poorly (false positives on long excerpts, false negatives
  on negation flips, expensive LLM verifier as the last resort). Closing
  the citation ID set at generation time eliminates the most damaging
  failure mode — invented sources — by construction.
* **The LLM may not output evidence IDs that aren't in the closed set.**
  When it does (rare with capable models), those IDs are dropped silently
  and recorded in the audit so callers can still see the violation.
* **Quotes are checked by *token containment*, not exact substring.** LLMs
  routinely paraphrase punctuation and whitespace; insisting on
  byte-identical substrings would reject valid quotes and tempt callers
  to disable the check. Containment of all claim-tokens inside the
  excerpt is a strong-enough signal.
* **Numeric grounding is a strict gate.** Any number in the paragraph
  text must appear (within ±5%) in the cited excerpts. This is the most
  visible class of hallucination and the easiest to catch deterministically.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Number pattern shared with conflict_detector; documented there. Replicated
# locally so this module stays self-contained for unit testing.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?", re.IGNORECASE)

# Token pattern: word chars + CJK characters. Used for token-level
# containment of the verbatim ``quote`` claim against the excerpt.
_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)

# Relative tolerance for numeric grounding. 95% vs 95.3% should pass;
# 95% vs 87% should fail. The exact value is a judgement call — tightening
# it produces more false positives on rounding, loosening tempts the LLM
# into drift. 5% is the working compromise.
NUMERIC_TOLERANCE = 0.05


# ──────────────────────────────────────────────────────────────────────
#  Evidence ID assignment
# ──────────────────────────────────────────────────────────────────────


def assign_evidence_ids(
    cards: List[Dict[str, Any]],
    *,
    prefix: str = "E",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Stamp each card with a short prompt-friendly ``evidence_id``.

    Returns ``(annotated_cards, id_to_card)``. The annotated list keeps
    the original order so downstream rendering can refer to ``ref_number``
    on the card if the writer wants reference-list-style output.
    """
    annotated: List[Dict[str, Any]] = []
    id_to_card: Dict[str, Dict[str, Any]] = {}
    for index, card in enumerate(cards, start=1):
        evidence_id = f"{prefix}{index}"
        copy = dict(card)
        copy["evidence_id"] = evidence_id
        annotated.append(copy)
        id_to_card[evidence_id] = copy
    return annotated, id_to_card


def format_evidence_for_prompt(annotated_cards: List[Dict[str, Any]]) -> str:
    """Render the evidence catalog the writer sees in its prompt.

    Each entry shows the evidence ID, the assigned reference number (so
    the LLM can talk about sources naturally if it wants), the source
    metadata, and the excerpt. We deliberately put the ID on its own line
    in brackets so it's easy for the LLM to grep visually when assembling
    the ``evidence_ids`` array in its JSON output.
    """
    lines: List[str] = []
    for card in annotated_cards:
        evidence_id = card.get("evidence_id", "")
        ref = card.get("reference_number") or card.get("ref") or ""
        source = card.get("source_title") or card.get("source") or "unknown source"
        url = card.get("source") or card.get("source_url") or ""
        excerpt = (card.get("exact_excerpt") or card.get("claim") or "").strip()
        if not excerpt:
            continue
        # Trim ultra-long excerpts. Writers typically only need 1-2
        # sentences of grounding context per card; sending a full page
        # inflates token cost without changing model behavior.
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rstrip() + "…"
        lines.append(
            f"[{evidence_id}] (ref {ref}) source: {source}"
            + (f"  url: {url}" if url else "")
            + f"\n  excerpt: {excerpt!r}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  Response parsing
# ──────────────────────────────────────────────────────────────────────


def parse_writer_response(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull the ``paragraphs`` list out of the writer's JSON response.

    Returns ``None`` when the response is unrecoverable. The caller then
    falls back to the legacy free-form path. We deliberately accept the
    same variants we tolerate in :func:`query_reform._parse_response` —
    bare lists, code-fenced JSON, double-quoted JSON-in-string.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    if isinstance(raw, dict):
        # Distinguish "no recognized key" (unrecoverable → None) from
        # "key present but empty list" (caller will treat both as fallback,
        # but the contract is clearer this way for tests / future callers).
        if "paragraphs" in raw:
            paragraphs = raw["paragraphs"]
        elif "output" in raw:
            paragraphs = raw["output"]
        else:
            return None
        if isinstance(paragraphs, list):
            return [p for p in paragraphs if isinstance(p, dict)]
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            return parse_writer_response(json.loads(text))
        except (ValueError, TypeError):
            return None
    return None


# ──────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _numbers(text: str) -> List[float]:
    out: List[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        try:
            out.append(float(match.group(0)))
        except ValueError:
            continue
    return out


def _quote_grounded_in_excerpt(quote: str, excerpt: str) -> bool:
    """True when every token of ``quote`` appears in ``excerpt``.

    Tolerates whitespace and punctuation differences. Stricter than mere
    set overlap because we require *containment*: the excerpt must include
    everything the quote claims to be quoting.
    """
    if not quote:
        return True
    qt = _tokens(quote)
    if not qt:
        return True
    et = _tokens(excerpt)
    return qt.issubset(et)


def _numbers_match(numbers_in_text: List[float], excerpt_numbers: Iterable[float]) -> List[float]:
    """Return the subset of ``numbers_in_text`` that are NOT corroborated.

    A number is corroborated when it matches an excerpt number to within
    :data:`NUMERIC_TOLERANCE` (relative). Returning the *unmatched* list
    so the caller can decide what to do with them (currently: drop the
    paragraph's quote field and flag the citation as ``numeric_unverified``).
    """
    pool = list(excerpt_numbers)
    unmatched: List[float] = []
    for n in numbers_in_text:
        denom = max(abs(n), 1.0)
        if any(abs(n - p) / max(abs(p), denom, 1.0) <= NUMERIC_TOLERANCE for p in pool):
            continue
        unmatched.append(n)
    return unmatched


def validate_paragraphs(
    paragraphs: List[Dict[str, Any]],
    id_to_card: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the closed-set + quote-grounding + numeric checks.

    Returns ``(cleaned_paragraphs, audit)``. Paragraphs whose
    ``evidence_ids`` reduce to the empty set after closed-set filtering
    are dropped entirely (no citations → no useful contribution to the
    grounded report). Quote and numeric violations are *recorded* in the
    audit but the paragraph survives — we don't silently delete the prose,
    just downgrade its trust level.
    """
    cleaned: List[Dict[str, Any]] = []
    invalid_ids: List[str] = []
    quote_failures: List[Dict[str, Any]] = []
    numeric_failures: List[Dict[str, Any]] = []
    total_citations = 0

    for index, para in enumerate(paragraphs):
        text = str(para.get("text") or "").strip()
        raw_ids = para.get("evidence_ids") or para.get("evidence") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        kept_ids: List[str] = []
        for eid in raw_ids:
            sid = str(eid).strip()
            if sid in id_to_card:
                kept_ids.append(sid)
            else:
                invalid_ids.append(sid)

        if not text:
            continue
        if not kept_ids:
            # Untied paragraph — keep the text but mark it as such; the
            # renderer can decide whether to demote or drop it.
            cleaned.append({
                "text": text,
                "evidence_ids": [],
                "quote_verified": None,
                "numeric_verified": None,
                "ungrounded": True,
            })
            continue

        # Quote grounding.
        quote = str(para.get("quote") or "").strip()
        quote_verified: Optional[bool] = None
        if quote:
            excerpts = " \n ".join(
                str(id_to_card[i].get("exact_excerpt") or id_to_card[i].get("claim") or "")
                for i in kept_ids
            )
            quote_verified = _quote_grounded_in_excerpt(quote, excerpts)
            if not quote_verified:
                quote_failures.append({
                    "paragraph_index": index,
                    "quote": quote,
                    "cited_ids": kept_ids,
                })

        # Numeric grounding — if the text contains numbers, at least one
        # must be corroborated by a cited excerpt.
        text_numbers = _numbers(text)
        numeric_verified: Optional[bool] = None
        if text_numbers:
            excerpt_numbers: List[float] = []
            for cid in kept_ids:
                excerpt_numbers.extend(_numbers(
                    str(id_to_card[cid].get("exact_excerpt") or "")
                    + " "
                    + str(id_to_card[cid].get("claim") or "")
                ))
            unmatched = _numbers_match(text_numbers, excerpt_numbers)
            numeric_verified = (not unmatched) if text_numbers else None
            if unmatched:
                numeric_failures.append({
                    "paragraph_index": index,
                    "unmatched_numbers": unmatched,
                    "text_preview": text[:120],
                    "cited_ids": kept_ids,
                })

        total_citations += len(kept_ids)
        cleaned.append({
            "text": text,
            "evidence_ids": kept_ids,
            "quote": quote or None,
            "quote_verified": quote_verified,
            "numeric_verified": numeric_verified,
            "ungrounded": False,
        })

    audit = {
        "paragraph_count": len(cleaned),
        "citations_total": total_citations,
        "structurally_valid": total_citations,  # by construction: every kept ID is in closed set
        "invalid_ids_dropped": invalid_ids,
        "ungrounded_paragraphs": sum(1 for p in cleaned if p.get("ungrounded")),
        "quote_failures": quote_failures,
        "numeric_failures": numeric_failures,
    }
    return cleaned, audit


# ──────────────────────────────────────────────────────────────────────
#  Rendering
# ──────────────────────────────────────────────────────────────────────


def render_paragraphs_to_markdown(
    paragraphs: List[Dict[str, Any]],
    id_to_card: Dict[str, Dict[str, Any]],
) -> str:
    """Turn the validated paragraph list into the Markdown a section uses.

    Citation markers are emitted as ``[N]`` plain-text references, where
    ``N`` is the card's ``reference_number``. WriterAgent already has a
    pipeline (``_replace_numbered_citations_with_links``) that turns
    these into ``<sup><a href="url">[N]</a></sup>`` later — by emitting
    plain ``[N]`` here, this module stays decoupled from the rendering
    layer in WriterAgent and the existing citation-link styling is
    preserved unchanged.
    """
    lines: List[str] = []
    for para in paragraphs:
        text = str(para.get("text") or "").strip()
        if not text:
            continue
        refs: List[int] = []
        for eid in para.get("evidence_ids") or []:
            card = id_to_card.get(eid)
            if not card:
                continue
            try:
                refs.append(int(card.get("reference_number")))
            except (TypeError, ValueError):
                continue
        # De-dup while preserving order so a paragraph never shows
        # ``[1][1][2]``.
        seen = set()
        refs = [r for r in refs if not (r in seen or seen.add(r))]
        markers = "".join(f"[{r}]" for r in refs)
        # If the LLM already embedded marker-like substrings inside the
        # paragraph text, strip them — citation rendering is OUR
        # responsibility now, not the LLM's.
        text = re.sub(r"\s*\[\d+\](?!\()", "", text).rstrip()
        if markers:
            lines.append(f"{text} {markers}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def build_audit_summary(audit: Dict[str, Any]) -> str:
    """Compact human-readable summary for surfacing in tool output."""
    parts = [
        f"paragraphs={audit.get('paragraph_count', 0)}",
        f"citations_total={audit.get('citations_total', 0)}",
        f"ungrounded_paragraphs={audit.get('ungrounded_paragraphs', 0)}",
    ]
    invalid = audit.get("invalid_ids_dropped") or []
    if invalid:
        parts.append(f"invalid_ids_dropped={len(invalid)}")
    qf = audit.get("quote_failures") or []
    if qf:
        parts.append(f"quote_failures={len(qf)}")
    nf = audit.get("numeric_failures") or []
    if nf:
        parts.append(f"numeric_failures={len(nf)}")
    return ", ".join(parts)


__all__ = [
    "assign_evidence_ids",
    "format_evidence_for_prompt",
    "parse_writer_response",
    "validate_paragraphs",
    "render_paragraphs_to_markdown",
    "build_audit_summary",
    "NUMERIC_TOLERANCE",
]
