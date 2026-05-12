"""Tests for the structural-grounding pipeline used by the writer.

Two things matter here that the legacy "audit prose after the fact" path
did not provide:

* **Closed-set citation IDs.** The LLM's JSON output is validated against
  the evidence catalog; invented IDs are dropped silently and recorded.
* **Numeric and quote grounding.** Any number in a paragraph must appear
  (within ±5%) in at least one cited excerpt, and a ``quote`` claim must
  be a token-subset of the cited excerpts.

These tests pin the contract so future changes to the writer can't
silently weaken either guarantee.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from deep_research_runtime.citation_grounding import (
    NUMERIC_TOLERANCE,
    assign_evidence_ids,
    build_audit_summary,
    format_evidence_for_prompt,
    parse_writer_response,
    render_paragraphs_to_markdown,
    validate_paragraphs,
)


def _card(claim: str, excerpt: str, ref: int, source: str = "http://x") -> Dict[str, Any]:
    """Minimal card shape — just the fields the grounding code touches."""
    return {
        "claim": claim,
        "exact_excerpt": excerpt,
        "reference_number": ref,
        "source": source,
        "source_title": f"Source {ref}",
    }


# ── ID assignment / prompt formatting ──────────────────────────────


def test_assign_evidence_ids_is_one_indexed_and_stable():
    cards = [_card("a", "ex a", 1), _card("b", "ex b", 2), _card("c", "ex c", 3)]
    annotated, id_to_card = assign_evidence_ids(cards)
    assert [c["evidence_id"] for c in annotated] == ["E1", "E2", "E3"]
    assert id_to_card["E2"]["claim"] == "b"


def test_format_evidence_for_prompt_includes_ids_and_excerpts():
    annotated, _ = assign_evidence_ids([_card("X is fast", "X scored 95% on benchmark Y", 1)])
    block = format_evidence_for_prompt(annotated)
    assert "[E1]" in block
    assert "95%" in block
    assert "ref 1" in block


def test_format_evidence_trims_long_excerpts():
    long_excerpt = "word " * 500  # ~2500 chars
    annotated, _ = assign_evidence_ids([_card("c", long_excerpt, 1)])
    block = format_evidence_for_prompt(annotated)
    assert len(block) < 1500  # trimmed to ~800 chars + framing


# ── Response parsing ───────────────────────────────────────────────


def test_parse_writer_response_dict_shape():
    raw = {"paragraphs": [{"text": "p1", "evidence_ids": ["E1"]}]}
    out = parse_writer_response(raw)
    assert out == [{"text": "p1", "evidence_ids": ["E1"]}]


def test_parse_writer_response_handles_fenced_json():
    raw = '```json\n{"paragraphs": [{"text": "x", "evidence_ids": ["E2"]}]}\n```'
    out = parse_writer_response(raw)
    assert out == [{"text": "x", "evidence_ids": ["E2"]}]


def test_parse_writer_response_bad_input_returns_none():
    """The caller relies on None to trigger the legacy fallback path."""
    assert parse_writer_response(None) is None
    assert parse_writer_response("not json at all") is None
    assert parse_writer_response(42) is None
    assert parse_writer_response({"wrong_key": []}) is None


def test_parse_writer_response_filters_non_dicts():
    raw = {"paragraphs": [{"text": "ok"}, "garbage", 123]}
    out = parse_writer_response(raw)
    assert out == [{"text": "ok"}]


# ── Validation: closed-set filter ──────────────────────────────────


def test_validation_drops_invented_evidence_ids():
    """The LLM tried to cite E99 (not in our set) — must be filtered out."""
    cards = [_card("a", "ex a", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "hello", "evidence_ids": ["E1", "E99"]}]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["evidence_ids"] == ["E1"]
    assert "E99" in audit["invalid_ids_dropped"]


def test_validation_ungrounded_when_all_ids_invalid():
    cards = [_card("a", "ex a", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "hello", "evidence_ids": ["E99", "E42"]}]
    cleaned, _audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["ungrounded"] is True
    assert cleaned[0]["evidence_ids"] == []


def test_validation_drops_empty_text_paragraphs():
    cards = [_card("a", "ex a", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "", "evidence_ids": ["E1"]}]
    cleaned, _ = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned == []


# ── Validation: quote grounding ────────────────────────────────────


def test_quote_grounded_passes_when_tokens_subset():
    cards = [_card("X is fast", "Model X scored 95% on benchmark Y", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{
        "text": "Model X reaches 95% accuracy.",
        "evidence_ids": ["E1"],
        "quote": "Model X scored 95%",
    }]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["quote_verified"] is True
    assert audit["quote_failures"] == []


def test_quote_grounded_fails_when_tokens_not_in_excerpt():
    cards = [_card("X is fast", "Model X is fast", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{
        "text": "Model X reaches 95% accuracy.",
        "evidence_ids": ["E1"],
        "quote": "Model X scored 95%",  # 95% / scored NOT in excerpt
    }]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["quote_verified"] is False
    assert len(audit["quote_failures"]) == 1
    # Paragraph survives — we record the failure, don't delete the prose
    assert cleaned[0]["text"]


def test_no_quote_means_no_quote_check():
    """Absent quote field is not a failure — many sentences don't need verbatim grounding."""
    cards = [_card("a", "ex a", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "narrative claim", "evidence_ids": ["E1"]}]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["quote_verified"] is None
    assert audit["quote_failures"] == []


# ── Validation: numeric grounding ──────────────────────────────────


def test_numeric_match_within_tolerance_passes():
    """95% in text vs 95.3% in excerpt → within 5% relative tolerance."""
    cards = [_card("X", "Model X scored 95.3% on bench", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Model X scored 95%.", "evidence_ids": ["E1"]}]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["numeric_verified"] is True
    assert audit["numeric_failures"] == []


def test_numeric_mismatch_beyond_tolerance_fails():
    """95% in text vs 80% in excerpt → numeric_failure."""
    cards = [_card("X", "Model X scored 80% on bench", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Model X scored 95%.", "evidence_ids": ["E1"]}]
    cleaned, audit = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["numeric_verified"] is False
    assert len(audit["numeric_failures"]) == 1
    assert 95.0 in audit["numeric_failures"][0]["unmatched_numbers"]


def test_no_numbers_means_no_numeric_check():
    cards = [_card("X is good", "Model X is widely used", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Model X performs well.", "evidence_ids": ["E1"]}]
    cleaned, _ = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["numeric_verified"] is None


def test_numeric_check_pools_across_cited_excerpts():
    """A number matching ANY cited excerpt counts as verified."""
    cards = [
        _card("a", "first source mentions 80%", 1),
        _card("b", "second source confirms 95%", 2),
    ]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Model X reaches 95%.", "evidence_ids": ["E1", "E2"]}]
    cleaned, _ = validate_paragraphs(paragraphs, id_to_card)
    assert cleaned[0]["numeric_verified"] is True


# ── Rendering ──────────────────────────────────────────────────────


def test_render_emits_reference_numbers_not_evidence_ids():
    """[E1] is the prompt-side identifier; rendered output uses ref numbers."""
    cards = [_card("a", "ex a", 7)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Hello world", "evidence_ids": ["E1"]}]
    out = render_paragraphs_to_markdown(paragraphs, id_to_card)
    assert "[7]" in out
    assert "E1" not in out


def test_render_dedupes_repeated_references():
    """A paragraph that cites E1 twice should produce a single [N] marker."""
    cards = [_card("a", "ex a", 3)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "p", "evidence_ids": ["E1", "E1"]}]
    out = render_paragraphs_to_markdown(paragraphs, id_to_card)
    assert out.count("[3]") == 1


def test_render_strips_llm_emitted_citation_markers_in_text():
    """LLM may sneak [3] into text — system, not LLM, owns citation rendering."""
    cards = [_card("a", "ex a", 4)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "Hello [99] world.", "evidence_ids": ["E1"]}]
    out = render_paragraphs_to_markdown(paragraphs, id_to_card)
    assert "[99]" not in out
    assert "[4]" in out


def test_render_joins_paragraphs_with_blank_lines():
    cards = [_card("a", "ex a", 1), _card("b", "ex b", 2)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [
        {"text": "p1", "evidence_ids": ["E1"]},
        {"text": "p2", "evidence_ids": ["E2"]},
    ]
    out = render_paragraphs_to_markdown(paragraphs, id_to_card)
    assert "p1 [1]\n\np2 [2]" in out


def test_render_handles_paragraph_without_evidence_gracefully():
    """Ungrounded paragraphs survive in the output with no citation markers."""
    cards = [_card("a", "ex a", 1)]
    _, id_to_card = assign_evidence_ids(cards)
    paragraphs = [{"text": "intro line", "evidence_ids": []}]
    out = render_paragraphs_to_markdown(paragraphs, id_to_card)
    assert out.strip() == "intro line"


# ── Audit summary ──────────────────────────────────────────────────


def test_build_audit_summary_compact_and_includes_only_real_failures():
    audit = {
        "paragraph_count": 5,
        "citations_total": 12,
        "ungrounded_paragraphs": 0,
        "invalid_ids_dropped": [],
        "quote_failures": [],
        "numeric_failures": [],
    }
    summary = build_audit_summary(audit)
    assert "paragraphs=5" in summary
    assert "citations_total=12" in summary
    # Zero-failure categories should NOT clutter the summary
    assert "quote_failures" not in summary
    assert "numeric_failures" not in summary


def test_build_audit_summary_reports_failures_when_present():
    audit = {
        "paragraph_count": 3,
        "citations_total": 5,
        "ungrounded_paragraphs": 1,
        "invalid_ids_dropped": ["E99"],
        "quote_failures": [{"paragraph_index": 0}],
        "numeric_failures": [{"paragraph_index": 1}, {"paragraph_index": 2}],
    }
    summary = build_audit_summary(audit)
    assert "invalid_ids_dropped=1" in summary
    assert "quote_failures=1" in summary
    assert "numeric_failures=2" in summary


# ── Sanity: tolerance constant matches documentation ───────────────


def test_numeric_tolerance_is_documented_value():
    """If anyone changes the tolerance, this test fails so docs get updated too."""
    assert NUMERIC_TOLERANCE == 0.05
