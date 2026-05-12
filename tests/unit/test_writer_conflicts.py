"""Tests for the conflict-aware prompt builder on WriterAgent.

The prompt injection is conditional: sections without conflicts must
produce NO header line at all (anchor effect: a "no conflicts" header
will prime the LLM to invent disagreement phrasing). Sections WITH
conflicts must produce a clearly numbered list plus an explicit
instruction to surface the disagreement.
"""

from __future__ import annotations

from deep_research_runtime.agents.writer import WriterAgent


def test_empty_records_produces_no_prompt_text():
    """Critical: empty list → empty string, not a 'Conflicts: none' header."""
    out = WriterAgent._format_conflicts_for_section_prompt([])
    assert out == "", "Empty conflict list MUST inject zero prompt text"


def test_single_record_produces_numbered_block():
    out = WriterAgent._format_conflicts_for_section_prompt([{
        "topic": "Model X accuracy",
        "severity": "strong",
        "disagreement_summary": "A reports 95%, B reports 80%",
        "claim_a": "Model X scores 95% on bench",
        "claim_b": "Model X scores 80% on bench",
        "source_a_title": "Paper A",
        "source_b_title": "Paper B",
    }])
    assert "Cross-source disagreements" in out
    assert "[strong]" in out
    assert "Model X accuracy" in out
    assert "Paper A" in out
    assert "Paper B" in out
    assert "Sources disagree on" in out, "Writer must be instructed to phrase as 'Sources disagree on'"


def test_multiple_records_are_numbered():
    out = WriterAgent._format_conflicts_for_section_prompt([
        {"topic": "T1", "severity": "moderate", "disagreement_summary": "s1",
         "claim_a": "a1", "claim_b": "b1",
         "source_a_title": "SA", "source_b_title": "SB"},
        {"topic": "T2", "severity": "weak", "disagreement_summary": "s2",
         "claim_a": "a2", "claim_b": "b2",
         "source_a_title": "SA", "source_b_title": "SB"},
    ])
    assert " 1. " in out and " 2. " in out


def test_fallback_to_url_when_source_title_missing():
    """A record without source_title falls back to the URL so the prompt still names a source."""
    out = WriterAgent._format_conflicts_for_section_prompt([{
        "topic": "Topic",
        "severity": "moderate",
        "disagreement_summary": "they disagree",
        "claim_a": "a",
        "claim_b": "b",
        "source_a_title": "",
        "source_a_url": "http://example.com/a",
        "source_b_title": "",
        "source_b_url": "http://example.com/b",
    }])
    assert "http://example.com/a" in out
    assert "http://example.com/b" in out


def test_no_phrase_in_empty_case():
    """Empty case must NOT mention 'Sources disagree' — otherwise the LLM will anchor on it."""
    out = WriterAgent._format_conflicts_for_section_prompt([])
    assert "Sources disagree" not in out
    assert "disagreement" not in out.lower()
