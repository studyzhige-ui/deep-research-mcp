"""Tests for the cross-source conflict detector.

These pin down the filtering pipeline so that:

* Small / homogeneous sections produce no work for the LLM.
* The various coarse filters (same source, stance pairs, time scopes)
  behave the way the writer's prompt expects.
* Severity is computed deterministically — never from the LLM.
* The LLM judge is parsed defensively (bad JSON / missing entries become
  COMPATIBLE rather than fake conflicts).
"""

from __future__ import annotations

from typing import Any, List

import pytest

from deep_research_runtime.conflict_detector import (
    _build_judge_prompt,
    _claim_similarity,
    _entities_overlap,
    _extract_numbers,
    _generate_candidate_pairs,
    _numeric_divergence,
    _parse_judge_response,
    _severity,
    _time_scopes_compatible,
    detect_conflicts_for_state,
    detect_section_conflicts,
)


def _card(
    unit_id: str,
    claim: str,
    *,
    source: str = "http://x",
    source_title: str = "X",
    stance: str = "neutral",
    confidence: str = "medium",
    entities: List[str] | None = None,
    section_id: str = "S1",
    time_scope: str = "",
) -> dict:
    return {
        "unit_id": unit_id,
        "claim": claim,
        "source": source,
        "source_title": source_title,
        "stance": stance,
        "confidence": confidence,
        "entities": entities or [],
        "section_id": section_id,
        "time_scope": time_scope,
    }


# ── Low-level helpers ───────────────────────────────────────────────


def test_entities_overlap_case_insensitive():
    a = _card("u1", "x", entities=["GPT-4"])
    b = _card("u2", "x", entities=["gpt-4", "Anthropic"])
    assert _entities_overlap(a, b)


def test_entities_overlap_no_shared_entity():
    a = _card("u1", "x", entities=["OpenAI"])
    b = _card("u2", "x", entities=["Anthropic"])
    assert not _entities_overlap(a, b)


def test_time_scopes_compatible_when_either_unset():
    """Missing metadata is generous — only reject obviously incompatible pairs."""
    a = _card("u1", "x")  # no time_scope
    b = _card("u2", "x", time_scope="recent")
    assert _time_scopes_compatible(a, b)


def test_time_scopes_reject_historical_vs_recent():
    a = _card("u1", "x", time_scope="historical")
    b = _card("u2", "x", time_scope="recent")
    assert not _time_scopes_compatible(a, b)


def test_extract_numbers_basic():
    assert _extract_numbers("accuracy of 95% on benchmark") == [95.0]
    assert _extract_numbers("revenue $2.5B in 2024") == [2.5, 2024.0]
    assert _extract_numbers("no numbers here") == []


def test_numeric_divergence_relative():
    """A 95 vs 87 disagreement is ~8 / 95 ≈ 0.084 relative."""
    div = _numeric_divergence("accuracy 95%", "accuracy 87%")
    assert div is not None
    assert 0.07 < div < 0.10


def test_numeric_divergence_none_when_no_numbers():
    assert _numeric_divergence("alpha", "beta") is None


def test_claim_similarity_partial_overlap():
    sim = _claim_similarity("model X is fast", "model X is slow")
    # "model" + "X" + "is" overlap; "fast"/"slow" differ → ~0.6
    assert 0.4 < sim < 0.8


def test_claim_similarity_unrelated():
    sim = _claim_similarity("apples are red", "code uses Python")
    assert sim == 0.0


# ── Candidate pair generation ───────────────────────────────────────


def test_no_pairs_when_same_source():
    """Two cards from the same URL can't be in conflict with each other."""
    cards = [
        _card("u1", "model X scores 95% on bench", source="http://a", stance="supporting",
              entities=["X"], confidence="high"),
        _card("u2", "model X scores 87% on bench", source="http://a", stance="counter",
              entities=["X"], confidence="high"),
    ]
    assert _generate_candidate_pairs(cards) == []


def test_no_pairs_when_only_neutral_stances():
    """neutral × neutral isn't a disagreement candidate."""
    cards = [
        _card("u1", "model X is good", stance="neutral", entities=["X"], confidence="high"),
        _card("u2", "model X is okay", stance="neutral", entities=["X"], confidence="high"),
    ]
    assert _generate_candidate_pairs(cards) == []


def test_pair_kept_for_supporting_vs_counter():
    cards = [
        _card("u1", "model X is fast", source="http://a", stance="supporting",
              entities=["X"], confidence="high"),
        _card("u2", "model X is slow", source="http://b", stance="counter",
              entities=["X"], confidence="high"),
    ]
    pairs = _generate_candidate_pairs(cards)
    assert pairs == [(0, 1)]


def test_pair_kept_for_numeric_disagreement_even_when_neutral():
    """Different numbers about same entity should be flagged regardless of stance."""
    cards = [
        _card("u1", "model X scores 95% on bench", source="http://a",
              stance="neutral", entities=["X"], confidence="high"),
        _card("u2", "model X scores 80% on bench", source="http://b",
              stance="neutral", entities=["X"], confidence="high"),
    ]
    pairs = _generate_candidate_pairs(cards)
    assert (0, 1) in pairs


def test_pair_rejected_when_time_scopes_incompatible():
    cards = [
        _card("u1", "rate is 4%", source="http://a", stance="supporting",
              entities=["unemployment"], time_scope="historical", confidence="high"),
        _card("u2", "rate is 6%", source="http://b", stance="counter",
              entities=["unemployment"], time_scope="recent", confidence="high"),
    ]
    assert _generate_candidate_pairs(cards) == []


def test_pairs_sorted_by_confidence_product():
    """The highest-confidence pair survives the cap when capacity is tight."""
    cards = [
        _card("hi_a", "X is fast", source="http://1", stance="supporting",
              entities=["X"], confidence="high"),
        _card("hi_b", "X is slow", source="http://2", stance="counter",
              entities=["X"], confidence="high"),
        _card("lo_a", "X is fast", source="http://3", stance="supporting",
              entities=["X"], confidence="low"),
        _card("lo_b", "X is slow", source="http://4", stance="counter",
              entities=["X"], confidence="low"),
    ]
    pairs = _generate_candidate_pairs(cards)
    # First pair should be the high×high one.
    assert pairs[0] == (0, 1)


# ── Severity ────────────────────────────────────────────────────────


def test_severity_strong_for_high_confidence_large_numeric_div():
    a = _card("a", "score is 95%", confidence="high")
    b = _card("b", "score is 50%", confidence="high")
    label, details = _severity(a, b)
    assert label == "strong"
    assert details["score"] >= 0.7
    assert details["numeric"] is not None


def test_severity_weak_for_low_confidence():
    a = _card("a", "score is 95%", confidence="low")
    b = _card("b", "score is 90%", confidence="low")
    label, _ = _severity(a, b)
    assert label == "weak"


def test_severity_non_numeric_falls_back_to_default_factor():
    """No numbers → severity comes from confidence alone with the 0.5 factor."""
    a = _card("a", "X is reliable", confidence="high")
    b = _card("b", "X is unreliable", confidence="high")
    label, details = _severity(a, b)
    # 1.0 × 1.0 × 0.5 = 0.5 → moderate
    assert label == "moderate"
    assert details["numeric"] is None


# ── LLM judge parsing ───────────────────────────────────────────────


def test_parse_judge_response_happy_path():
    raw = {"pairs": [
        {"id": 0, "verdict": "CONTRADICTORY", "summary": "scores disagree"},
        {"id": 1, "verdict": "COMPATIBLE", "summary": ""},
    ]}
    parsed = _parse_judge_response(raw, n_pairs=2)
    assert parsed[0] == ("CONTRADICTORY", "scores disagree")
    assert parsed[1] == ("COMPATIBLE", "")


def test_parse_judge_response_invalid_verdict_drops_to_compatible():
    raw = {"pairs": [{"id": 0, "verdict": "MAYBE", "summary": "x"}]}
    parsed = _parse_judge_response(raw, n_pairs=1)
    assert parsed[0] == ("COMPATIBLE", "")


def test_parse_judge_response_string_input_is_parsed_as_json():
    raw = '{"pairs":[{"id":0,"verdict":"PARTIAL","summary":"only sometimes"}]}'
    parsed = _parse_judge_response(raw, n_pairs=1)
    assert parsed[0] == ("PARTIAL", "only sometimes")


def test_parse_judge_response_garbage_yields_all_compatible():
    """Defensive: a broken LLM response must never invent conflicts."""
    assert _parse_judge_response("not json", n_pairs=2) == {
        0: ("COMPATIBLE", ""), 1: ("COMPATIBLE", ""),
    }


# ── Public entry: detect_section_conflicts ──────────────────────────


class _FakeLLM:
    def __init__(self, response: Any):
        self.response = response
        self.call_count = 0

    async def __call__(self, prompt: str, **kwargs: Any) -> Any:
        self.call_count += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_detect_section_conflicts_skips_small_sections():
    """min_cards gate avoids spending an LLM call on 1-2 card sections."""
    cards = [
        _card("u1", "X is fast", stance="supporting", entities=["X"]),
        _card("u2", "X is slow", stance="counter", entities=["X"]),
    ]
    llm = _FakeLLM({"pairs": []})
    out = await detect_section_conflicts(
        "S1", "Section", cards, call_llm_json=llm, min_cards=3,
    )
    assert out == []
    assert llm.call_count == 0, "LLM should not be invoked for small sections"


@pytest.mark.asyncio
async def test_detect_section_conflicts_returns_contradictory_only():
    """Only CONTRADICTORY/PARTIAL verdicts produce records; COMPATIBLE are dropped."""
    cards = [
        _card("u1", "model X scores 95% on bench", source="http://a",
              stance="supporting", entities=["X"], confidence="high"),
        _card("u2", "model X scores 80% on bench", source="http://b",
              stance="counter", entities=["X"], confidence="high"),
        _card("u3", "model X is widely used", source="http://c",
              stance="neutral", entities=["X"], confidence="medium"),
    ]
    llm = _FakeLLM({"pairs": [{"id": 0, "verdict": "CONTRADICTORY",
                                "summary": "95% vs 80%"}]})
    out = await detect_section_conflicts(
        "S1", "Performance", cards, call_llm_json=llm, min_cards=3,
    )
    assert len(out) == 1
    assert out[0]["verdict"] == "CONTRADICTORY"
    assert out[0]["disagreement_summary"] == "95% vs 80%"
    assert out[0]["severity"] in {"weak", "moderate", "strong"}


@pytest.mark.asyncio
async def test_detect_section_conflicts_llm_error_returns_empty():
    """LLM crash must not blow up the graph — degrade to no conflicts."""
    cards = [
        _card("u1", "X is fast", source="http://a", stance="supporting",
              entities=["X"], confidence="high"),
        _card("u2", "X is slow", source="http://b", stance="counter",
              entities=["X"], confidence="high"),
        _card("u3", "X exists", source="http://c", stance="neutral",
              entities=["X"], confidence="medium"),
    ]
    llm = _FakeLLM(RuntimeError("LLM died"))
    out = await detect_section_conflicts(
        "S1", "Section", cards, call_llm_json=llm, min_cards=3,
    )
    assert out == []


@pytest.mark.asyncio
async def test_detect_conflicts_for_state_groups_by_section():
    """Cards split across sections produce a sparse output map."""
    cards = [
        _card("a1", "X is fast", source="http://1", stance="supporting",
              entities=["X"], confidence="high", section_id="alpha"),
        _card("a2", "X is slow", source="http://2", stance="counter",
              entities=["X"], confidence="high", section_id="alpha"),
        _card("a3", "X is mainstream", source="http://3", stance="neutral",
              entities=["X"], confidence="high", section_id="alpha"),
        _card("b1", "Y costs $1B", source="http://4", stance="neutral",
              entities=["Y"], confidence="high", section_id="beta"),
        # beta has only 1 card → no conflicts produced
    ]
    digests = [
        {"section_id": "alpha", "title": "Alpha"},
        {"section_id": "beta", "title": "Beta"},
    ]
    llm = _FakeLLM({"pairs": [{"id": 0, "verdict": "CONTRADICTORY", "summary": "speed"}]})
    out = await detect_conflicts_for_state(
        cards, digests, call_llm_json=llm, min_cards=3,
    )
    assert "alpha" in out
    assert "beta" not in out, "beta has <min_cards, must be absent from sparse map"


def test_build_judge_prompt_contains_each_pair():
    """The LLM-facing prompt enumerates every pair so positional parsing works."""
    a = _card("u1", "A is good")
    b = _card("u2", "A is bad")
    c = _card("u3", "B is great")
    d = _card("u4", "B is awful")
    prompt = _build_judge_prompt("My Section", [(a, b), (c, d)])
    assert "Pair 0" in prompt
    assert "Pair 1" in prompt
    assert "A is good" in prompt
    assert "B is awful" in prompt
    assert "COMPATIBLE" in prompt and "CONTRADICTORY" in prompt
