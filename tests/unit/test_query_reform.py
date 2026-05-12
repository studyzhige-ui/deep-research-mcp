"""Tests for the multi-strategy query reformulation helper.

The function is the LLM-facing recall booster: when a search returns zero
results, we ask the model for N alternatives covering different strategies.
These tests fix down the contract the researcher relies on:

* The prompt mentions every strategy we want covered.
* The response is parsed robustly (dict, list, fenced JSON, garbage).
* Duplicates and the original query are filtered out.
* Errors propagate as an empty list, never as an exception, so the
  researcher's degradation path stays clean.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from deep_research_runtime.query_reform import (
    STRATEGIES,
    _build_prompt,
    _parse_response,
    reformulate_queries,
)


def test_build_prompt_lists_requested_strategies():
    """Strategies are explicitly numbered so the LLM has a forcing function for diversity."""
    prompt = _build_prompt("test query", "find details", max_attempts=3)
    # First three strategy names must appear in the prompt body.
    for name, _desc in STRATEGIES[:3]:
        assert name in prompt
    # Original query is quoted in the prompt for context.
    assert "test query" in prompt
    assert "find details" in prompt


def test_build_prompt_respects_max_attempts():
    """Asking for 2 attempts should not surface strategies 3+."""
    prompt = _build_prompt("q", "i", max_attempts=2)
    assert STRATEGIES[0][0] in prompt
    assert STRATEGIES[1][0] in prompt
    assert STRATEGIES[2][0] not in prompt


def test_parse_response_dict_shape():
    """Happy path: the model returned the requested shape."""
    raw = {"queries": ["alt one", "alt two", "alt three"]}
    assert _parse_response(raw) == ["alt one", "alt two", "alt three"]


def test_parse_response_bare_list():
    """Some models return just the list, ignoring the wrapper."""
    raw = ["alt one", "alt two"]
    assert _parse_response(raw) == ["alt one", "alt two"]


def test_parse_response_fenced_json():
    """Models often wrap JSON in ```json fences. Strip and re-parse."""
    raw = '```json\n{"queries": ["one", "two"]}\n```'
    assert _parse_response(raw) == ["one", "two"]


def test_parse_response_garbage_returns_empty():
    """Malformed input must never raise — the caller's degradation path handles []."""
    assert _parse_response("not json at all") == []
    assert _parse_response(None) == []
    assert _parse_response(42) == []
    assert _parse_response({"wrong_key": [1, 2]}) == []


def test_parse_response_strips_empty_strings():
    """Empty / whitespace-only strings get dropped before reaching the researcher."""
    raw = {"queries": ["good", "", "  ", "also good"]}
    assert _parse_response(raw) == ["good", "also good"]


# ── reformulate_queries (the public entry point) ────────────────────────


class _FakeLLM:
    """Test double that records calls and returns canned JSON responses."""

    def __init__(self, response: Any):
        self.response = response
        self.calls: list[Dict[str, Any]] = []

    async def __call__(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append({"prompt": prompt, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_reformulate_filters_original_query():
    """The model occasionally echoes the original — drop it case-insensitively."""
    llm = _FakeLLM({"queries": ["My Query", "alternative one", "alternative two"]})
    out = await reformulate_queries(
        "my query", "intent", max_attempts=3, call_llm_json=llm,
    )
    assert "my query" not in [s.lower() for s in out]
    assert out == ["alternative one", "alternative two"]


@pytest.mark.asyncio
async def test_reformulate_dedupes_case_insensitive():
    """If the model returns the same alternative twice, we keep one."""
    llm = _FakeLLM({"queries": ["Foo bar", "foo bar", "different one"]})
    out = await reformulate_queries(
        "original", "intent", max_attempts=5, call_llm_json=llm,
    )
    assert out == ["Foo bar", "different one"]


@pytest.mark.asyncio
async def test_reformulate_caps_at_max_attempts():
    """If the LLM returns 5 but the caller only wants 2, slice."""
    llm = _FakeLLM({"queries": ["a", "b", "c", "d", "e"]})
    out = await reformulate_queries(
        "q", "i", max_attempts=2, call_llm_json=llm,
    )
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_reformulate_caps_at_strategy_count():
    """max_attempts greater than the strategy catalog is clamped."""
    llm = _FakeLLM({"queries": ["a", "b", "c", "d", "e", "f", "g", "h"]})
    out = await reformulate_queries(
        "q", "i", max_attempts=100, call_llm_json=llm,
    )
    assert len(out) <= len(STRATEGIES)


@pytest.mark.asyncio
async def test_reformulate_llm_error_returns_empty():
    """The researcher relies on [] (not an exception) for degradation."""
    llm = _FakeLLM(RuntimeError("network died"))
    out = await reformulate_queries(
        "q", "i", max_attempts=3, call_llm_json=llm,
    )
    assert out == []


@pytest.mark.asyncio
async def test_reformulate_zero_attempts_short_circuits():
    """A non-positive cap means reformulation is disabled — don't even call the LLM."""
    llm = _FakeLLM({"queries": ["unused"]})
    out = await reformulate_queries(
        "q", "i", max_attempts=0, call_llm_json=llm,
    )
    assert out == []
    assert llm.calls == [], "LLM must not be invoked when max_attempts <= 0"


@pytest.mark.asyncio
async def test_reformulate_passes_tracing_context():
    """task_id / topic / stage propagate into the LLM call for tracing."""
    llm = _FakeLLM({"queries": ["one"]})
    await reformulate_queries(
        "q", "i", max_attempts=1, call_llm_json=llm,
        task_id="task-123", topic="my topic", stage="researcher",
    )
    assert llm.calls[0]["task_id"] == "task-123"
    assert llm.calls[0]["topic"] == "my topic"
    assert llm.calls[0]["stage"] == "researcher"
    assert llm.calls[0]["name"] == "query_reformulation"
