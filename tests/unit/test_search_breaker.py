"""Tests for the per-engine circuit breaker in SearchService.

The breaker exists so a single flaky search engine (5xx-ing for a few
minutes) cannot drag every research task to a halt. These tests use a
fake retriever to assert the breaker opens after the configured threshold
and resets on success.
"""

from __future__ import annotations

import time

import pytest


class _FakeRetriever:
    """Test double: configurable to raise or return canned results."""

    source_layer = "general"

    def __init__(self, name: str, *, raise_with: Exception | None = None, payload=None):
        self.name = name
        self.raise_with = raise_with
        self.payload = payload or []
        self.call_count = 0

    async def search(self, session, query, profile):
        self.call_count += 1
        if self.raise_with is not None:
            raise self.raise_with
        return self.payload


def _make_service(settings, threshold: int, retrievers):
    """Helper: build a SearchService with the breaker threshold overridden.

    We override the attribute directly because Settings field defaults are
    evaluated at import time and `monkeypatch.setenv` after that point has
    no effect on already-imported defaults.
    """
    from deep_research_runtime.search_service import SearchService

    settings.search_circuit_breaker_threshold = threshold
    settings.search_circuit_breaker_cooldown_sec = 30
    settings.search_engine_rate_limit = 0
    return SearchService(settings, general_retrievers=retrievers, vertical_retrievers={})


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold(settings):
    retriever = _FakeRetriever("flaky", raise_with=RuntimeError("boom"))
    svc = _make_service(settings, threshold=2, retrievers=[retriever])

    # Two failures trip the breaker.
    await svc._run_retrievers(None, svc.general_retrievers, "q", {})
    await svc._run_retrievers(None, svc.general_retrievers, "q", {})
    assert retriever.call_count == 2

    # Third call is skipped — call_count must NOT increment.
    await svc._run_retrievers(None, svc.general_retrievers, "q", {})
    assert retriever.call_count == 2, "Breaker should skip the retriever once open"


@pytest.mark.asyncio
async def test_breaker_resets_on_success(settings):
    retriever = _FakeRetriever("ok", payload=[{"url": "http://x", "title": "ok"}])
    svc = _make_service(settings, threshold=3, retrievers=[retriever])

    # A single failure followed by a success should clear the counter.
    breaker = svc._breaker_for("ok")
    breaker.consecutive_failures = 2  # simulate prior failures

    docs = await svc._run_retrievers(None, svc.general_retrievers, "q", {})
    assert len(docs) == 1
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_breaker_disabled_when_threshold_zero(settings):
    """Operators can disable the breaker by setting threshold=0."""
    retriever = _FakeRetriever("flaky", raise_with=RuntimeError("boom"))
    svc = _make_service(settings, threshold=0, retrievers=[retriever])

    for _ in range(5):
        await svc._run_retrievers(None, svc.general_retrievers, "q", {})
    assert retriever.call_count == 5, "Breaker disabled → every call should reach retriever"


def test_circuit_breaker_state_transitions():
    """Direct unit test on the _CircuitBreaker class — no SearchService involved."""
    from deep_research_runtime.search_service import _CircuitBreaker

    b = _CircuitBreaker(threshold=2, cooldown=10)
    now = 1000.0
    assert not b.is_open(now)
    b.record_failure(now)
    assert not b.is_open(now), "1 failure < threshold(2)"
    b.record_failure(now)
    assert b.is_open(now), "2 failures hit threshold → open"
    # Cooldown not elapsed
    assert b.is_open(now + 5)
    # Cooldown elapsed → half-open / closed again (we just check is_open=False)
    assert not b.is_open(now + 11)
    # Success clears state entirely.
    b.record_success()
    assert b.consecutive_failures == 0
