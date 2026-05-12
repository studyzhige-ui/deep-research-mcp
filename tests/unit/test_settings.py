"""Tests for the Settings dataclass and its validation report."""

from __future__ import annotations

import pytest


def test_no_hardcoded_keys_in_source():
    """Regression guard: the previously-leaked API keys must never come back.

    If anyone re-adds a fallback like ``or "tvly-dev-..."`` for convenience,
    this test fails loudly. The actual strings are checked rather than
    individual settings values because the leak happened in fallback
    literals, not in defaults.
    """
    import pathlib

    leaked_fragments = [
        "tavily-leaked-test-fragment",
        "ad13fffd-d7b9-42df",
        "bcd75088a2d7413a",
        "975d9ebc825e41c9",
    ]
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for py_file in repo_root.rglob("*.py"):
        if "tests" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for frag in leaked_fragments:
            assert frag not in text, f"Leaked API key fragment {frag!r} found in {py_file}"


def _blank_keys(s):
    """Helper: zero out every key on a Settings instance for clean validation tests.

    Settings dataclass defaults are evaluated at class-definition time so any
    real keys present in the user's env at import-time would leak into tests.
    We override on the instance instead.
    """
    s.tavily_api_key = ""
    s.exa_api_key = ""
    s.serper_api_key = ""
    s.bocha_api_key = ""
    s.jina_api_key = ""
    s.serpapi_api_key = ""
    s.bing_api_key = ""
    s.google_api_key = ""
    s.llm_api_key = ""
    s.enable_duckduckgo_fallback = False


def test_validate_and_report_with_empty_env(settings):
    """With no keys configured, validate_and_report should flag a blocking error."""
    _blank_keys(settings)
    report = settings.validate_and_report(emit=False)

    assert report["search_engines"]["active"] == []
    assert any("No search engine" in err for err in report["blocking_errors"])
    assert any("LLM" in w or "API_KEY" in w for w in report["warnings"])


def test_validate_and_report_with_tavily_only(settings):
    """A single Tavily key should be enough for a healthy startup."""
    _blank_keys(settings)
    settings.tavily_api_key = "fake-tavily-key-for-tests"
    settings.llm_api_key = "fake-llm-key-for-tests"

    report = settings.validate_and_report(emit=False)
    assert "tavily" in report["search_engines"]["active"]
    assert report["blocking_errors"] == []
