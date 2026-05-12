"""Tests for the input validation helper used by tool_draft_plan / tool_follow_up_research."""

from __future__ import annotations

import pytest


def test_rejects_empty_string():
    from deep_research_runtime.tools import _validate_text_input

    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_text_input("", field="topic", max_length=100)
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_text_input("   \n\t ", field="topic", max_length=100)


def test_rejects_non_string():
    from deep_research_runtime.tools import _validate_text_input

    with pytest.raises(ValueError, match="must be a string"):
        _validate_text_input(123, field="topic", max_length=100)  # type: ignore[arg-type]


def test_rejects_overlong():
    from deep_research_runtime.tools import _validate_text_input

    with pytest.raises(ValueError, match="too long"):
        _validate_text_input("x" * 101, field="topic", max_length=100)


def test_trims_whitespace_and_returns_value():
    from deep_research_runtime.tools import _validate_text_input

    result = _validate_text_input("  hello world  ", field="topic", max_length=100)
    assert result == "hello world"


def test_max_length_constants_are_reasonable():
    """Cheap regression: someone might accidentally drop the cap to 0."""
    from deep_research_runtime.tools import (
        MAX_BACKGROUND_LENGTH,
        MAX_FOLLOW_UP_LENGTH,
        MAX_TOPIC_LENGTH,
    )

    assert 100 <= MAX_TOPIC_LENGTH <= 10000
    assert 100 <= MAX_FOLLOW_UP_LENGTH <= 10000
    assert 100 <= MAX_BACKGROUND_LENGTH <= 20000
