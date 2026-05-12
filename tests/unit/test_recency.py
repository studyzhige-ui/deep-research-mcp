"""Tests for the recency weighting helper.

These pin down the public contract that researcher.py relies on: weight is
deterministic given (published, time_scope, now), monotonic in age, opt-out
on timeless scopes, and robust against the messy date formats real search
APIs emit.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from deep_research_runtime.recency import (
    MIN_WEIGHT,
    TIMELESS_SCOPES,
    recency_weight,
)


NOW = datetime(2026, 5, 12, tzinfo=timezone.utc)


def test_timeless_scope_returns_one():
    """Tasks marked timeless/historical opt out of decay entirely."""
    for scope in TIMELESS_SCOPES:
        assert recency_weight("2010", scope, now=NOW) == 1.0


def test_missing_date_returns_one():
    """Unknown publish date must not penalize — recency is about age, not metadata quality."""
    assert recency_weight(None, "recent", now=NOW) == 1.0
    assert recency_weight("", "recent", now=NOW) == 1.0
    assert recency_weight("not a date", "recent", now=NOW) == 1.0


def test_missing_time_scope_returns_one():
    """No time_scope means the caller doesn't care about freshness."""
    assert recency_weight("2010", "", now=NOW) == 1.0
    assert recency_weight("2010", None, now=NOW) == 1.0


def test_fresh_document_close_to_one():
    """A document published last month under the 6-month half-life should be ~0.9."""
    fresh = datetime(2026, 4, 12, tzinfo=timezone.utc).isoformat()
    w = recency_weight(fresh, "recent", now=NOW)
    assert 0.85 < w <= 1.0


def test_one_half_life_old_is_half():
    """At exactly one half-life, weight = 0.5 (formula is monotone with age)."""
    # recent → half_life = 6 months; published 6 months before NOW
    pub = datetime(2025, 11, 12, tzinfo=timezone.utc).isoformat()
    w = recency_weight(pub, "recent", now=NOW)
    assert 0.45 < w < 0.55


def test_floor_applies_for_very_old():
    """Long-tail old sources don't go to zero; they floor at MIN_WEIGHT."""
    w = recency_weight("2000-01-01", "recent", now=NOW)
    assert w == MIN_WEIGHT


def test_future_dates_clamped_to_now():
    """A document dated 2030 should NOT boost above 1.0 via negative age."""
    w = recency_weight("2030-01-01", "recent", now=NOW)
    assert w == 1.0


def test_monotonicity_in_age():
    """Older documents must weigh less than newer ones, all else equal."""
    a = recency_weight("2024-01-01", "current", now=NOW)
    b = recency_weight("2025-01-01", "current", now=NOW)
    c = recency_weight("2026-01-01", "current", now=NOW)
    assert a < b < c


def test_year_only_parses():
    """Semantic Scholar and arXiv often emit just the year as an int."""
    w_int = recency_weight(2024, "current", now=NOW)
    w_str = recency_weight("2024", "current", now=NOW)
    # Both should resolve to roughly the same value (mid-year approximation).
    assert abs(w_int - w_str) < 1e-9


def test_year_month_parses():
    """`2024-03` is a common partial-date format from PubMed."""
    w = recency_weight("2024-03", "current", now=NOW)
    assert MIN_WEIGHT < w < 1.0


def test_iso_datetime_with_z():
    """`2024-03-15T12:00:00Z` from Tavily — fromisoformat needs the Z swap."""
    w = recency_weight("2024-03-15T12:00:00Z", "current", now=NOW)
    assert MIN_WEIGHT < w < 1.0


def test_english_date_string_parses():
    """`Mar 15, 2024` style strings from page scrapers."""
    w = recency_weight("Mar 15, 2024", "current", now=NOW)
    assert MIN_WEIGHT < w < 1.0


def test_half_life_override_changes_curve():
    """Caller-supplied half-life override actually takes effect."""
    pub = "2024-01-01"
    short = recency_weight(pub, "recent", half_lives_override={"recent": 3}, now=NOW)
    long_ = recency_weight(pub, "recent", half_lives_override={"recent": 60}, now=NOW)
    # Same age, shorter half-life → smaller weight
    assert short < long_


def test_default_half_life_for_unknown_scope():
    """Unknown scopes get the default half-life rather than 1.0."""
    pub = "2023-05-12"  # 3 years before NOW
    w = recency_weight(pub, "weird_custom_scope", default_half_life_months=36, now=NOW)
    # 3 years at 36-month half-life ≈ 0.5
    assert 0.45 < w < 0.55


def test_zero_or_negative_half_life_disables_decay():
    """Defensive: hl ≤ 0 means caller wants no decay regardless of scope."""
    assert recency_weight("2010", "recent", half_lives_override={"recent": 0}, now=NOW) == 1.0
    assert recency_weight("2010", "recent", half_lives_override={"recent": -5}, now=NOW) == 1.0
