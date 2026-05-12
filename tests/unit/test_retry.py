"""Tests for the shared retry/backoff helper."""

from __future__ import annotations

import random

from deep_research_runtime.retry import compute_backoff_delay


def test_deterministic_schedule_without_jitter():
    """Without jitter the formula is base * factor**attempt, capped."""
    schedule = [
        compute_backoff_delay(i, base=1.0, cap=30.0, factor=2.0, jitter=False)
        for i in range(6)
    ]
    assert schedule == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]


def test_cap_clamps_large_attempts():
    """A high attempt index must not produce an arbitrarily large delay."""
    delay = compute_backoff_delay(20, base=1.0, cap=10.0, factor=2.0, jitter=False)
    assert delay == 10.0


def test_jitter_stays_within_full_range():
    """Full jitter ⇒ delay is uniform in [0, raw)."""
    rng = random.Random(42)
    delays = [
        compute_backoff_delay(3, base=1.0, cap=30.0, factor=2.0, jitter=True, rng=rng)
        for _ in range(100)
    ]
    assert all(0.0 <= d <= 8.0 for d in delays)
    # Sanity: at least some variation across samples.
    assert max(delays) - min(delays) > 1.0


def test_negative_attempt_returns_zero():
    """Defensive: a negative index is treated as "don't sleep"."""
    assert compute_backoff_delay(-1) == 0.0


def test_jitter_is_seedable_for_reproducibility():
    """Two RNGs with the same seed must produce identical schedules."""
    a = random.Random(7)
    b = random.Random(7)
    schedule_a = [compute_backoff_delay(i, rng=a) for i in range(5)]
    schedule_b = [compute_backoff_delay(i, rng=b) for i in range(5)]
    assert schedule_a == schedule_b
