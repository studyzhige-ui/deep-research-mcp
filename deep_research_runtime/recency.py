"""Recency weighting for source documents.

When a research task is time-sensitive (`time_scope` ∈ {"recent",
"current", "current_year"}), recent sources should outscore older ones —
otherwise a 2019 survey can edge out a 2025 preprint on the same topic
just because it has more body text. This module produces a multiplicative
weight ∈ [0.3, 1.0] that researcher.py applies on top of the existing
quality score.

Design choices the rest of the codebase relies on:

* **Half-life model**: ``weight = 0.5 ** (age_months / half_life_months)``.
  A document one half-life old gets half the weight of a brand-new one;
  two half-lives old gets a quarter; and so on.
* **Per-time-scope half-life**: the recency-sensitive scopes share a short
  half-life (months), neutral scopes use 3 years, and "historical"
  scopes opt out entirely (weight = 1.0).
* **Floor at 0.3**: an old but still relevant source should not be
  effectively excluded from ranking. A floor keeps it in the running.
* **Permissive date parsing**: search providers return dates in many
  formats (``"2024"``, ``"2024-03-15"``, ISO 8601, ``"Mar 15, 2024"``).
  We try ISO first, then a small regex set, then bail to weight 1.0.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional, Union

# Time scopes that explicitly opt out of recency decay. The researcher /
# planner produce these strings; keep the set narrow so a typo there doesn't
# accidentally disable decay for everything.
TIMELESS_SCOPES = frozenset({"timeless", "historical", "all_time", "any"})

# Half-life in months per scope. Anything not listed falls back to
# ``default_months``. The numeric choices are not magical — they encode:
#  * "recent" (~6 months) — current news / preprints territory
#  * "current" / "current_year" (~18 months) — last calendar cycle still wins
#  * "future" (~12 months) — forward-looking tasks weight fresh sources too
DEFAULT_HALF_LIFE_MONTHS = {
    "recent": 6,
    "current": 18,
    "current_year": 18,
    "future": 12,
}

# Lower bound on the returned weight. We never want a long-tail old source
# to score effectively 0 — it should still be a tiebreaker, just much
# weaker than a fresh source.
MIN_WEIGHT = 0.3


def _parse_published(value: Union[str, int, float, None]) -> Optional[datetime]:
    """Best-effort date parser tolerant of the formats search APIs emit.

    Returns ``None`` when nothing parseable is found — the caller treats
    that as "unknown date → no decay applied", which is the safe default.
    """
    if value is None or value == "":
        return None
    # Plain integer year (e.g. ``2024``) — very common from Semantic Scholar.
    if isinstance(value, (int, float)):
        year = int(value)
        if 1900 <= year <= 2200:
            return datetime(year, 6, 15, tzinfo=timezone.utc)
        return None
    text = str(value).strip()
    if not text:
        return None
    # ISO-8601 covers the majority of formats, including ``YYYY-MM-DD`` and
    # ``YYYY-MM-DDTHH:MM:SSZ``. ``fromisoformat`` accepts both since 3.11
    # but be defensive on the trailing Z. Date-only inputs produce a naive
    # datetime, so attach UTC if no tzinfo was parsed — every other code
    # path here is tz-aware, and naive vs aware can't be subtracted.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        pass
    # Year-only (very common in scholarly metadata).
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if 1900 <= year <= 2200:
            return datetime(year, 6, 15, tzinfo=timezone.utc)
    # Year-month (``2024-03``).
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1900 <= year <= 2200 and 1 <= month <= 12:
            return datetime(year, month, 15, tzinfo=timezone.utc)
    # ``Mar 15, 2024`` style English dates surfaced by some scrapers.
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


__all__ = ["recency_weight", "TIMELESS_SCOPES", "MIN_WEIGHT", "DEFAULT_HALF_LIFE_MONTHS"]


def _months_between(then: datetime, now: datetime) -> float:
    """Return age in months as a float (1 month ≈ 30.44 days)."""
    delta = now - then
    return delta.total_seconds() / (60 * 60 * 24 * 30.44)


def recency_weight(
    published: Union[str, int, float, None],
    time_scope: Optional[str],
    *,
    half_lives_override: Optional[dict] = None,
    default_half_life_months: float = 36.0,
    now: Optional[datetime] = None,
) -> float:
    """Return a multiplicative recency weight in ``[MIN_WEIGHT, 1.0]``.

    Parameters
    ----------
    published:
        Whatever the source metadata calls a publication date — a year
        integer, ISO date string, or plain ``None``. Unparseable inputs
        yield 1.0 (no decay) rather than 0 because we don't want absent
        metadata to penalize a source.
    time_scope:
        The task-level scope string from ``query_strategy`` (e.g.
        ``"recent"``, ``"current_year"``, ``"timeless"``). Anything in
        :data:`TIMELESS_SCOPES` opts out entirely.
    half_lives_override:
        Optional mapping that overrides :data:`DEFAULT_HALF_LIFE_MONTHS`
        for the call. Settings inject this so users can tune per-scope
        half-lives via env vars without re-importing.
    default_half_life_months:
        Used when ``time_scope`` is set but not in the half-life map. The
        36-month default means "no opinion on freshness" tasks see a very
        gentle decay rather than no decay at all.
    now:
        Test seam — fix the clock so date-based unit tests are
        deterministic. Production callers leave it as ``None``.
    """
    if not time_scope or time_scope in TIMELESS_SCOPES:
        return 1.0
    when = _parse_published(published)
    if when is None:
        # Missing date → treat as fresh. Penalizing unknown-date sources
        # would amount to fingerprinting on metadata quality, not on
        # publication recency, which is not what users mean by recency.
        return 1.0
    half_lives = dict(DEFAULT_HALF_LIFE_MONTHS)
    if half_lives_override:
        half_lives.update(half_lives_override)
    hl = float(half_lives.get(time_scope, default_half_life_months))
    if hl <= 0:
        return 1.0
    now_dt = now or datetime.now(timezone.utc)
    # Dirty data can produce future dates; clamp to 0 so a 2030-dated
    # source isn't accidentally boosted beyond 1.0 via a negative exponent.
    age_months = max(0.0, _months_between(when, now_dt))
    weight = 0.5 ** (age_months / hl)
    return max(MIN_WEIGHT, min(1.0, weight))
