"""Shared retry / backoff helpers.

The deep-research workflow has a few different sites that need to retry a
flaky operation (search calls, content fetches, LLM JSON parses). They all
end up wanting the same thing:

* exponential growth so we back off quickly on systemic outages,
* jitter so concurrent callers don't synchronize on the same retry instant
  (the "thundering herd" problem),
* a hard ceiling on the delay so a single failure doesn't stall the workflow
  for minutes.

Centralizing the formula avoids drift between call sites and makes the
behavior testable in isolation.
"""

from __future__ import annotations

import random
from typing import Optional


def compute_backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
    factor: float = 2.0,
    jitter: bool = True,
    rng: Optional[random.Random] = None,
) -> float:
    """Return the number of seconds to sleep before retry ``attempt``.

    Parameters
    ----------
    attempt:
        Zero-based attempt index. ``attempt=0`` is the first retry (i.e. the
        delay *after* the original call has already failed once).
    base:
        Multiplier for the exponential. With ``base=1.0`` the raw schedule
        is ``1, 2, 4, 8, ...``; with ``base=0.5`` it's ``0.5, 1, 2, ...``.
    cap:
        Hard upper bound. After the formula exceeds ``cap`` we clamp.
    factor:
        Growth factor between attempts. ``2.0`` is the textbook default.
    jitter:
        When ``True`` use "full jitter" (sleep is uniform in ``[0, raw)``).
        This is the AWS Architecture Blog recommendation — it spreads retries
        uniformly across the window and outperforms decorrelated jitter for
        small attempt counts. Set ``False`` for deterministic tests.
    rng:
        Optional :class:`random.Random` instance, used so tests can pin a
        seed without touching the global PRNG.

    Returns the delay in seconds, ``>= 0``.
    """
    if attempt < 0:
        return 0.0
    raw = min(cap, base * (factor ** attempt))
    if not jitter:
        return max(0.0, raw)
    rand = rng if rng is not None else random
    return max(0.0, rand.uniform(0.0, raw))
