"""Stay-length estimates that account for cars still plugged in.

At any decision time, recent cars are often still plugged in: their stay is only known to
be at least as long as it has lasted so far. Dropping them would bias estimates towards
short stays (only the cars that left early would count). The Kaplan-Meier estimator uses
them correctly: a car still plugged in counts as "at risk of leaving" up to its time so far.
"""

from __future__ import annotations

import numpy as np

_REL_TOL = 1e-12


def km_curve(observed_h: np.ndarray, ended: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of P(stay > t).

    Returns the distinct lengths of the stays that ended, in increasing order, and the
    estimated probability of staying longer than each, just after it.
    """
    observed_h = np.asarray(observed_h, dtype=float)
    ended = np.asarray(ended, dtype=bool)
    if observed_h.shape != ended.shape:
        raise ValueError("observed_h and ended must have the same shape")
    times, leaving = np.unique(observed_h[ended], return_counts=True)
    if times.size == 0:
        return times, np.empty(0)
    at_risk = observed_h.size - np.searchsorted(np.sort(observed_h), times, side="left")
    return times, np.cumprod(1.0 - leaving / at_risk)


def lower_quantile(
    times: np.ndarray, survival: np.ndarray, level: float, elapsed_h: float = 0.0
) -> float | None:
    """The largest promise X with estimated P(stay < elapsed + X | stay >= elapsed) <= level.

    This is the one-sided promise "a car that has been here `elapsed_h` hours will stay at
    least X more", set so that it is estimated to fail at most a `level` share of the time.
    Returns None when the data cannot identify X (too few stays have ended).
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if elapsed_h < 0:
        raise ValueError("elapsed_h must be non-negative")
    start = int(np.searchsorted(times, elapsed_h, side="left"))
    surviving = 1.0 if start == 0 else float(survival[start - 1])
    if surviving <= 0.0:
        return None
    threshold = (1.0 - level) * surviving * (1.0 - _REL_TOL)
    below = np.flatnonzero(survival[start:] < threshold)
    if below.size == 0:
        return None
    return float(times[start + below[0]] - elapsed_h)
