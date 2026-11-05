"""Per-car bounds on remaining stay: the Kaplan-Meier population rule and its baselines.

At a decision time, recent cars are often still plugged in: their stay is only known to be
at least as long as it has lasted so far. Dropping them would bias estimates towards short
stays (only the cars that left early would count). The Kaplan-Meier estimator uses them
correctly: a car still plugged in counts as "at risk of leaving" up to its time so far.

A bound here is one-sided: "a car that has been plugged in for `elapsed` hours will stay at
least X more", set as a lower quantile of remaining stay given the time already plugged in.
The rules below differ only in which earlier sessions they learn from. Their settings
(window lengths, level, fit dates) are fixed by the analysis plan, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleetoffer.data import timestamp_to_hours
from fleetoffer.interface import History, Request, Rule

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
    """The largest X with estimated P(stay < elapsed + X | stay >= elapsed) <= level.

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


class RollingQuantile(Rule):
    """Baseline: learn from the sessions that plugged in during the last `window_days` days."""

    def __init__(self, window_days: float, level: float = 0.1, min_sessions: int = 1) -> None:
        if window_days <= 0:
            raise ValueError("window_days must be positive")
        self.window_h = 24.0 * window_days
        self.level = level
        self.min_sessions = min_sessions
        self.name = f"rolling {window_days:g} d"

    def bound(self, request: Request, history: History) -> float | None:
        window = history.window(since_h=history.now_h - self.window_h)
        if len(window) < self.min_sessions:
            return None
        times, survival = km_curve(window.observed_h, window.ended)
        return lower_quantile(times, survival, self.level, request.elapsed_h)


class FrozenQuantile(Rule):
    """Baseline: fitted once and never updated.

    The fit uses the sessions that plugged in between `fit_since` and `fit_until`, as known
    at the first decision on or after `fit_until`. Before that, the rule gives no bound.
    """

    def __init__(
        self,
        fit_until: pd.Timestamp | str,
        level: float = 0.1,
        fit_since: pd.Timestamp | str | None = None,
    ) -> None:
        self.fit_until_h = timestamp_to_hours(fit_until)
        self.fit_since_h = -np.inf if fit_since is None else timestamp_to_hours(fit_since)
        self.level = level
        self.name = "frozen"
        self._curve: tuple[np.ndarray, np.ndarray] | None = None

    def bound(self, request: Request, history: History) -> float | None:
        if history.now_h < self.fit_until_h:
            return None
        if self._curve is None:
            window = history.window(since_h=self.fit_since_h, until_h=self.fit_until_h)
            self._curve = km_curve(window.observed_h, window.ended)
        return lower_quantile(*self._curve, self.level, request.elapsed_h)
