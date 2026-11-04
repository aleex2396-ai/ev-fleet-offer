"""Promise rules.

Each rule sets the promise as a lower quantile of stay length (see `survival`), and they
differ only in which earlier sessions they learn from. Their settings (window lengths,
level, fit dates) are fixed by the pre-registration, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from staypromise.interface import History, Request, Rule
from staypromise.sessions import timestamp_to_hours
from staypromise.survival import km_curve, lower_quantile


class FixedPromise(Rule):
    """The same promise for every car; useful only as a reference in checks."""

    def __init__(self, hours: float) -> None:
        if hours < 0:
            raise ValueError("hours must be non-negative")
        self.hours = float(hours)
        self.name = f"fixed {hours:g} h"

    def promise(self, request: Request, history: History) -> float:
        return self.hours


class RollingQuantile(Rule):
    """Learn from the sessions that plugged in during the last `window_days` days."""

    def __init__(self, window_days: float, level: float = 0.1, min_sessions: int = 1) -> None:
        if window_days <= 0:
            raise ValueError("window_days must be positive")
        self.window_h = 24.0 * window_days
        self.level = level
        self.min_sessions = min_sessions
        self.name = f"rolling {window_days:g} d"

    def promise(self, request: Request, history: History) -> float | None:
        window = history.window(since_h=history.now_h - self.window_h)
        if len(window) < self.min_sessions:
            return None
        times, survival = km_curve(window.observed_h, window.ended)
        return lower_quantile(times, survival, self.level, request.elapsed_h)


class FrozenQuantile(Rule):
    """Fitted once and never updated.

    The fit uses the sessions that plugged in between `fit_since` and `fit_until`, as known
    at the first decision on or after `fit_until`. Before that, the rule makes no promise.
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

    def promise(self, request: Request, history: History) -> float | None:
        if history.now_h < self.fit_until_h:
            return None
        if self._curve is None:
            window = history.window(since_h=self.fit_since_h, until_h=self.fit_until_h)
            self._curve = km_curve(window.observed_h, window.ended)
        return lower_quantile(*self._curve, self.level, request.elapsed_h)
