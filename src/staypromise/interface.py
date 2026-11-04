"""What a rule is allowed to see.

A rule answers one question at a time: "how many more hours can be promised that this car
will stay?" It is given three things, and nothing else:

- a `Request`: the car being asked about, with only what is known at the decision time;
- a `History`: every session that plugged in strictly before the decision time, with its
  stay length if it has already ended, or the time it has been plugged in so far if it has
  not (a censored stay);
- the `Outcome` of each of its own earlier promises, delivered at the moment the outcome
  becomes known: when the car leaves, if before the promised time, or when the promised
  time passes with the car still there.

The `History` is built from the replay engine's running record, which learns a departure
only when it happens, so a rule cannot see the future through it. Events at exactly the
decision time count as not yet known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Request:
    """The car a promise is asked for."""

    session_id: str
    user_id: str | None
    connect_h: float
    decision_h: float

    @property
    def elapsed_h(self) -> float:
        """Hours the car has already been plugged in at the decision time."""
        return self.decision_h - self.connect_h


@dataclass(frozen=True, slots=True)
class Outcome:
    """Whether an earlier promise was kept, delivered when that became known."""

    session_id: str
    decision_h: float
    promised_h: float
    kept: bool
    revealed_h: float


@dataclass(frozen=True, slots=True)
class Window:
    """Earlier sessions as known at the decision time (arrays share one order)."""

    connect_h: np.ndarray
    observed_h: np.ndarray  # stay length if ended, else hours plugged in so far
    ended: np.ndarray  # True if the car had left before the decision time
    user_id: np.ndarray

    def __len__(self) -> int:
        return len(self.connect_h)


class History:
    """Read-only view of earlier sessions, valid only during one `Rule.promise` call."""

    __slots__ = ("_connect_h", "_ended_at_h", "_user_id", "_now_h", "_expired")

    def __init__(
        self,
        connect_h: np.ndarray,
        ended_at_h: np.ndarray,
        user_id: np.ndarray,
        now_h: float,
    ) -> None:
        self._connect_h = connect_h
        self._ended_at_h = ended_at_h
        self._user_id = user_id
        self._now_h = now_h
        self._expired = False

    @property
    def now_h(self) -> float:
        return self._now_h

    def window(self, since_h: float = -np.inf, until_h: float = np.inf) -> Window:
        """Sessions that plugged in at or after `since_h` and before `min(until_h, now)`."""
        if self._expired:
            raise RuntimeError("a History can only be read during the promise call it was made for")
        until_h = min(until_h, self._now_h)
        lo = int(np.searchsorted(self._connect_h, since_h, side="left"))
        hi = int(np.searchsorted(self._connect_h, until_h, side="left"))
        hi = max(lo, hi)
        connect = self._connect_h[lo:hi].copy()
        ended_at = self._ended_at_h[lo:hi]
        ended = ~np.isnan(ended_at)
        observed = np.where(ended, ended_at - connect, self._now_h - connect)
        return Window(connect, observed, ended, self._user_id[lo:hi].copy())

    def _expire(self) -> None:
        self._expired = True


class Rule:
    """Base class for a promise rule.

    `promise` returns the promised hours (finite, at least 0), or ``None`` to make no
    promise for this car. `observe` receives each earlier promise's outcome when it
    becomes known; rules that do not learn from outcomes can ignore it.
    """

    name: str = "rule"

    def promise(self, request: Request, history: History) -> float | None:
        raise NotImplementedError

    def observe(self, outcome: Outcome) -> None:
        return None
