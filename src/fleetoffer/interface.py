"""What a per-car rule is allowed to see at the decision time.

A rule answers one question at a time: "how many more hours will this car stay, at least?"
It is given three things, and nothing else:

- a `Request`: the car being asked about, with only what is known at the decision time;
- a `History`: every session that plugged in strictly before the decision time, with its
  stay length if it has already ended, or the time it has been plugged in so far if it has
  not (a censored stay);
- the `Outcome` of each of its own earlier bounds, delivered at the moment the outcome
  becomes known: when the car leaves, if before the bound, or when the bound's time passes
  with the car still there.

The `History` is built from the replay engine's running record, which learns a departure
only when it happens, so a rule cannot see the future through it. Events at exactly the
decision time count as not yet known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Request:
    """The car a bound is asked for."""

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
    """Whether an earlier bound held, delivered when that became known."""

    session_id: str
    decision_h: float
    bound_h: float
    held: bool
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
    """Read-only view of earlier sessions, valid only during one `Rule.bound` call."""

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
            raise RuntimeError("a History can only be read during the call it was made for")
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
    """Base class for a per-car rule.

    `bound` returns a lower bound on the car's remaining stay in hours (finite, at least 0),
    or ``None`` to give no bound for this car. `observe` receives each earlier bound's outcome
    when it becomes known; rules that do not learn from outcomes can ignore it.
    """

    name: str = "rule"

    def bound(self, request: Request, history: History) -> float | None:
        raise NotImplementedError

    def observe(self, outcome: Outcome) -> None:
        return None


# The day level: one decision a day for the whole fleet present at that moment.


@dataclass(frozen=True)
class DayView:
    """What is known at a day's decision time, as given to a fleet rule.

    ``present`` has one row per car plugged in at the decision time, with only what is known
    then: ``session_id``, ``connect_time``, ``elapsed_h``, ``station_id`` and ``space_id``
    where the table has them, and ``user_id`` only where the link to the driver was recorded
    before the decision time (a ``linked_time`` column; without one, no link is shown).
    ``ended`` holds every session that had ended before the decision time, with all its
    fields, and ``user_id`` shown on the same condition.
    """

    decision_time: pd.Timestamp
    present: pd.DataFrame
    ended: pd.DataFrame


@dataclass(frozen=True)
class DayOutcome:
    """A day's per-car bounds against what happened, given to the rule once all is known.

    Deadlines are in steps of the horizon, energies in kWh between the decision time and the
    end of the horizon. ``revealed_at`` is when the last input to the outcome became known:
    the end of the horizon, or the latest unplug among the day's cars if later, because the
    energy estimate uses each session's recorded total (`docs/data.md`, check D1).
    """

    decision_time: pd.Timestamp
    session_id: np.ndarray
    learned_deadline: np.ndarray
    learned_energy: np.ndarray
    true_deadline: np.ndarray
    true_energy: np.ndarray
    revealed_at: pd.Timestamp


class FleetRule:
    """Base class for a day-level rule: per-car bounds for every car present at 10:00.

    `bounds` returns two arrays in the order of ``day.present``: a lower bound on each car's
    remaining stay in hours and a lower bound on the energy it will take before the end of
    the horizon in kWh (both finite and at least 0). `observe` receives each earlier day's
    outcome once it is known.
    """

    name: str = "fleet rule"

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def observe(self, outcome: DayOutcome) -> None:
        return None
