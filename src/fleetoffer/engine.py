"""Replay sessions in time order and ask a rule for a bound at each decision.

This is the only module that reads true departure times. It processes three kinds of
event in time order:

- a decision: the rule is asked for a lower bound on one car's remaining stay;
- a departure: the car leaves, and from then on the `History` shows its full stay;
- a reveal: the outcome of an earlier bound becomes known and is passed to the rule.

At equal times, decisions come first, so anything that happens at exactly the decision
time is not yet known to the rule. In particular, a car that plugs in at exactly the
decision time is not yet present, and a car that leaves at exactly the decision time still
is.

Decisions are made on one of three schedules: once per car at plug-in; repeatedly every
`every_h` hours while the car is plugged in; or once a day at a local clock time (`Daily`),
for every car plugged in at that moment, which is the morning decision of the analysis plan.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fleetoffer.data import from_hours, to_hours, validate_sessions
from fleetoffer.interface import History, Outcome, Request, Rule

_DECISION, _DEPARTURE, _REVEAL = 0, 1, 2

DECISION_COLUMNS = [
    "rule",
    "session_id",
    "user_id",
    "connect_time",
    "decision_time",
    "elapsed_h",
    "given",
    "bound_h",
    "held",
    "remaining_h",
]


@dataclass(frozen=True)
class Daily:
    """One decision a day at local clock time `at`, for every car plugged in at that moment."""

    at: str = "10:00"
    tz: str = "America/Los_Angeles"
    weekdays_only: bool = True

    def times_h(self, first_h: float, last_h: float) -> np.ndarray:
        """Decision times, in hours since 1970 UTC, on the local days from `first_h` to `last_h`."""
        first, last = from_hours([first_h, last_h]).tz_convert(self.tz)
        days = pd.date_range(
            first.normalize().tz_localize(None), last.normalize().tz_localize(None)
        )
        if self.weekdays_only:
            days = days[days.day_of_week < 5]
        local = days + pd.Timedelta(self.at + ":00")
        return to_hours(local.tz_localize(self.tz).tz_convert("UTC"))


def decision_times(
    connect_h: np.ndarray,
    disconnect_h: np.ndarray,
    every_h: float | None = None,
    daily: Daily | None = None,
) -> list[tuple[int, float]]:
    """(session index, decision time) pairs.

    With neither `every_h` nor `daily`, each car gets one decision, at plug-in. With
    `every_h`, it also gets one every `every_h` hours while it is still plugged in. With
    `daily`, each car plugged in at a daily decision time gets one decision then.
    """
    if every_h is not None and daily is not None:
        raise ValueError("give every_h or daily, not both")
    if daily is not None:
        if len(connect_h) == 0:
            return []
        out = []
        for t in daily.times_h(float(connect_h.min()), float(disconnect_h.max())):
            present = np.flatnonzero((connect_h < t) & (disconnect_h >= t))
            out.extend((int(i), float(t)) for i in present)
        return out
    if every_h is None:
        return [(i, float(t)) for i, t in enumerate(connect_h)]
    if not every_h > 0:
        raise ValueError("every_h must be positive")
    out = []
    for i, (start, end) in enumerate(zip(connect_h, disconnect_h, strict=True)):
        steps = math.floor((end - start) / every_h)
        out.extend((i, float(start + m * every_h)) for m in range(steps + 1))
    return out


def energy_in_window(
    connect_h: np.ndarray,
    done_h: np.ndarray,
    disconnect_h: np.ndarray,
    kwh: np.ndarray,
    start_h: float | np.ndarray,
    end_h: float | np.ndarray,
) -> np.ndarray:
    """Energy each session took between `start_h` and `end_h`, by the fallback of check D1.

    The per-session time series failed check D1 (analysis plan, Section 10a; `docs/data.md`),
    so the session's recorded energy is spread evenly from plug-in to the end of charging. The
    end of charging is the charging-done time if one is recorded after plug-in, otherwise
    unplug; a charging-done time after unplug counts as unplug. The estimate uses the
    recorded total and the charging-done time, which are known only later, so it is a score
    and a training target, never an input at the decision time.
    """
    connect_h = np.asarray(connect_h, dtype=float)
    disconnect_h = np.asarray(disconnect_h, dtype=float)
    done_h = np.asarray(done_h, dtype=float)
    ends = np.where(np.isfinite(done_h) & (done_h > connect_h), done_h, disconnect_h)
    ends = np.minimum(ends, disconnect_h)
    overlap = np.minimum(ends, end_h) - np.maximum(connect_h, start_h)
    return np.asarray(kwh, dtype=float) * np.clip(overlap, 0.0, None) / (ends - connect_h)


def replay(
    sessions: pd.DataFrame,
    rule: Rule,
    *,
    every_h: float | None = None,
    daily: Daily | None = None,
) -> pd.DataFrame:
    """Run one rule over the sessions and return one row per decision.

    Columns: the rule's name; the car (``session_id``, ``user_id``, ``connect_time``); the
    decision (``decision_time``, ``elapsed_h``); the bound (``given``, ``bound_h``, which is
    missing when no bound was given); and the evaluation, which the rule never saw (``held``;
    ``remaining_h``, the true hours the car stayed after the decision).
    """
    table = validate_sessions(sessions)
    n = len(table)
    connect_h = to_hours(table["connect_time"])
    disconnect_h = to_hours(table["disconnect_time"])
    session_ids = table["session_id"].to_numpy(dtype=object)
    user_ids = table["user_id"].to_numpy(dtype=object)
    ended_at_h = np.full(n, np.nan)

    order = itertools.count()
    events = [(disconnect_h[i], _DEPARTURE, next(order), i, None) for i in range(n)]
    events += [
        (t, _DECISION, next(order), i, None)
        for i, t in decision_times(connect_h, disconnect_h, every_h, daily)
    ]
    heapq.heapify(events)

    rows = []
    while events:
        t, kind, _, i, payload = heapq.heappop(events)
        if kind == _DEPARTURE:
            ended_at_h[i] = disconnect_h[i]
        elif kind == _REVEAL:
            decided_h, bound_h, held = payload
            rule.observe(Outcome(session_ids[i], decided_h, bound_h, held, t))
        else:
            request = Request(session_ids[i], user_ids[i], float(connect_h[i]), t)
            history = History(connect_h, ended_at_h, user_ids, t)
            bound = rule.bound(request, history)
            history._expire()

            if bound is None:
                rows.append((i, t, False, np.nan, pd.NA))
                continue
            bound = float(bound)
            if not (math.isfinite(bound) and bound >= 0):
                raise ValueError(f"{rule.name} gave a bound of {bound!r} hours; must be >= 0")
            held = bool(disconnect_h[i] >= t + bound)
            reveal_h = t + bound if held else disconnect_h[i]
            heapq.heappush(events, (reveal_h, _REVEAL, next(order), i, (t, bound, held)))
            rows.append((i, t, True, bound, held))

    return _decision_table(rule.name, table, connect_h, disconnect_h, rows)


def _decision_table(name, table, connect_h, disconnect_h, rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    idx = np.array([r[0] for r in rows], dtype=int)
    t = np.array([r[1] for r in rows], dtype=float)
    out = pd.DataFrame(
        {
            "rule": name,
            "session_id": table["session_id"].to_numpy()[idx],
            "user_id": table["user_id"].to_numpy()[idx],
            "connect_time": table["connect_time"].to_numpy()[idx],
            "decision_time": from_hours(t),
            "elapsed_h": t - connect_h[idx],
            "given": np.array([r[2] for r in rows], dtype=bool),
            "bound_h": np.array([r[3] for r in rows], dtype=float),
            "held": pd.array([r[4] for r in rows], dtype="boolean"),
            "remaining_h": disconnect_h[idx] - t,
        }
    )
    out["connect_time"] = pd.to_datetime(out["connect_time"], utc=True)
    return out.sort_values(["decision_time", "session_id"], kind="stable").reset_index(drop=True)
