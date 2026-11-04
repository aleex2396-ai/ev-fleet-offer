"""Replay sessions in time order and ask a rule for a promise at each decision.

This is the only module that reads true departure times. It processes three kinds of
event in time order:

- a decision: the rule is asked for a promise about one car;
- a departure: the car leaves, and from then on the `History` shows its full stay;
- a reveal: the outcome of an earlier promise becomes known and is passed to the rule.

At equal times, decisions come first, so anything that happens at exactly the decision
time is not yet known to the rule.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np
import pandas as pd

from staypromise.interface import History, Outcome, Request, Rule
from staypromise.sessions import from_hours, to_hours, validate_sessions

_DECISION, _DEPARTURE, _REVEAL = 0, 1, 2

DECISION_COLUMNS = [
    "rule",
    "session_id",
    "user_id",
    "connect_time",
    "decision_time",
    "elapsed_h",
    "offered",
    "promised_h",
    "kept",
    "remaining_h",
]


def decision_times(connect_h: np.ndarray, disconnect_h: np.ndarray, every_h: float | None):
    """(session index, decision time) pairs.

    With ``every_h=None`` each car gets one decision, at plug-in. Otherwise it also gets one
    every ``every_h`` hours while it is still plugged in (a car leaving at exactly a
    decision time still counts as plugged in, as the departure is not yet known).
    """
    if every_h is None:
        return [(i, float(t)) for i, t in enumerate(connect_h)]
    if not every_h > 0:
        raise ValueError("every_h must be positive")
    out = []
    for i, (start, end) in enumerate(zip(connect_h, disconnect_h, strict=True)):
        steps = math.floor((end - start) / every_h)
        out.extend((i, float(start + m * every_h)) for m in range(steps + 1))
    return out


def replay(sessions: pd.DataFrame, rule: Rule, *, every_h: float | None = None) -> pd.DataFrame:
    """Run one rule over the sessions and return one row per decision.

    Columns: the rule's name; the car (``session_id``, ``user_id``, ``connect_time``); the
    decision (``decision_time``, ``elapsed_h``); the promise (``offered``, ``promised_h``,
    which is missing when no promise was made); and the evaluation, which the rule never
    saw (``kept``; ``remaining_h``, the true hours the car stayed after the decision).
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
        for i, t in decision_times(connect_h, disconnect_h, every_h)
    ]
    heapq.heapify(events)

    rows = []
    while events:
        t, kind, _, i, payload = heapq.heappop(events)
        if kind == _DEPARTURE:
            ended_at_h[i] = disconnect_h[i]
        elif kind == _REVEAL:
            decided_h, promised_h, kept = payload
            rule.observe(Outcome(session_ids[i], decided_h, promised_h, kept, t))
        else:
            request = Request(session_ids[i], user_ids[i], float(connect_h[i]), t)
            history = History(connect_h, ended_at_h, user_ids, t)
            promised = rule.promise(request, history)
            history._expire()

            if promised is None:
                rows.append((i, t, False, np.nan, pd.NA))
                continue
            promised = float(promised)
            if not (math.isfinite(promised) and promised >= 0):
                raise ValueError(
                    f"{rule.name} promised {promised!r} hours; must be finite and >= 0"
                )
            kept = bool(disconnect_h[i] >= t + promised)
            reveal_h = t + promised if kept else disconnect_h[i]
            heapq.heappush(events, (reveal_h, _REVEAL, next(order), i, (t, promised, kept)))
            rows.append((i, t, True, promised, kept))

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
            "offered": np.array([r[2] for r in rows], dtype=bool),
            "promised_h": np.array([r[3] for r in rows], dtype=float),
            "kept": pd.array([r[4] for r in rows], dtype="boolean"),
            "remaining_h": disconnect_h[idx] - t,
        }
    )
    out["connect_time"] = pd.to_datetime(out["connect_time"], utc=True)
    return out.sort_values(["decision_time", "session_id"], kind="stable").reset_index(drop=True)
