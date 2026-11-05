"""The look-ahead check: a rule's bounds must not depend on anything that has not happened.

For each of several cutoff times, the check replays a rule twice: once on the real session
table, and once on a copy in which everything at or after the cutoff has been scrambled
(departures moved to just after the cutoff or far beyond it, later sessions dropped and
made-up ones added). Every bound given before the cutoff must come out identical. If one
differs, the rule used information from the future. The later values are drawn from a
different distribution, not permuted, so a rule that uses only counts of later events is
caught too.

A cutoff only exposes decisions shortly before it (those are the ones with cars still
plugged in across it), so the default uses several cutoffs spread over busy times.

The rule factory is given the table being replayed, so the check also catches rules whose
settings were fitted on the whole data set, the time-series version of tuning on the test
set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fleetoffer.data import validate_sessions
from fleetoffer.engine import Daily, replay
from fleetoffer.interface import Rule


class LookaheadError(AssertionError):
    """A rule's bounds before the cutoff changed when only the future changed."""


def scramble_future(sessions: pd.DataFrame, cutoff: pd.Timestamp, *, seed: int = 0) -> pd.DataFrame:
    """A copy of the table that agrees with it on everything known before `cutoff`."""
    table = validate_sessions(sessions)
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
    rng = np.random.default_rng(seed)

    # Departures at or after the cutoff are re-drawn, still at or after it: half to within
    # three minutes of it, half to many hours later, so any use of them moves the bounds.
    later = table["disconnect_time"] >= cutoff
    n_later = int(later.sum())
    floor = table.loc[later, "connect_time"].where(
        table.loc[later, "connect_time"] > cutoff, cutoff
    )
    soon = rng.random(n_later) < 0.5
    redraw = np.where(soon, rng.uniform(0.0, 0.05, n_later), rng.uniform(12.0, 48.0, n_later))
    table.loc[later, "disconnect_time"] = floor + pd.to_timedelta(redraw + 1e-4, unit="h")

    # Half of the sessions that plug in at or after the cutoff are dropped and more made-up
    # ones are added than were dropped, so the number of later sessions changes too.
    future = table.index[table["connect_time"] >= cutoff]
    drop = rng.choice(future, size=len(future) // 2, replace=False) if len(future) else []
    table = table.drop(index=drop)
    n_new = len(future) // 2 + 10
    starts = cutoff + pd.to_timedelta(rng.uniform(0.0, 24.0 * 30, n_new), unit="h")
    added = pd.DataFrame(
        {
            "session_id": [f"scrambled-{k}" for k in range(n_new)],
            "connect_time": starts,
            "disconnect_time": starts
            + pd.to_timedelta(rng.exponential(6.0, n_new) + 0.1, unit="h"),
            "user_id": [f"scrambled-user-{k % 7}" for k in range(n_new)],
        }
    )
    combined = pd.concat([table, added], ignore_index=True)
    return validate_sessions(combined) if len(combined) else combined


def default_cutoffs(sessions: pd.DataFrame, count: int = 6) -> list[pd.Timestamp]:
    """Cutoffs spread over the data, each two hours after some car plugged in (a busy time)."""
    table = validate_sessions(sessions)
    shares = np.linspace(0.1, 0.9, count)
    picks = table["connect_time"].iloc[(shares * (len(table) - 1)).astype(int)]
    return [t + pd.Timedelta(hours=2) for t in picks]


def check_no_lookahead(
    rule_factory: Callable[[pd.DataFrame], Rule],
    sessions: pd.DataFrame,
    cutoffs: pd.Timestamp | list[pd.Timestamp] | None = None,
    *,
    every_h: float | None = None,
    daily: Daily | None = None,
    seed: int = 0,
) -> int:
    """Raise `LookaheadError` if any bound before a cutoff depends on what follows it.

    Returns the number of decisions compared, so a caller can make sure it was not zero.
    """
    if cutoffs is None:
        cutoffs = default_cutoffs(sessions)
    elif isinstance(cutoffs, pd.Timestamp):
        cutoffs = [cutoffs]
    return sum(
        _check_one_cutoff(
            rule_factory, sessions, cutoff, every_h=every_h, daily=daily, seed=seed + k
        )
        for k, cutoff in enumerate(cutoffs)
    )


def _check_one_cutoff(rule_factory, sessions, cutoff, *, every_h, daily, seed) -> int:
    real = validate_sessions(sessions)
    scrambled = scramble_future(real, cutoff, seed=seed)
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")

    def before(table: pd.DataFrame) -> pd.DataFrame:
        decisions = replay(table, rule_factory(table), every_h=every_h, daily=daily)
        early = decisions[decisions["decision_time"] < cutoff]
        cols = ["session_id", "decision_time", "given", "bound_h"]
        return early[cols].sort_values(["decision_time", "session_id"]).reset_index(drop=True)

    a, b = before(real), before(scrambled)
    if (
        len(a) != len(b)
        or not (
            a[["session_id", "decision_time", "given"]]
            == b[["session_id", "decision_time", "given"]]
        )
        .all()
        .all()
    ):
        raise LookaheadError("the set of decisions before the cutoff changed")
    same = np.isclose(a["bound_h"], b["bound_h"], rtol=0, atol=0, equal_nan=True)
    if not same.all():
        first = a.loc[~same].iloc[0]
        raise LookaheadError(
            f"{int((~same).sum())} of {len(a)} bounds before the cutoff changed when only "
            f"the future changed; first: session {first.session_id} at {first.decision_time}"
        )
    return len(a)
