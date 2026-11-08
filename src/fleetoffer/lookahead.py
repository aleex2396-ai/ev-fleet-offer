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

from fleetoffer.data import from_hours, to_hours, validate_sessions
from fleetoffer.engine import MORNING, Daily, fleet_replay, replay
from fleetoffer.interface import FleetRule, Rule


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

    # Fields recorded at or after the cutoff are re-drawn too: the energy total (recorded at
    # unplug) of every session still plugged in, and a charging-done time or driver link not
    # yet recorded, each moved to between the cutoff and the new unplug time.
    if "kwh_delivered" in table.columns:
        table.loc[later, "kwh_delivered"] = rng.uniform(30.0, 60.0, n_later)
    for col in ("done_charging_time", "linked_time"):
        if col in table.columns:
            unknown = later & ~(table[col] < cutoff)
            start = table.loc[unknown, "connect_time"].where(
                table.loc[unknown, "connect_time"] > cutoff, cutoff
            )
            span = table.loc[unknown, "disconnect_time"] - start
            table.loc[unknown, col] = start + span * rng.uniform(0.0, 1.0, int(unknown.sum()))

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
    if "kwh_delivered" in table.columns:
        added["kwh_delivered"] = rng.uniform(30.0, 60.0, n_new)
    if "done_charging_time" in table.columns:
        span = added["disconnect_time"] - added["connect_time"]
        added["done_charging_time"] = added["connect_time"] + span * rng.uniform(0.1, 1.0, n_new)
    if "linked_time" in table.columns:
        added["linked_time"] = added["connect_time"]
    if "station_id" in table.columns:
        added["station_id"] = "scrambled-station"
    combined = pd.concat([table, added], ignore_index=True)
    return validate_sessions(combined) if len(combined) else combined


def scramble_entries(entries: pd.DataFrame, cutoff: pd.Timestamp, *, seed: int = 0) -> pd.DataFrame:
    """A copy of the app entries that agrees with them on every entry recorded before
    `cutoff`: later entries get new values and times (still at or after the cutoff), half of
    them are dropped, and made-up entries are added for made-up sessions."""
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
    rng = np.random.default_rng([seed, 2])
    out = entries.copy()
    later = out.index[out["modified_at"] >= cutoff]
    out = out.drop(index=rng.choice(later, size=len(later) // 2, replace=False))
    later = out["modified_at"] >= cutoff
    n = int(later.sum())
    out.loc[later, "modified_at"] = cutoff + pd.to_timedelta(rng.uniform(0.0, 2.0, n), unit="h")
    minutes = rng.choice([30.0, 60.0, 480.0, 900.0], size=n)
    out.loc[later, "minutes_available"] = minutes
    out.loc[later, "requested_departure"] = out.loc[later, "modified_at"] + pd.to_timedelta(
        minutes, unit="m"
    )
    out.loc[later, "kwh_requested"] = rng.uniform(30.0, 60.0, n)
    starts = cutoff + pd.to_timedelta(rng.uniform(0.0, 24.0 * 30, 10), unit="h")
    added = pd.DataFrame(
        {
            "session_id": [f"scrambled-{k}" for k in range(10)],
            "entry": 0,
            "modified_at": starts,
            "minutes_available": 600.0,
            "requested_departure": starts + pd.Timedelta(hours=10),
            "kwh_requested": 50.0,
        }
    )
    out = pd.concat([out, added], ignore_index=True)
    return out.sort_values("modified_at", kind="stable").reset_index(drop=True)


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


def default_day_cutoffs(sessions: pd.DataFrame, daily: Daily, count: int = 4) -> list[pd.Timestamp]:
    """Cutoffs at decision days spread over the data: alternately four hours after the
    decision (that day's cars still plugged in, its outcome unknown) and one hour before it
    (earlier days' outcomes possibly still pending)."""
    table = validate_sessions(sessions)
    times = daily.times_h(
        float(to_hours(table["connect_time"]).min()),
        float(to_hours(table["disconnect_time"]).max()),
    )
    picks = times[(np.linspace(0.3, 0.9, count) * (len(times) - 1)).astype(int)]
    shifts = [4.0 if k % 2 == 0 else -1.0 for k in range(count)]
    return [from_hours([t + s])[0] for t, s in zip(picks, shifts, strict=True)]


def check_no_lookahead_days(
    rule_factory: Callable[[pd.DataFrame], FleetRule],
    sessions: pd.DataFrame,
    cutoffs: list[pd.Timestamp] | None = None,
    *,
    seed: int = 0,
    **fleet_options,
) -> int:
    """Raise `LookaheadError` if any offer made before a cutoff depends on what follows it.

    The day-level version of the check (analysis plan, Section 7): the whole pipeline is run
    on the real table and on a copy scrambled after the cutoff, linear programme included,
    and every day before the cutoff must have the same cars, the same per-car bounds and the
    same programme output to the last bit. App entries given as ``entries`` are scrambled
    after the cutoff too. Returns the number of days compared.
    """
    daily = fleet_options.get("daily", MORNING)
    entries = fleet_options.pop("entries", None)
    cutoffs = default_day_cutoffs(sessions, daily) if cutoffs is None else cutoffs
    compared = 0
    for k, cutoff in enumerate(cutoffs):
        real = validate_sessions(sessions)
        scrambled = scramble_future(real, cutoff, seed=seed + k)
        later = None if entries is None else scramble_entries(entries, cutoff, seed=seed + k)
        cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
        a, b = (
            fleet_replay(table, rule_factory(table), entries=typed, **fleet_options)
            for table, typed in ((real, entries), (scrambled, later))
        )
        days_a, days_b = (r.days[r.days["decision_time"] < cutoff] for r in (a, b))
        cars_a, cars_b = (r.cars[r.cars["decision_time"] < cutoff] for r in (a, b))
        day_cols = ["decision_time", "cars", "digest"]
        car_cols = ["decision_time", "session_id", "learned_deadline", "learned_energy_kwh"]
        same_days = (
            days_a[day_cols].reset_index(drop=True).equals(days_b[day_cols].reset_index(drop=True))
        )
        same_cars = (
            cars_a[car_cols].reset_index(drop=True).equals(cars_b[car_cols].reset_index(drop=True))
        )
        if not (same_days and same_cars):
            raise LookaheadError(
                f"an offer made before {cutoff} changed when only later data changed"
            )
        compared += len(days_a)
    return compared
