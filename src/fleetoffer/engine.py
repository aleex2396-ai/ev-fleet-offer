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

import hashlib
import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fleetoffer.data import from_hours, timestamp_to_hours, to_hours, validate_sessions
from fleetoffer.interface import DayOutcome, DayView, FleetRule, History, Outcome, Request, Rule
from fleetoffer.offer import (
    CAP_KW,
    SIZE_CHECKPOINTS,
    STEP_H,
    STEPS,
    BaseSet,
    Fleet,
    check_delivery,
    solve,
    solve_segment,
)

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


# The day level (analysis plan, Sections 3 and 7).

MORNING = Daily()  # 10:00 Pacific on weekdays

DAY_COLUMNS = [
    "rule",
    "decision_time",
    "cars",
    "method",
    "alpha",
    "size_kwh",
    "exact_true_kwh",
    "exact_learned_kwh",
    "all_bounds_held",
    "fixed_delivered",
    "fixed_shortfall_kwh",
    "hindsight_delivered",
    "hindsight_shortfall_kwh",
    "profile_energy_kwh",
    "solve_seconds",
    "revealed_at",
    "digest",
]
CAR_COLUMNS = [
    "rule",
    "decision_time",
    "session_id",
    "learned_deadline",
    "true_deadline",
    "learned_energy_kwh",
    "true_energy_kwh",
    "deadline_held",
    "energy_held",
]
_KNOWN_ON_ARRIVAL = ("session_id", "connect_time", "station_id", "space_id")
_ENERGY_TOL_KWH = 1e-9


class Oracle(FleetRule):
    """Arm 1 of the analysis plan: each car's true set. Not a rule an aggregator could run;
    the engine, the only reader of unplug times, gives it the true sets itself."""

    name = "oracle"


@dataclass(frozen=True)
class FleetResult:
    days: pd.DataFrame  # one row per decision day
    cars: pd.DataFrame  # one row per car present at a decision


def _view(
    table: pd.DataFrame,
    t: pd.Timestamp,
    present: np.ndarray,
    ended: np.ndarray,
    entries: pd.DataFrame | None = None,
):
    def linked(rows: pd.DataFrame) -> pd.Series:
        if "linked_time" not in rows.columns:
            return pd.Series(None, index=rows.index, dtype=object)
        known = (rows["linked_time"].notna() & (rows["linked_time"] < t)).to_numpy()
        users = rows["user_id"].to_numpy(object)
        shown = [u if k and isinstance(u, str) else None for u, k in zip(users, known, strict=True)]
        return pd.Series(shown, index=rows.index, dtype=object)

    now = table.loc[present, [c for c in _KNOWN_ON_ARRIVAL if c in table.columns]].copy()
    now["user_id"] = linked(table.loc[present])
    now["elapsed_h"] = (t - now["connect_time"]) / pd.Timedelta(hours=1)
    past = table.loc[ended].copy()
    past["user_id"] = linked(past)
    if "linked_time" in past.columns:
        past["linked_time"] = past["linked_time"].where(past["linked_time"] < t)
    if entries is None:
        typed = pd.DataFrame()
    else:
        typed = entries.iloc[: int(entries["modified_at"].searchsorted(t, side="left"))]
        typed = typed.reset_index(drop=True)
    return DayView(t, now.reset_index(drop=True), past.reset_index(drop=True), typed)


def _steps(hours: np.ndarray, step_h: float, steps: int) -> np.ndarray:
    """Hours to whole steps, rounded down (a small tolerance absorbs float error)."""
    return np.clip(np.floor(np.asarray(hours, dtype=float) / step_h + 1e-9), 0, steps).astype(int)


def true_fleet(table: pd.DataFrame, t_h: float, rows: np.ndarray, *, cap, steps, step_h) -> Fleet:
    """The true sets of the given cars at decision time t_h (hours): deadline at unplug or
    the end of the horizon, rounded down to the grid; energy by check D1's fallback."""
    horizon_h = steps * step_h
    connect = to_hours(table.loc[rows, "connect_time"])
    disconnect = to_hours(table.loc[rows, "disconnect_time"])
    done = to_hours(table.loc[rows, "done_charging_time"])
    energy = energy_in_window(
        connect,
        done,
        disconnect,
        table.loc[rows, "kwh_delivered"].to_numpy(float),
        t_h,
        t_h + horizon_h,
    )
    deadline = _steps(np.minimum(disconnect - t_h, horizon_h), step_h, steps)
    return Fleet.of(deadline, energy, cap=cap, steps=steps, step_h=step_h)


def training_base_set(
    sessions: pd.DataFrame,
    until: pd.Timestamp,
    *,
    daily: Daily = MORNING,
    cap: float = CAP_KW,
    steps: int = STEPS,
    step_h: float = STEP_H,
) -> BaseSet:
    """Equation (11) over the training period: the average of the true vectors of every car
    present at a decision time before `until` (analysis plan, Section 3)."""
    table = validate_sessions(sessions)
    connect_h, disconnect_h = to_hours(table["connect_time"]), to_hours(table["disconnect_time"])
    last = min(float(disconnect_h.max()), timestamp_to_hours(until))
    fleets = []
    for t in daily.times_h(float(connect_h.min()), last):
        if t >= timestamp_to_hours(until) or t + steps * step_h > timestamp_to_hours(until):
            continue
        rows = np.flatnonzero((connect_h < t) & (disconnect_h >= t))
        if len(rows):
            fleets.append(true_fleet(table, t, rows, cap=cap, steps=steps, step_h=step_h))
    if not fleets:
        raise ValueError("no car was present at a decision time in the training period")
    pooled = Fleet.of(
        np.concatenate([f.deadline for f in fleets]),
        np.concatenate([f.energy for f in fleets]),
        cap=np.concatenate([f.cap for f in fleets]),
        steps=steps,
        step_h=step_h,
    )
    return BaseSet.average(pooled)


def fleet_replay(
    sessions: pd.DataFrame,
    rule: FleetRule,
    *,
    daily: Daily = MORNING,
    base: BaseSet | None = None,
    method: str = "programme",
    entries: pd.DataFrame | None = None,
    until: pd.Timestamp | None = None,
    cap: float = CAP_KW,
    steps: int = STEPS,
    step_h: float = STEP_H,
) -> FleetResult:
    """Run one fleet rule over the decision days and score each day's offer.

    Each decision day, in time order: earlier days' outcomes known by now are passed to the
    rule; the rule gives per-car bounds from what is known at the decision time, including
    the app entries recorded by then (`entries`, as made by `data.app_entries`); the learned
    sets give the offer (programme (24) on the fixed base set `base`, or on the average of the
    day's learned sets if `base` is None, the paper's own choice; or the one-dimensional
    fallback if `method` is "segment"); the offer is checked against the true sets. With
    `method` "bounds" no offer is made and only the per-car bounds are scored (the per-car
    layer). A day's outcome is passed to the rule once every input to it is known
    (`DayOutcome`). Decisions stop before `until`, if given.
    """
    if method not in ("programme", "segment", "bounds"):
        raise ValueError("method must be 'programme', 'segment' or 'bounds'")
    table = validate_sessions(sessions)
    for col in ("kwh_delivered", "done_charging_time"):
        if col not in table.columns:
            raise ValueError(f"the day level needs the column {col}")
    if entries is not None:
        entries = entries.sort_values("modified_at", kind="stable").reset_index(drop=True)
    connect_h, disconnect_h = to_hours(table["connect_time"]), to_hours(table["disconnect_time"])
    horizon_h = steps * step_h
    pending: list[tuple[float, int, DayOutcome]] = []
    order = itertools.count()
    day_rows, car_rows = [], []

    times = daily.times_h(float(connect_h.min()), float(disconnect_h.max())) if len(table) else []
    if until is not None:
        times = [t for t in times if t < timestamp_to_hours(until)]
    for t in times:
        while pending and pending[0][0] < t:
            rule.observe(heapq.heappop(pending)[2])
        when = from_hours([t])[0]
        present = np.flatnonzero((connect_h < t) & (disconnect_h >= t))
        if len(present) == 0:
            day_rows.append({"rule": rule.name, "decision_time": when, "cars": 0, "method": method})
            continue
        true = true_fleet(table, t, present, cap=cap, steps=steps, step_h=step_h)
        if isinstance(rule, Oracle):
            learned = true
        else:
            view = _view(table, when, present, np.flatnonzero(disconnect_h < t), entries)
            stay_h, energy = (np.asarray(b, dtype=float) for b in rule.bounds(view))
            if stay_h.shape != (len(present),) or energy.shape != (len(present),):
                raise ValueError(f"{rule.name} must give one stay and one energy bound per car")
            if not (np.isfinite(stay_h).all() and np.isfinite(energy).all()):
                raise ValueError(f"{rule.name} gave a bound that is not finite")
            if (stay_h < 0).any() or (energy < 0).any():
                raise ValueError(f"{rule.name} gave a negative bound")
            learned = Fleet.of(
                _steps(np.minimum(stay_h, horizon_h), step_h, steps), energy, cap, steps, step_h
            )

        offer = None
        if method == "segment":
            offer = solve_segment(learned)
            alpha, digest = np.nan, _digest(offer.amplitude, offer.b, offer.w)
        elif method == "programme":
            offer = solve(learned, base if base is not None else BaseSet.average(learned))
            alpha, digest = offer.alpha, offer.digest()
        else:
            alpha, digest = np.nan, _digest(learned.deadline, learned.energy)
        deadline_held = learned.deadline <= true.deadline
        energy_held = learned.energy <= true.energy + _ENERGY_TOL_KWH
        revealed = max(t + horizon_h, float(disconnect_h[present].max()))
        revealed_at = max(
            when + pd.Timedelta(hours=horizon_h), table["disconnect_time"].iloc[present].max()
        )
        sizes = SIZE_CHECKPOINTS
        row = {
            "rule": rule.name,
            "decision_time": when,
            "cars": len(present),
            "method": method,
            "alpha": alpha,
            "exact_true_kwh": float(np.mean([true.band_width(k) for k in sizes])),
            "exact_learned_kwh": float(np.mean([learned.band_width(k) for k in sizes])),
            "all_bounds_held": bool(deadline_held.all() and energy_held.all()),
            "revealed_at": revealed_at,
            "digest": digest,
        }
        if offer is not None:
            delivery = check_delivery(offer, true)
            row |= {
                "size_kwh": offer.size(),
                "fixed_delivered": delivery.fixed_delivered,
                "fixed_shortfall_kwh": max(delivery.fixed_kwh.values()),
                "hindsight_delivered": delivery.hindsight_delivered,
                "hindsight_shortfall_kwh": max(delivery.hindsight_kwh.values()),
                "profile_energy_kwh": float(np.mean(list(delivery.energy_kwh.values()))),
                "solve_seconds": offer.seconds,
            }
        day_rows.append(row)
        ids = table["session_id"].to_numpy()[present]
        car_rows.append(
            pd.DataFrame(
                {
                    "rule": rule.name,
                    "decision_time": when,
                    "session_id": ids,
                    "learned_deadline": learned.deadline,
                    "true_deadline": true.deadline,
                    "learned_energy_kwh": learned.energy,
                    "true_energy_kwh": true.energy,
                    "deadline_held": deadline_held,
                    "energy_held": energy_held,
                }
            )
        )
        outcome = DayOutcome(
            when,
            ids,
            learned.deadline,
            learned.energy,
            true.deadline,
            true.energy,
            revealed_at,
        )
        heapq.heappush(pending, (revealed, next(order), outcome))

    days = pd.DataFrame(day_rows, columns=DAY_COLUMNS)
    cars = pd.concat(car_rows, ignore_index=True) if car_rows else pd.DataFrame(columns=CAR_COLUMNS)
    return FleetResult(days, cars)


def _digest(*arrays) -> str:
    parts = [np.atleast_1d(np.asarray(a, dtype=float)).ravel() for a in arrays]
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest()
