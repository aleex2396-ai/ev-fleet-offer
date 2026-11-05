"""Made-up charging sessions with behaviour that is known exactly.

Used to check every method where the answer is known, before any real data are spent.
Nothing here is a model of any real site.

Behaviour is a sequence of regimes. Within a regime, cars arrive on each day in a Poisson
number (fewer at weekends), mostly in a morning peak, and stay for a log-normal number of
hours whose median can drift by a fixed share per week. A new regime can change all of
this at once, and replace part of the driver population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Regime:
    """One stretch of behaviour, from `start` (a local date) until the next regime starts."""

    start: str
    sessions_per_weekday: float
    median_stay_h: float
    stay_spread: float = 0.6  # standard deviation of log stay length
    drift_per_week: float = 0.0  # change in log median stay per week (0.01 is about +1%)
    weekend_factor: float = 0.3
    drivers: int = 300
    returning_share: float = 1.0  # share of drivers carried over from the previous regime


def make_sessions(
    regimes: list[Regime],
    end: str,
    *,
    tz: str = "America/Los_Angeles",
    seed: int = 0,
    max_stay_h: float = 96.0,
) -> pd.DataFrame:
    """A session table covering each regime's days up to `end` (a local date, exclusive)."""
    if not regimes:
        raise ValueError("need at least one regime")
    starts = [pd.Timestamp(r.start) for r in regimes]
    if starts != sorted(starts):
        raise ValueError("regimes must be in date order")
    rng = np.random.default_rng(seed)
    pools = _driver_pools(regimes, rng)

    frames = []
    bounds = [*starts[1:], pd.Timestamp(end)]
    for regime, first_day, stop_day, pool in zip(regimes, starts, bounds, pools, strict=True):
        days = pd.date_range(first_day, stop_day, freq="D", inclusive="left")
        if len(days) == 0:
            continue
        rate = (
            np.where(days.day_of_week < 5, 1.0, regime.weekend_factor) * regime.sessions_per_weekday
        )
        counts = rng.poisson(rate)
        n = int(counts.sum())
        if n == 0:
            continue
        day = np.repeat(days.to_numpy(), counts)
        arrival_h = _arrival_hours(rng, n)
        weeks = (pd.DatetimeIndex(day) - first_day).days.to_numpy() / 7.0
        log_median = np.log(regime.median_stay_h) + regime.drift_per_week * weeks
        stay_h = np.minimum(np.exp(rng.normal(log_median, regime.stay_spread)), max_stay_h)
        stay_h = np.maximum(stay_h, 1.0 / 60.0)
        local = pd.DatetimeIndex(day) + pd.to_timedelta(arrival_h, unit="h")
        # Clock times skipped or repeated by daylight saving: shift forward / take standard time.
        connect = local.tz_localize(tz, nonexistent="shift_forward", ambiguous=np.zeros(n, bool))
        frames.append(
            pd.DataFrame(
                {
                    "connect_time": connect,
                    "disconnect_time": connect + pd.to_timedelta(stay_h, unit="h"),
                    "user_id": rng.choice(pool, size=n),
                }
            )
        )

    table = pd.concat(frames, ignore_index=True).sort_values("connect_time", kind="stable")
    table.insert(0, "session_id", [f"syn-{k:06d}" for k in range(len(table))])
    for col in ("connect_time", "disconnect_time"):
        table[col] = table[col].dt.tz_convert("UTC")
    table = table.reset_index(drop=True)
    return _add_energy(table, np.random.default_rng([seed, 1]))


def _add_energy(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Energy, charging-done time, station and driver link, drawn after everything else so
    the stays above do not depend on them.

    Each car needs a log-normal amount of energy (median 8 kWh) and charges at a constant
    rate (3 to 6.6 kW) from plug-in until it has it or leaves. Because the rate is constant,
    spreading the recorded energy evenly from plug-in to charging-done, the estimate the
    analysis uses (check D1), gives the energy in any window exactly. The driver link is
    recorded at plug-in and the station is one of 54.
    """
    n = len(table)
    need = np.clip(np.exp(rng.normal(np.log(8.0), 0.6, n)), 0.5, 60.0)
    rate = rng.uniform(3.0, 6.6, n)
    stay_h = (table["disconnect_time"] - table["connect_time"]) / pd.Timedelta(hours=1)
    charge_h = np.minimum(need / rate, stay_h.to_numpy())
    table["kwh_delivered"] = rate * charge_h
    table["done_charging_time"] = table["connect_time"] + pd.to_timedelta(charge_h, unit="h")
    table["linked_time"] = table["connect_time"]
    table["station_id"] = [f"station-{k:02d}" for k in rng.integers(0, 54, n)]
    return table


def with_app_entries(
    table: pd.DataFrame, *, seed: int = 0, share: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A copy of the table in which a `share` of sessions are claimed in the app, and their
    entries (in the form of `data.app_entries`).

    A claimed session's first entry comes a few minutes to a few hours after plug-in and
    states the true stay, rounded to 15 minutes, with noise, or one hour; a third of claimed
    sessions edit the plan once, an hour later. Unclaimed sessions lose their driver.
    """
    rng = np.random.default_rng([seed, 3])
    out = table.copy()
    claimed = rng.random(len(out)) < share
    out["user_id"] = out["user_id"].where(claimed, None).astype(object)
    rows = out[claimed]
    n = len(rows)
    lag = pd.to_timedelta(np.where(rng.random(n) < 0.7, 0.01, rng.uniform(0.1, 3.0, n)), unit="h")
    stay = (rows["disconnect_time"] - rows["connect_time"]) / pd.Timedelta(minutes=1)
    typed = np.round(stay.to_numpy() * rng.lognormal(0.0, 0.2, n) / 15.0) * 15.0
    typed = np.where(rng.random(n) < 0.2, 60.0, np.maximum(typed, 15.0))
    first = pd.DataFrame(
        {
            "session_id": rows["session_id"].to_numpy(),
            "entry": 0,
            "modified_at": (rows["connect_time"] + lag).to_numpy(),
            "minutes_available": typed,
            "requested_departure": (
                rows["connect_time"] + pd.to_timedelta(typed, unit="m")
            ).to_numpy(),
            "kwh_requested": rows["kwh_delivered"].to_numpy() * rng.lognormal(0.0, 0.3, n),
        }
    )
    edit = first[rng.random(n) < 1 / 3].copy()
    edit["entry"] = 1
    edit["modified_at"] = edit["modified_at"] + pd.Timedelta(hours=1)
    edit["minutes_available"] = edit["minutes_available"] + 60.0
    edit["requested_departure"] = edit["requested_departure"] + pd.Timedelta(hours=1)
    entries = pd.concat([first, edit], ignore_index=True)
    out["linked_time"] = out["session_id"].map(first.set_index("session_id")["modified_at"])
    entries = entries.sort_values(["modified_at", "session_id", "entry"], kind="stable")
    return out, entries.reset_index(drop=True)


def _arrival_hours(rng: np.random.Generator, n: int) -> np.ndarray:
    """Hour of day: three in four arrive in a morning peak, the rest spread over the day."""
    morning = rng.random(n) < 0.75
    hours = np.where(morning, rng.normal(8.5, 1.2, n), rng.uniform(6.0, 20.0, n))
    return np.clip(hours, 0.0, 23.99)


def _driver_pools(regimes: list[Regime], rng: np.random.Generator) -> list[np.ndarray]:
    pools: list[np.ndarray] = []
    next_id = 0
    for k, regime in enumerate(regimes):
        carried = np.empty(0, dtype=object)
        if k > 0 and len(pools[-1]):
            n_carry = min(round(regime.returning_share * regime.drivers), len(pools[-1]))
            carried = rng.choice(pools[-1], size=n_carry, replace=False)
        n_new = max(regime.drivers - len(carried), 0)
        fresh = np.array([f"driver-{next_id + j:05d}" for j in range(n_new)], dtype=object)
        next_id += n_new
        pools.append(np.concatenate([carried, fresh]).astype(object))
    return pools


def demo_regimes() -> list[Regime]:
    """A made-up history: a stable year, a near-empty half year, then a new population that
    stays for less time. Dates are deliberately far from any real data set."""
    return [
        Regime("2001-01-01", sessions_per_weekday=30, median_stay_h=6.5, stay_spread=0.55),
        Regime(
            "2002-04-01",
            sessions_per_weekday=0.5,
            median_stay_h=3.0,
            drivers=20,
            returning_share=0.5,
        ),
        Regime(
            "2002-11-01",
            sessions_per_weekday=12,
            median_stay_h=2.5,
            stay_spread=0.7,
            drivers=150,
            returning_share=0.3,
        ),
    ]
