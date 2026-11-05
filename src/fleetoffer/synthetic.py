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
    return table.reset_index(drop=True)


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
