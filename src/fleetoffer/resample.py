"""Uncertainty by resampling whole calendar weeks (analysis plan, Section 9).

Cars on the same day share its events, and days in the same week share its pattern, so
neither cars nor days are treated as independent: whole weeks (Monday to Sunday, local
dates) are drawn with replacement, and both arms of a paired comparison are drawn together.
Two-week blocks are the pre-registered sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRAWS = 10_000
SEED = 0
LOCAL_TZ = "America/Los_Angeles"


def week_of(decision_time: pd.Series, weeks: int = 1, tz: str = LOCAL_TZ) -> np.ndarray:
    """A block label per decision: the local Monday of its week, grouped `weeks` at a time
    counting from the first Monday in the data."""
    local = pd.to_datetime(decision_time).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    monday = local - pd.to_timedelta(local.dt.day_of_week, unit="D")
    index = ((monday - monday.min()).dt.days // 7).to_numpy()
    return index // weeks


def paired_interval(
    days: pd.DataFrame,
    a: str,
    b: str,
    *,
    weeks: int = 1,
    draws: int = DRAWS,
    seed: int = SEED,
    level: float = 0.95,
    relative: bool = False,
) -> dict:
    """The mean of a minus b over the days (or (mean a - mean b) / mean b if `relative`),
    with a percentile interval from resampling whole blocks of weeks, both columns together."""
    days = days[[a, b, "decision_time"]].dropna().reset_index(drop=True)
    if days.empty:
        return {"days": 0, "blocks": 0, "estimate": np.nan, "low": np.nan, "high": np.nan}
    block = week_of(days["decision_time"], weeks)
    labels, index = np.unique(block, return_inverse=True)
    sums_a = np.bincount(index, days[a].to_numpy(float))
    sums_b = np.bincount(index, days[b].to_numpy(float))
    counts = np.bincount(index).astype(float)

    def statistic(weights: np.ndarray) -> np.ndarray:
        total_a, total_b, n = weights @ sums_a, weights @ sums_b, weights @ counts
        if relative:
            return (total_a - total_b) / total_b
        return (total_a - total_b) / n

    estimate = float(statistic(np.ones((1, len(labels))))[0])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(draws, len(labels)))
    weights = np.zeros((draws, len(labels)))
    np.add.at(weights, (np.repeat(np.arange(draws), len(labels)), picks.ravel()), 1.0)
    values = statistic(weights)
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return {
        "days": len(days),
        "blocks": len(labels),
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
    }
