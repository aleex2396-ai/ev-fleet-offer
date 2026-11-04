"""Scores for a set of decisions: how often promises were kept, and how large they were.

The two are always reported together, because a rule can always be more reliable by
promising less.

- ``kept_share``: share of the promises made that were kept.
- ``mean_promised_h``: average promise per decision, counting a decision with no promise
  as 0 hours (the aggregator could not offer that car).
- ``offered_share``: share of decisions where a promise was made.
"""

from __future__ import annotations

import pandas as pd


def _summarise(decisions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = decisions.assign(
        _kept=decisions["kept"].astype("Float64"),
        _promised=decisions["promised_h"].fillna(0.0),
        _offered=decisions["offered"].astype(int),
    )
    if not keys:
        d, keys = d.assign(_all="all"), ["_all"]
    out = d.groupby(keys, sort=True).agg(
        decisions=("_offered", "size"),
        offered=("_offered", "sum"),
        kept=("_kept", "sum"),
        mean_promised_h=("_promised", "mean"),
    )
    out["kept_share"] = (out["kept"] / out["offered"].where(out["offered"] > 0)).astype(float)
    out["offered_share"] = out["offered"] / out["decisions"]
    out["kept"] = out["kept"].astype(int)
    return out.reset_index().drop(columns="_all", errors="ignore")


def monthly_summary(decisions: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """One row per rule and calendar month (in time zone `tz`) of the decision time."""
    local = decisions["decision_time"].dt.tz_convert(tz).dt.tz_localize(None)
    d = decisions.assign(month=local.dt.to_period("M"))
    keys = ["rule", "month"] if "rule" in d.columns else ["month"]
    return _summarise(d, keys)


def overall_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per rule."""
    return _summarise(decisions, ["rule"] if "rule" in decisions.columns else [])
