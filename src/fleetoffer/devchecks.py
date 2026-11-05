"""Checks D2 to D12 on development data (analysis plan, Section 10a), and the two reports the
plan asks for before the tag: the counts behind D3 and the week-to-week spread of offer sizes
on the model-selection months.

The checks choose settings, and nothing in the frozen development evaluation (September 2019
to February 2020) may choose one (analysis plan, Section 7). So every check runs on Caltech
from the first download month to August 2019, except D3, whose counts the plan sets on the
model-selection and frozen evaluation months together; for those months only the counts of
cars present at the decision time, and of those with energy remaining, are kept. D1 is in
`docs/data.md` (failed; its fallback applies). D11 is built into every refit
(`quantiles.training_rows`) and tested.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer import data, percar
from fleetoffer.baselines import Population, Stratified
from fleetoffer.daylevel import Calibrated, UnionBound
from fleetoffer.engine import (
    ENERGY_REMAINING_KWH,
    MORNING,
    Daily,
    Oracle,
    fleet_replay,
    training_base_set,
)
from fleetoffer.interface import DayView, FleetRule
from fleetoffer.offer import CAP_KW, BaseSet
from fleetoffer.quantiles import GBDT, PerCarRule
from fleetoffer.resample import paired_interval, week_of

LOCAL_TZ = "America/Los_Angeles"
SELECTION_START = percar.SPLIT_END  # 1 May 2019: model-selection months, offers made from here
DEVELOPMENT_END = pd.Timestamp("2020-03-01", tz=LOCAL_TZ)

D2_LINE_KW = 7.5
D3_MEDIAN, D3_P10, D3_MIN_CARS = 10, 4, 5
D4_SHARE = 0.30
D5_SHARE, D5_DAYS = 0.30, 200
D7_RATIO, D7_RANK = (0.8, 1.2), 0.6
D9_ALPHA, D9_DAY_SHARE, D9_SECONDS, D9_SLOW_SHARE = 0.2, 0.25, 60.0, 0.05
# D10 on the model-selection months (`fleetoffer percar`, d10_reading.csv): the grid levels at
# which the primary model's miss rate is within one point of the level (half a point at
# 0.005). Arm 3 uses a fitted bound only at these levels and between two of them.
D10_SUPPORTED = {"stay": (0.002, 0.005), "energy": (0.002, 0.005, 0.01)}


class Nothing(FleetRule):
    """The trivial bound for every car: used only to let the engine count cars."""

    name = "nothing"

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        n = len(day.present)
        return np.zeros(n), np.zeros(n)


# D3: cars with energy remaining at the decision time.


def development_sessions(data_root: Path) -> pd.DataFrame:
    """Every Caltech development session (to 29 February 2020); for D3's counts only."""
    items = data.load_items(Path(data_root), "caltech")
    sessions, _ = data.clean_sessions(data.parse_sessions(items))
    return sessions[sessions["connect_time"] < DEVELOPMENT_END].reset_index(drop=True)


def d3_counts(sessions: pd.DataFrame, daily: Daily = MORNING) -> pd.DataFrame:
    """Per weekday from May 2019 to February 2020: cars present and cars with more than
    0.5 kWh remaining. Nothing else is kept."""
    days = fleet_replay(sessions, Nothing(), method="bounds", daily=daily, until=DEVELOPMENT_END)
    days = days.days
    days = days[days["decision_time"] >= SELECTION_START]
    out = days[["decision_time", "cars"]].copy()
    out["cars_with_energy"] = days["cars_with_energy"].fillna(0).astype(int)
    local = out["decision_time"].dt.tz_convert(LOCAL_TZ)
    out["period"] = np.where(local < percar.SELECTION_END, "model selection", "frozen evaluation")
    return out.reset_index(drop=True)


def d3_reading(counts: pd.DataFrame) -> dict:
    c = counts["cars_with_energy"].to_numpy()
    median, p10 = float(np.median(c)), float(np.percentile(c, 10))
    per_period = {
        name: {
            "weekdays": len(g),
            "median": float(np.median(g["cars_with_energy"])),
            "p10": float(np.percentile(g["cars_with_energy"], 10)),
            "below_5": int((g["cars_with_energy"] < D3_MIN_CARS).sum()),
        }
        for name, g in counts.groupby("period", sort=False)
    }
    return {
        "weekdays": len(c),
        "median": median,
        "p10": p10,
        "below_5": int((c < D3_MIN_CARS).sum()),
        "passes": median >= D3_MEDIAN and p10 >= D3_P10,
        "by_period": per_period,
    }


# D4 to D7 on the per-car record (inputs at 10:00 and outcomes), through August 2019.


def d4(cardays: pd.DataFrame) -> dict:
    """Present cars not linked to a driver at 10:00 whose estimated energy after 10:00 is 0."""
    zero = cardays["target_energy_kwh"] <= 1e-9
    unclaimed = cardays["linked"] == 0
    share = float((zero & unclaimed).mean())
    return {
        "car_days": len(cardays),
        "unclaimed_zero_share": share,
        "unclaimed_share": float(unclaimed.mean()),
        "zero_share": float(zero.mean()),
        "rule_applies": share > D4_SHARE,
    }


def d5(cardays: pd.DataFrame) -> dict:
    """Weekdays on which at least 30% of present cars with energy remaining had a typed
    energy request recorded before 10:00."""
    has = cardays[cardays["target_energy_kwh"] > ENERGY_REMAINING_KWH]
    typed = has["typed_kwh"].notna() & (has["typed_kwh"] > 0)
    share = typed.groupby(has["decision_time"]).mean()
    days = int((share >= D5_SHARE).sum())
    return {
        "weekdays_with_cars": len(share),
        "weekdays_at_30_percent": days,
        "median_share": float(share.median()),
        "keep_sensitivity": days >= D5_DAYS,
    }


def d6(cardays: pd.DataFrame) -> pd.DataFrame:
    """Per month: share of present cars linked by 10:00, and linked with at least three
    earlier ended sessions."""
    local = cardays["decision_time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    month = local.dt.to_period("M").astype(str)
    linked = cardays["linked"] == 1
    history = linked & (cardays["n_past"].fillna(0) >= 3)
    return (
        pd.DataFrame({"month": month, "linked": linked, "history": history})
        .groupby("month")
        .agg(car_days=("linked", "size"), linked=("linked", "mean"), history=("history", "mean"))
        .reset_index()
    )


def d7(cardays: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """Sessions present at 10:00 with an energy request typed before 10:00 (their first such
    morning): the session's recorded energy against the latest request typed before 10:00."""
    typed = cardays[cardays["typed_kwh"].notna() & (cardays["typed_kwh"] > 0)]
    first = typed.sort_values("decision_time").drop_duplicates("session_id")
    joined = first.merge(sessions[["session_id", "kwh_delivered"]], on="session_id")
    ratio = joined["kwh_delivered"] / joined["typed_kwh"]
    rank = float(joined["kwh_delivered"].corr(joined["typed_kwh"], method="spearman"))
    median = float(ratio.median())
    return {
        "sessions": len(joined),
        "median_ratio": median,
        "rank_correlation": rank,
        "adequate_proxy": D7_RATIO[0] <= median <= D7_RATIO[1] and rank >= D7_RANK,
    }


# D2 and D12: the cap and the base set.


def d2_reading(sessions: pd.DataFrame) -> dict:
    """D2 needs the measured power of each session, which the time series were to give; they
    failed D1, so 6.7 kW is an assumption for every station. For context only: the average
    power from plug-in to charging-done, a lower bound on the session's peak power."""
    hours = (sessions["done_charging_time"] - sessions["connect_time"]) / pd.Timedelta(hours=1)
    ok = hours > 0.05
    power = sessions.loc[ok, "kwh_delivered"] / hours[ok]
    return {
        "rule": "D2 not measurable (D1 failed); 6.7 kW for all stations, an assumption",
        "sessions_with_times": int(ok.sum()),
        "average_power_above_cap_share": float((power > CAP_KW).mean()),
        "average_power_above_7_5_share": float((power > D2_LINE_KW).mean()),
        "average_power_p99_kw": float(power.quantile(0.99)),
    }


def d12(base: BaseSet) -> dict:
    """h0 of the base set, and whether U0 is non-empty with positive width wherever the
    training fleet can charge (the power limit is an average, so positive where any car's
    is)."""
    return {
        "h0_power_kw": [round(float(v), 6) for v in base.power],
        "h0_energy_kwh": round(float(base.energy), 6),
        "non_empty_with_width": bool(base.energy > 0 and (base.power > 0).any()),
        "steps_with_power": int((base.power > 0).sum()),
    }


# The arms on the model-selection months (offers from 1 May 2019, warm-up before).


def _per_car(model, level: float = 0.1) -> PerCarRule:
    return PerCarRule(
        [model],
        level=level,
        window_days=percar.CHOSEN["window_days"],
        min_rows=percar.FIXED["min_rows"],
    )


def _gbdt() -> GBDT:
    return GBDT(**percar.CHOSEN["gbdt"], **percar.FIXED["gbdt"])


def make_arm(name: str) -> FleetRule:
    """The arms of the analysis plan, Section 5; "s" marks the secondary calibration rule."""
    if name == "1":
        oracle = Oracle()
        oracle.name = "1 oracle"
        return oracle
    if name == "2":
        rule = _per_car(Population())
        rule.name = "2 population"
        return rule
    if name == "3":
        return UnionBound(_per_car(_gbdt()), D10_SUPPORTED, name="3 learned, union bound")
    models = {
        "4": (_gbdt, "learned"),
        "5": (Population, "population"),
        "6": (lambda: Stratified(**percar.FIXED["stratified"]), "stratified"),
    }
    key, secondary = name[0], name.endswith("s")
    make, label = models[key]
    rule = "split" if secondary else "aci"
    return Calibrated(_per_car(make()), rule=rule, name=f"{name} {label}, {rule}")


ARMS = ("1", "2", "3", "4", "5", "6", "4s", "5s", "6s")


def run_arm(data_root: str, name: str) -> dict:
    """One arm over development data to August 2019, offers on the model-selection months on
    the base set fixed from training (through April 2019) and on the day-average base set."""
    sessions, entries = percar.load(Path(data_root))
    base = training_base_set(sessions, percar.SPLIT_END)
    rule = make_arm(name)
    result = fleet_replay(
        sessions,
        rule,
        base=base,
        entries=entries,
        until=percar.SELECTION_END,
        offers_from=SELECTION_START,
        day_average_too=True,
    )
    out = {"name": name, "days": result.days}
    if hasattr(rule, "log"):
        out["log"] = pd.DataFrame(rule.log)
    if name == "4":
        frame = percar.scored(rule.per_car, result.cars)
        out["d10"] = percar.d10(percar.score_table(frame, ["gbdt"]))
    return out


def run_arms(data_root: Path, workers: int = 9) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_arm, [str(data_root)] * len(ARMS), ARMS))


# Reports from the arms (fleet layer).


def selection_days(days: pd.DataFrame) -> pd.DataFrame:
    """Scored weekdays of the model-selection months on which an offer was made."""
    keep = (days["decision_time"] >= SELECTION_START) & days["size_kwh"].notna()
    keep &= days["cars_with_energy"] >= D3_MIN_CARS
    return days[keep].reset_index(drop=True)


def d9(oracle_days: pd.DataFrame, every_arm: pd.DataFrame) -> dict:
    days = selection_days(oracle_days)
    low = float((days["alpha"] < D9_ALPHA).mean())
    offered = every_arm[every_arm["size_kwh"].notna()]
    slow = float((offered["solve_seconds"] > D9_SECONDS).mean())
    return {
        "scored_weekdays": len(days),
        "oracle_alpha_below_0_2_share": low,
        "solves": len(offered),
        "solves_over_60_s_share": slow,
        "max_solve_seconds": float(offered["solve_seconds"].max()),
        "fallback": low > D9_DAY_SHARE or slow > D9_SLOW_SHARE,
    }


def feedback_delay(log: pd.DataFrame, since: pd.Timestamp | None = None) -> dict:
    """How late a day-level rule learned its outcomes (`docs/theory.md`, late outcomes): of
    the scored days decided from `since`, the share whose outcome was known before the next
    decision, and the most scored days still pending at any decision from `since`."""
    log = log[log["fitted"]].sort_values("decision_time").reset_index(drop=True)
    decisions = log["decision_time"]
    following = decisions.shift(-1)
    scored = log["scored"].fillna(False).astype(bool)
    keep = scored & following.notna()
    if since is not None:
        keep &= decisions >= since
    known = (log.loc[keep, "revealed_at"] < following[keep]).astype(float)
    decided = log.loc[scored, "decision_time"].to_numpy()
    revealed = log.loc[scored, "revealed_at"].to_numpy()
    at = decisions if since is None else decisions[decisions >= since]
    pending = [int(((decided < now) & (revealed > now)).sum()) for now in at]
    return {
        "scored_days": int(keep.sum()),
        "known_before_next": float(known.mean()) if len(known) else float("nan"),
        "most_pending": max(pending, default=0),
    }


def alpha_zero(days: pd.DataFrame) -> pd.DataFrame:
    """Per arm, on scored weekdays of the model-selection months: share of days with alpha = 0
    on the fixed base set and on the day-average base set, and mean sizes."""
    rows = []
    for rule, g in days.groupby("rule", sort=False):
        g = selection_days(g)
        rows.append(
            {
                "rule": rule,
                "days": len(g),
                "alpha_zero_fixed": float((g["alpha"] <= 1e-9).mean()),
                "alpha_zero_day_average": float((g["alpha_day_average"] <= 1e-9).mean()),
                "size_fixed_kwh": float(g["size_kwh"].mean()),
                "size_day_average_kwh": float(g["size_day_average_kwh"].mean()),
                "fixed_delivered": float(g["fixed_delivered"].astype(float).mean()),
                "hindsight_delivered": float(g["hindsight_delivered"].astype(float).mean()),
                "all_bounds_held": float(g["all_bounds_held"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def spread(learned: pd.DataFrame, population: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Week-to-week spread of offer sizes, arm 4 against arm 5, on the model-selection months:
    weekly mean sizes, their paired differences, and what that spread implies for H1 over
    the 26 weeks of the frozen evaluation."""
    a = selection_days(learned)[["decision_time", "size_kwh"]]
    b = selection_days(population)[["decision_time", "size_kwh"]]
    both = a.merge(b, on="decision_time", suffixes=("_learned", "_population"))
    both["week"] = week_of(both["decision_time"])
    weekly = (
        both.groupby("week")
        .agg(
            days=("decision_time", "size"),
            start=("decision_time", "min"),
            learned_kwh=("size_kwh_learned", "mean"),
            population_kwh=("size_kwh_population", "mean"),
        )
        .reset_index(drop=True)
    )
    weekly["difference_kwh"] = weekly["learned_kwh"] - weekly["population_kwh"]
    sd = float(weekly["difference_kwh"].std(ddof=1))
    mean_population = float(both["size_kwh_population"].mean())
    interval = paired_interval(both, "size_kwh_learned", "size_kwh_population", relative=True)
    summary = {
        "days": len(both),
        "weeks": len(weekly),
        "learned_mean_kwh": float(both["size_kwh_learned"].mean()),
        "population_mean_kwh": mean_population,
        "weekly_difference_sd_kwh": sd,
        "ten_percent_of_population_kwh": 0.1 * mean_population,
        "projected_se_26_weeks_kwh": sd / np.sqrt(26),
        "relative_difference": interval,
    }
    return weekly, summary
