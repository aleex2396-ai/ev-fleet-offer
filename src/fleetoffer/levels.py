"""The level sweep (`daylevel.LevelSweep`), a descriptive diagnostic chosen after the first
fleet run and not a registered arm.

For each scored weekday, each per-car model (arm 4 learned, 5 population, 6 stratified), each
offer form (programme (24) on the fixed base set, on the day-average base set, and the
one-dimensional fallback) and each level of `daylevel.DELIVERY_LEVELS`, the offer made from
that morning's bounds at that level, with its size, its delivery under the fixed allocation
and with a split chosen with hindsight, whether every bound held, and how many cars had a zero
energy ceiling or a zero stay; the oracle (arm 1) beside them. From that one table: the
frontier of offer size against delivery for each model and form, the same with hindsight
delivery (the largest offer of the form that a per-request split chosen with hindsight could
deliver: an upper bound, not a method that could be deployed, and not the registered service,
which fixes one split for every request), and the share of cars given a zero energy ceiling at
each level. The run here covers May to August 2019 only.

Beside it, the one-fact diagnostic (models marked "k"): the same sweep when the cars that
truly take nothing after 10:00 are known and dropped, and the models are refitted on car-days
that took energy (`LevelSweep`'s `drop_empty`). A perfect-knowledge yardstick for what live
charging status at 10:00 could recover, not a method that could be deployed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer import options, percar
from fleetoffer.baselines import Population
from fleetoffer.daylevel import LevelSweep
from fleetoffer.devchecks import D3_MIN_CARS, SELECTION_START, make_arm
from fleetoffer.engine import fleet_replay, training_base_set

MODELS = options.MODELS  # "4" learned, "5" population, "6" stratified
FORMS = options.FORMS
RUNS = (
    [(model, form) for model in MODELS for form in FORMS]
    + [("1", form) for form in FORMS]
    + [(f"{model}k", form) for model in MODELS for form in FORMS]
)
MATCHED_DELIVERY = (0.90, 0.95)  # delivery rates at which models are compared


def sweep_rule(model: str, form: str, base, offers_from=SELECTION_START) -> LevelSweep:
    """The sweep of one model ("4", "5", "6", or with "k" the diagnostic) and one form."""
    known = model.endswith("k")
    per_car = make_arm(model[0]).per_car
    if known:
        per_car.energy_left_only = True
        for m in per_car.models:
            if isinstance(m, Population):
                m.energy_left_only = True
    label = MODELS[model[0]] + (", empty cars known" if known else "")
    return LevelSweep(
        per_car,
        {form: options.make_offer(form, base)},
        offers_from=offers_from,
        drop_empty=known,
        name=f"{model} {label}, level sweep | {form}",
    )


def sweep_rows(rule: LevelSweep, model: str) -> pd.DataFrame:
    label = MODELS[model[0]] + (", empty cars known" if model.endswith("k") else "")
    return pd.DataFrame(rule.log).assign(model=model, model_name=label)


def run_one(data_root: str, model: str, form: str) -> dict:
    """One model and one form over development data to August 2019, offers from May 2019."""
    if model == "1":
        oracle = options.run_one(data_root, "1", form)["days"]
        return {"model": model, "form": form, "oracle": oracle}
    sessions, entries = percar.load(Path(data_root))
    base = training_base_set(sessions, percar.SPLIT_END)
    rule = sweep_rule(model, form, base)
    fleet_replay(
        sessions,
        rule,
        entries=entries,
        until=percar.SELECTION_END,
        method="bounds",
        reveal_empty=rule.drop_empty,
    )
    return {"model": model, "form": form, "sweep": sweep_rows(rule, model)}


def run_all(data_root: Path, workers: int = 10, store: Path | None = None) -> list[dict]:
    jobs = [(str(data_root), model, form) for model, form in RUNS]
    return options.run_saved(run_one, jobs, workers, store)


def scored(rows: pd.DataFrame, start: pd.Timestamp | None = SELECTION_START) -> pd.DataFrame:
    """The scored weekdays from `start`: at least five cars with energy remaining (check D3)."""
    keep = rows["cars_with_energy"] >= D3_MIN_CARS
    if start is not None:
        keep &= rows["decision_time"] >= start
    return rows[keep].reset_index(drop=True)


def table(sweep: pd.DataFrame, start: pd.Timestamp | None = SELECTION_START) -> pd.DataFrame:
    """Per model, form and level, over the scored days."""
    rows = scored(sweep, start)
    for col in ("fixed_delivered", "hindsight_delivered", "all_bounds_held"):
        rows[col] = rows[col].astype(float)
    grouped = rows.groupby(["model", "model_name", "form", "level"], sort=True)
    out = grouped.agg(
        days=("decision_time", "size"),
        size_mean_kwh=("size_kwh", "mean"),
        size_median_kwh=("size_kwh", "median"),
        fixed_delivered=("fixed_delivered", "mean"),
        hindsight_delivered=("hindsight_delivered", "mean"),
        all_bounds_held=("all_bounds_held", "mean"),
        exact_learned_kwh=("exact_learned_kwh", "mean"),
        exact_true_kwh=("exact_true_kwh", "mean"),
        solve_seconds_max=("solve_seconds", "max"),
    )
    sums = grouped[
        [
            "cars",
            "cars_with_energy",
            "zero_ceiling_cars",
            "zero_ceiling_cars_with_energy",
            "zero_stay_cars",
        ]
    ].sum()
    out["zero_offer_share"] = grouped["size_kwh"].apply(lambda s: float((s <= 1e-9).mean()))
    out["zero_ceiling_share"] = sums["zero_ceiling_cars"] / sums["cars"]
    out["zero_ceiling_share_with_energy"] = (
        sums["zero_ceiling_cars_with_energy"] / sums["cars_with_energy"]
    )
    out["zero_stay_share"] = sums["zero_stay_cars"] / sums["cars"]
    return out.reset_index()


def oracle_table(
    oracle: pd.DataFrame, start: pd.Timestamp | None = SELECTION_START
) -> pd.DataFrame:
    """Arm 1 per form, over the same scored days."""
    rows = scored(oracle[oracle["size_kwh"].notna()], start)
    return (
        rows.assign(
            fixed_delivered=rows["fixed_delivered"].astype(float),
            hindsight_delivered=rows["hindsight_delivered"].astype(float),
        )
        .groupby("form")
        .agg(
            days=("decision_time", "size"),
            size_mean_kwh=("size_kwh", "mean"),
            fixed_delivered=("fixed_delivered", "mean"),
            hindsight_delivered=("hindsight_delivered", "mean"),
            exact_true_kwh=("exact_true_kwh", "mean"),
        )
        .reset_index()
    )


def matched(levels: pd.DataFrame, rates=MATCHED_DELIVERY) -> pd.DataFrame:
    """For each model, form, delivery measure and rate: the largest mean offer among the swept
    levels whose delivery rate is at least that rate, with the level and the rate achieved.
    No interpolation between levels."""
    rows = []
    for (model, name, form), curve in levels.groupby(["model", "model_name", "form"]):
        for measure in ("fixed_delivered", "hindsight_delivered"):
            for rate in rates:
                ok = curve[curve[measure] >= rate - 1e-12]
                best = ok.loc[ok["size_mean_kwh"].idxmax()] if len(ok) else None
                rows.append(
                    {
                        "model": model,
                        "model_name": name,
                        "form": form,
                        "measure": measure,
                        "rate": rate,
                        "size_mean_kwh": float(best["size_mean_kwh"]) if best is not None else 0.0,
                        "level": float(best["level"]) if best is not None else np.nan,
                        "achieved": float(best[measure]) if best is not None else np.nan,
                    }
                )
    return pd.DataFrame(rows)


ACCOUNTING_RATE = 0.90
ACCOUNTING_STEPS = (
    ("exact band of the true sets", "a ceiling: each checkpoint maximised apart"),
    ("perfect knowledge", "the oracle's offer: the inner approximation's own loss"),
    (
        "forecasts at 90% delivery",
        "best swept level delivered on 90% of days; level chosen with hindsight",
    ),
    ("empty cars known", "the same, with the cars that take nothing known; a yardstick"),
    (
        "hindsight split at 90%",
        "the same, delivered by a split chosen with hindsight; an upper bound",
    ),
    ("day-level rule as run", "the level chosen online from past days"),
)


def accounting(
    table: pd.DataFrame,
    oracle: pd.DataFrame,
    online: pd.DataFrame,
    rate: float = ACCOUNTING_RATE,
) -> pd.DataFrame:
    """Where the perfect-knowledge offer goes, per offer form: the exact band of the true sets
    (programme forms only: the band of the exact aggregate set, which an offer of the form
    p + alpha U0 need not reach), the oracle's offer, the
    largest swept offer of each model delivered on `rate` of days under the fixed allocation,
    the same with the empty cars known and with a split chosen with hindsight, and each
    day-level rule as run. Every row is compared at its own delivery rate, shown beside it.

    `table` is `table`'s output; `oracle` has one row per form (`form`, `size_mean_kwh`,
    `fixed_delivered`, `exact_true_kwh`); `online` one row per day-level arm (`arm`, `form`,
    `size_mean_kwh`, `fixed_delivered`)."""
    best = matched(table, rates=(rate,))
    notes = dict(ACCOUNTING_STEPS)
    rows = []
    for form in sorted(set(table["form"]) | set(oracle["form"]) | set(online["form"])):
        mine = oracle[oracle["form"] == form]
        perfect = float(mine["size_mean_kwh"].iloc[0]) if len(mine) else np.nan

        def add(
            step,
            who,
            size,
            delivered,
            level=np.nan,
            measure="fixed_delivered",
            form=form,
            perfect=perfect,
        ):
            share = size / perfect if perfect > 1e-9 else np.nan
            rows.append(
                {
                    "form": form,
                    "step": step,
                    "who": who,
                    "size_mean_kwh": float(size),
                    "share_of_perfect": share,
                    "delivered": float(delivered),
                    "measure": measure,
                    "level": float(level),
                    "note": notes[step],
                }
            )

        if len(mine) and form != "segment":
            add("exact band of the true sets", "1", mine["exact_true_kwh"].iloc[0], np.nan)
        if len(mine):
            add("perfect knowledge", "1", perfect, mine["fixed_delivered"].iloc[0])
        for step, models, measure in (
            ("forecasts at 90% delivery", MODELS, "fixed_delivered"),
            ("empty cars known", [f"{m}k" for m in MODELS], "fixed_delivered"),
            ("hindsight split at 90%", MODELS, "hindsight_delivered"),
        ):
            for model in models:
                row = best[
                    (best["form"] == form) & (best["model"] == model) & (best["measure"] == measure)
                ]
                if len(row):
                    r = row.iloc[0]
                    add(step, model, r["size_mean_kwh"], r["achieved"], r["level"], measure)
        for _, r in online[online["form"] == form].iterrows():
            add("day-level rule as run", r["arm"], r["size_mean_kwh"], r["fixed_delivered"])
    return pd.DataFrame(rows)


SWEPT_STEPS = ("forecasts at 90% delivery", "empty cars known", "hindsight split at 90%")


def same_days(
    accounts: pd.DataFrame,
    sweep: pd.DataFrame,
    days: pd.DataFrame,
    start: pd.Timestamp | None = SELECTION_START,
) -> pd.DataFrame:
    """`accounting`'s rows with each share of the perfect-knowledge offer taken over the days
    that row was measured on. A swept model makes no offer on a day without a fit, so a row can
    cover fewer scored days than the perfect-knowledge offer, and dividing by the perfect-
    knowledge mean over every scored day would compare different days. `sweep` holds the
    sweep's day rows, `days` the day rows of the oracle (arms 1 and 1a) and of the day-level
    arms. Adds `days` (scored weekdays behind the row) and `perfect_same_days_kwh`; the exact
    band and the perfect-knowledge rows keep every scored day."""
    swept = scored(sweep[sweep["size_kwh"].notna()], start)
    rows = scored(days[days["size_kwh"].notna()], start)
    oracle = rows[rows["arm"].astype(str).isin(["1", "1a"])]
    out = accounts.copy()
    out["days"] = np.nan
    out["perfect_same_days_kwh"] = np.nan
    for i, r in out.iterrows():
        if r["step"] in SWEPT_STEPS:
            mine = swept[
                (swept["model"].astype(str) == str(r["who"]))
                & (swept["form"] == r["form"])
                & np.isclose(swept["level"], r["level"])
            ]
        elif r["step"] == "day-level rule as run":
            mine = rows[(rows["arm"].astype(str) == str(r["who"])) & (rows["form"] == r["form"])]
        else:
            out.at[i, "days"] = int((oracle["form"] == r["form"]).sum())
            continue
        times = set(mine["decision_time"])
        perfect = oracle[(oracle["form"] == r["form"]) & oracle["decision_time"].isin(times)]
        mean = float(perfect["size_kwh"].mean()) if len(perfect) else np.nan
        out.at[i, "days"] = len(times)
        out.at[i, "perfect_same_days_kwh"] = mean
        out.at[i, "share_of_perfect"] = r["size_mean_kwh"] / mean if mean > 1e-9 else np.nan
    return out
