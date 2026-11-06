"""The per-car layer on development data: settings, monthly refits, scores and Figure 1.

The order follows the analysis plan (Sections 4, 7 and 8):

1. One pass of the engine over the development days collects, for every car present at a
   weekday 10:00, its inputs and (once revealed) its outcome, and the sessions known at each
   monthly origin.
2. Settings are chosen on the internal split of the training period: every model is fitted
   once at 1 January 2019 on what was known then and scored on the car-days of January to
   April 2019. The grid is `GRID`; the score is `split_score`.
3. With the chosen settings, every model is refitted at each monthly origin and scored on
   later car-days, through the model-selection months May to August 2019.

Nothing from 1 September 2019 is used: the stored months (by UTC plug-in time) from September
2019 are not opened, and the sessions and app entries are cut at 1 September 2019 Pacific
time before the engine starts (`SELECTION_END`). The frozen development evaluation
(September 2019 to February 2020) is scored only after the analysis plan's tag.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fleetoffer import data
from fleetoffer.baselines import DriverPast, Mixture, Population, Stratified, TypedPlan
from fleetoffer.daylevel import at_levels, conformal_level, held_level
from fleetoffer.engine import fleet_replay
from fleetoffer.features import LOCAL_TZ
from fleetoffer.offer import STEP_H
from fleetoffer.quantiles import (
    GBDT,
    LEVELS,
    TARGETS,
    LinearQuantiles,
    PerCarRule,
    QuantileNet,
    at_level,
    ended_rows,
    pinball,
    training_rows,
)

SPLIT_ORIGIN = pd.Timestamp("2019-01-01", tz=LOCAL_TZ)
SPLIT_END = pd.Timestamp("2019-05-01", tz=LOCAL_TZ)
SELECTION_END = pd.Timestamp("2019-09-01", tz=LOCAL_TZ)
PERIODS = (
    ("before the split", None, SPLIT_ORIGIN),
    ("internal split, Jan-Apr 2019", SPLIT_ORIGIN, SPLIT_END),
    ("model selection, May-Aug 2019", SPLIT_END, SELECTION_END),
)
FIGURE_LEVEL = 0.1
D10_TOLERANCE = {0.005: 0.005}  # D10: within one point of the level, half a point at 0.005
GRID = {
    "window_days": [None, 90, 180, 365],
    "gbdt": [
        {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": leaf}
        for lr in (0.05, 0.1)
        for leaves in (7, 15, 31)
        for leaf in (20, 100)
    ],
    "net": [{"hidden": h, "weight_decay": wd} for h in (16, 64) for wd in (1e-4, 1e-2)],
    "mixture_components": list(range(2, 9)),
}
# The settings the search chose on the internal split (`fleetoffer percar`, settings_search.csv),
# committed so that later runs load them rather than search again.
CHOSEN = {
    "window_days": 90,
    "gbdt": {"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 20},
    "net": {"hidden": 16, "weight_decay": 0.0001},
    "mixture_components": 7,
}
FIXED = {
    "gbdt": {"max_iter": 200, "seed": 0},
    "net": {"learning_rate": 3e-3, "epochs": 300, "patience": 20, "seed": 0},
    "mixture": {"n_init": 3, "seed": 0, "min_driver_sessions": 5},
    "stratified": {"min_cell": 30},
    "driver": {"min_sessions": 3},
    "min_rows": 200,
}


def load(
    data_root,
    until: pd.Timestamp | None = SELECTION_END,
    *,
    site: str = "caltech",
    unseal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A site's sessions plugged in before `until` (all, if None), with driver links, and the
    app entries recorded before `until`. Only the monthly files before the UTC month in which
    `until` falls are opened, so no file of a later period is read; at the default this
    leaves out the one session plugged in on the evening of 31 August 2019, Pacific time,
    which is stored with September's sessions. Caltech from 1 March 2020 and JPL need
    `unseal`, and every period after August 2019 needs the analysis plan's tag."""
    if (until is None or until > SELECTION_END) and not data.plan_is_tagged():
        raise data.SealedPeriodError(
            "months from September 2019 are the frozen development evaluation; they are "
            f"scored only once the '{data.PLAN_TAG}' tag exists"
        )
    first_unread = None
    if until is not None:
        utc = until.tz_convert("UTC")
        first_unread = pd.Timestamp(year=utc.year, month=utc.month, day=1, tz="UTC")
    items = data.load_items(data_root, site, before=first_unread, unseal=unseal)
    sessions, _ = data.clean_sessions(data.parse_sessions(items))
    inputs = data.parse_user_inputs(items)
    sessions = data.driver_links(sessions, inputs)
    if until is not None:
        sessions = sessions[sessions["connect_time"] < until].reset_index(drop=True)
    entries = data.app_entries(inputs)
    keep = entries["session_id"].isin(set(sessions["session_id"]))
    if until is not None:
        keep &= entries["modified_at"] < until
    return sessions, entries[keep].reset_index(drop=True)


def record(sessions, entries, until=SELECTION_END) -> PerCarRule:
    """Step 1: one pass that keeps every car-day's inputs and outcome."""
    rule = PerCarRule([TypedPlan()], name="record")
    fleet_replay(sessions, rule, method="bounds", entries=entries, until=until)
    return rule


def _scores(model, train, ended, test, origin) -> dict[str, float]:
    start = time.perf_counter()
    model.fit(train, ended, origin)
    stay, energy = model.predict(None, test)
    out = {"seconds": time.perf_counter() - start}
    for target, pred, y in (
        ("stay", stay, test["target_stay_h"].to_numpy(float)),
        ("energy", energy, test["target_energy_kwh"].to_numpy(float)),
    ):
        ok = ~np.isnan(pred).any(axis=1)
        out[f"{target}_pinball"] = float(pinball(y[ok], pred[ok]).mean()) if ok.any() else np.nan
        out[f"{target}_rows"] = int(ok.sum())
    return out


def split_score(row: dict, reference: dict) -> float:
    """The selection score: each target's mean pinball loss over levels and car-days,
    divided by that of the population rule with all history on the same rows, summed."""
    return row["stay_pinball"] / reference["stay_pinball"] + (
        row["energy_pinball"] / reference["energy_pinball"]
    )


def search(rule: PerCarRule) -> tuple[pd.DataFrame, dict]:
    """Step 2: the settings grid on the internal split. Returns every score and the choice.

    The history window is chosen on the population rule, whose only setting it is, and then
    shared by every model; the tree and network settings are chosen with that window; the
    mixture's number of components by the held-out log-likelihood of the split's sessions,
    as Lee, Li and Low did. Ties go to the earlier grid entry.
    """
    origin = SPLIT_ORIGIN.tz_convert("UTC")
    cardays = rule.cardays()
    test = cardays[
        (cardays["decision_time"] >= origin) & (cardays["decision_time"] < SPLIT_END)
    ].reset_index(drop=True)
    snapshot = rule.ended_at_origin[origin]
    later = rule.ended_at_origin[SPLIT_END.tz_convert("UTC")]
    held_out = later[later["connect_time"] >= origin]
    rows = []

    def run(family, settings, model, window):
        train = training_rows(cardays, origin, window)
        ended = ended_rows(snapshot, origin, window)
        rows.append(
            {
                "family": family,
                "settings": settings,
                "window_days": window,
                "train_rows": len(train),
            }
            | _scores(model, train, ended, test, origin)
        )
        return rows[-1]

    for window in GRID["window_days"]:
        run("population", {}, Population(), window)
    reference = rows[0]
    for row in rows:
        row["score"] = split_score(row, reference)
    window = min((r for r in rows), key=lambda r: r["score"])["window_days"]
    for settings in GRID["gbdt"]:
        run("gbdt", settings, GBDT(**settings, **FIXED["gbdt"]), window)
    for settings in GRID["net"]:
        run("net", settings, QuantileNet(**settings, **FIXED["net"]), window)
    for row in rows:
        row["score"] = split_score(row, reference)
    best = {
        family: min((r for r in rows if r["family"] == family), key=lambda r: r["score"])
        for family in ("gbdt", "net")
    }
    for k in GRID["mixture_components"]:
        mixture = Mixture(
            components=k, seed=FIXED["mixture"]["seed"], n_init=FIXED["mixture"]["n_init"]
        )
        start = time.perf_counter()
        mixture.fit(None, ended_rows(snapshot, origin, window), origin)
        rows.append(
            {
                "family": "mixture",
                "settings": {"components": k},
                "window_days": window,
                "held_out_loglik": mixture.held_out_loglik(held_out),
                "seconds": time.perf_counter() - start,
            }
        )
    components = max(
        (r for r in rows if r["family"] == "mixture"), key=lambda r: r["held_out_loglik"]
    )["settings"]["components"]
    chosen = {
        "window_days": window,
        "gbdt": best["gbdt"]["settings"],
        "net": best["net"]["settings"],
        "mixture_components": components,
    }
    table = pd.DataFrame(rows)
    table["settings"] = table["settings"].map(lambda s: ", ".join(f"{k}={v}" for k, v in s.items()))
    return table, chosen


def models(chosen: dict) -> list:
    """Every per-car model with the chosen settings; the first is the primary family."""
    mix = FIXED["mixture"]
    return [
        GBDT(**chosen["gbdt"], **FIXED["gbdt"]),
        QuantileNet(**chosen["net"], **FIXED["net"]),
        LinearQuantiles(),
        Population(),
        Stratified(**FIXED["stratified"]),
        DriverPast(**FIXED["driver"]),
        Mixture(components=chosen["mixture_components"], seed=mix["seed"], n_init=mix["n_init"]),
        Mixture(
            components=chosen["mixture_components"],
            per_driver=True,
            min_driver_sessions=mix["min_driver_sessions"],
            seed=mix["seed"],
            n_init=mix["n_init"],
            name="mixture-driver",
        ),
        TypedPlan(),
    ]


def rolling(sessions, entries, chosen: dict, until=SELECTION_END):
    """Step 3: monthly refits of every model with the chosen settings."""
    rule = PerCarRule(models(chosen), window_days=chosen["window_days"], min_rows=FIXED["min_rows"])
    result = fleet_replay(sessions, rule, method="bounds", entries=entries, until=until)
    return rule, result


def scored(rule: PerCarRule, cars: pd.DataFrame) -> pd.DataFrame:
    """Each recorded prediction beside the truth (stay on the engine's 15-minute grid)."""
    truth = cars[["decision_time", "session_id", "true_deadline", "true_energy_kwh"]]
    out = rule.predictions().merge(truth, on=["decision_time", "session_id"], validate="1:1")
    out["true_stay_h"] = out["true_deadline"] * STEP_H
    local = out["decision_time"].dt.tz_convert(LOCAL_TZ)
    out["month"] = local.dt.tz_localize(None).dt.to_period("M")
    out["period"] = None
    for name, first, last in PERIODS:
        keep = local < last if first is None else (local >= first) & (local < last)
        out.loc[keep, "period"] = name
    out["driver"] = np.where(out["linked"] == 1, "known", "anonymous")
    return out


def _columns(name: str, target: str) -> list[str]:
    return [f"{name}:{target}:{q:g}" for q in LEVELS]


def misses(frame: pd.DataFrame, name: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows with a prediction, and for each level whether the truth fell below the bound as
    the engine would use it (stay rounded down to the 15-minute grid)."""
    pred = frame[_columns(name, target)].to_numpy(float)
    ok = ~np.isnan(pred).any(axis=1)
    if target == "stay":
        steps = np.floor(np.minimum(pred[ok], 10.0) / STEP_H + 1e-9)
        missed = steps > frame["true_deadline"].to_numpy()[ok][:, None]
    else:
        missed = pred[ok] > frame["true_energy_kwh"].to_numpy()[ok][:, None] + 1e-9
    return ok, missed


def score_table(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Mean pinball loss and miss rate per level, by period, driver group and model."""
    rows = []
    groups = [("all", frame)] + [(g, frame[frame["driver"] == g]) for g in ("known", "anonymous")]
    for period in frame["period"].dropna().unique():
        for group, part in groups:
            part = part[part["period"] == period]
            for name in names:
                for target in TARGETS:
                    if not len(part):
                        continue
                    ok, missed = misses(part, name, target)
                    if not ok.any():
                        continue
                    y = part["true_stay_h" if target == "stay" else "true_energy_kwh"]
                    loss = pinball(
                        y.to_numpy(float)[ok], part[_columns(name, target)].to_numpy()[ok]
                    )
                    for k, q in enumerate(LEVELS):
                        rows.append(
                            {
                                "period": period,
                                "drivers": group,
                                "model": name,
                                "target": target,
                                "level": q,
                                "car_days": int(ok.sum()),
                                "of_car_days": len(part),
                                "pinball": float(loss[:, k].mean()),
                                "miss_rate": float(missed[:, k].mean()),
                            }
                        )
    return pd.DataFrame(rows)


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean pinball loss over levels, per period, group, model and target."""
    keys = ["period", "drivers", "model", "target"]
    return (
        scores.groupby(keys, sort=False)
        .agg(
            car_days=("car_days", "first"),
            of_car_days=("of_car_days", "first"),
            pinball=("pinball", "mean"),
        )
        .reset_index()
    )


def d10(scores: pd.DataFrame, model: str = "gbdt") -> pd.DataFrame:
    """Check D10's reading on the model-selection months: is each level's miss rate within
    one percentage point of the level (half a point at 0.005)?"""
    part = scores[
        (scores["period"] == PERIODS[2][0])
        & (scores["drivers"] == "all")
        & (scores["model"] == model)
    ].copy()
    part["tolerance"] = part["level"].map(lambda q: D10_TOLERANCE.get(round(q, 6), 0.01))
    part["within"] = (part["miss_rate"] - part["level"]).abs() <= part["tolerance"]
    return part[["model", "target", "level", "car_days", "miss_rate", "tolerance", "within"]]


def smape(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Equation (6) of Lee, Li and Low (2019) for linked drivers on the model-selection
    months, with each model's median as the point forecast: the mean over drivers of each
    driver's mean |x - x^| / (x + x^), in percent (0 when both are 0)."""
    part = frame[(frame["period"] == PERIODS[2][0]) & (frame["linked"] == 1)]
    rows = []
    for name in names:
        for target, truth in (("stay", "true_stay_h"), ("energy", "true_energy_kwh")):
            col = f"{name}:{target}:0.5"
            have = part[part[col].notna()]
            if not len(have):
                continue
            x, xhat = have[truth].to_numpy(float), have[col].to_numpy(float)
            total = x + xhat
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.where(total > 0, np.abs(x - xhat) / total, 0.0)
            per_driver = pd.Series(err, index=have["user_id"].to_numpy()).groupby(level=0).mean()
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "drivers": len(per_driver),
                    "car_days": len(have),
                    "smape_percent": 100 * float(per_driver.mean()),
                }
            )
    return pd.DataFrame(rows)


def matched(
    frame: pd.DataFrame,
    revealed: pd.DataFrame,
    names: dict[str, str],
    *,
    miss: float = 0.10,
    window_days: float = 90,
) -> pd.DataFrame:
    """H4 (analysis plan, Section 9): each model's stay bound at matched 90% reliability.

    Each decision day, a model's level is the split conformal level (`daylevel`) of its own
    earlier car-days' stay scores (the largest level at which each bound held), over the
    car-days revealed before that morning and decided within the last `window_days`, at a
    miss rate of 10% of car-days; its bound for each car that day is read at that level. The
    same rule for every model, so the comparison is at matched reliability; the achieved
    coverage is reported beside the hours kept. `revealed` gives each decision's
    ``revealed_at``. Car-days without a prediction are left out.
    """
    days = frame[["decision_time"]].merge(revealed, on="decision_time", how="left")
    times = frame["decision_time"].to_numpy()
    seen = days["revealed_at"].to_numpy()
    deadline = frame["true_deadline"].to_numpy(float)
    rows = []
    for name, label in names.items():
        grid = frame[_columns(name, "stay")].to_numpy(float)
        ok = ~np.isnan(grid).any(axis=1)
        score = held_level(grid, deadline, "stay")
        for t in np.unique(times[ok]):
            past = ok & (seen < t) & (times >= t - np.timedelta64(int(window_days * 86400), "s"))
            level = conformal_level(score[past], miss)
            today = np.flatnonzero(ok & (times == t))
            bound = at_levels(grid[today], level)
            steps = np.floor(np.minimum(bound, 10.0) / STEP_H + 1e-9)
            rows.append(
                pd.DataFrame(
                    {
                        "decision_time": frame["decision_time"].iloc[today].to_numpy(),
                        "session_id": frame["session_id"].iloc[today].to_numpy(),
                        "rule": label,
                        "level": level,
                        "past_car_days": int(past.sum()),
                        "bound_h": steps * STEP_H,
                        "held": steps <= deadline[today],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def figure_one(
    frame: pd.DataFrame, names: dict[str, str], level: float = FIGURE_LEVEL
) -> pd.DataFrame:
    """Per month and model: bounds given, share that held, and mean hours kept by the bound
    at `level` as the engine uses it (rounded down to 15 minutes, at most the horizon)."""
    rows = []
    for name, label in names.items():
        pred = frame[_columns(name, "stay")].to_numpy(float)
        bound = at_level(pred, level)
        ok = ~np.isnan(bound)
        steps = np.floor(np.minimum(bound, 10.0) / STEP_H + 1e-9)
        part = pd.DataFrame(
            {
                "month": frame["month"],
                "given": ok,
                "held": np.where(ok, steps <= frame["true_deadline"].to_numpy(), np.nan),
                "hours": np.where(ok, steps * STEP_H, np.nan),
            }
        )
        monthly = part.groupby("month").agg(
            given=("given", "sum"),
            car_days=("given", "size"),
            held_share=("held", "mean"),
            mean_bound_h=("hours", "mean"),
        )
        rows.append(monthly.reset_index().assign(rule=label))
    return pd.concat(rows, ignore_index=True)
