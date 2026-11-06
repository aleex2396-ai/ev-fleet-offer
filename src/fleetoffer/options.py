"""Day-level calibration on delivery, measured on the model-selection months (May to August 2019).

The first fleet-layer run (`fleetoffer checks`) gave a learned offer of zero on every scored
weekday of those months under the registered day-level rule, on both base sets, and check D9
called for the one-dimensional fallback (`docs/prior-looks.md`). This module measures, on the
same months only, the registered rule under the fallback and the same rule calibrated on
delivery instead of on every bound holding (`daylevel.DeliveryCalibrated`), each with the
three offer forms: programme (24) on the fixed base set, programme (24) on the day-average
base set, and the fallback. Nothing from September 2019 onward is read.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fleetoffer import devchecks, percar
from fleetoffer.daylevel import DeliveryCalibrated, MakeOffer
from fleetoffer.devchecks import SELECTION_START, make_arm, selection_days
from fleetoffer.engine import fleet_replay, training_base_set
from fleetoffer.offer import BaseSet, solve, solve_segment

FORMS = ("fixed", "average", "segment")  # base set fixed from training, day-average, fallback
MODELS = {"4": "learned", "5": "population", "6": "stratified"}
RUNS = (
    [("1", form) for form in FORMS]
    + [("4", "segment"), ("5", "segment")]
    + [(f"{arm}d", form) for arm in MODELS for form in FORMS]
)


def make_offer(form: str, base: BaseSet) -> MakeOffer:
    if form == "fixed":
        return lambda fleet: solve(fleet, base)
    if form == "average":
        return lambda fleet: solve(fleet, BaseSet.average(fleet))
    if form == "segment":
        return solve_segment
    raise ValueError(f"unknown offer form {form}")


def engine_options(form: str, base: BaseSet) -> dict:
    """What `fleet_replay` needs to make the same offer as `make_offer`."""
    if form == "segment":
        return {"method": "segment"}
    return {"method": "programme", "base": base if form == "fixed" else None}


def make_run(name: str, form: str, base: BaseSet):
    """Arm `name` of `devchecks.make_arm`, or, with a trailing "d", arm 4, 5 or 6 with the
    primary rule calibrated on delivery of the offer form `form`."""
    if not name.endswith("d"):
        rule = make_arm(name)
    else:
        model = devchecks.make_arm(name[0]).per_car
        rule = DeliveryCalibrated(
            model, make_offer(form, base), name=f"{name} {MODELS[name[0]]}, aci on delivery"
        )
    rule.name = f"{rule.name} | {form}"
    return rule


def run_one(data_root: str, name: str, form: str) -> dict:
    sessions, entries = percar.load(Path(data_root))
    base = training_base_set(sessions, percar.SPLIT_END)
    rule = make_run(name, form, base)
    result = fleet_replay(
        sessions,
        rule,
        entries=entries,
        until=percar.SELECTION_END,
        offers_from=SELECTION_START,
        **engine_options(form, base),
    )
    days = result.days.assign(arm=name, form=form)
    out = {"name": name, "form": form, "days": days}
    if hasattr(rule, "log"):
        out["log"] = pd.DataFrame(rule.log).assign(arm=name, form=form)
    return out


def run_saved(fn, jobs: list[tuple], workers: int, store: Path | None = None) -> list:
    """`fn(*job)` for every job, in parallel. With `store`, each finished job is saved there
    as it ends and a job already saved is read back instead of run again, so an interrupted
    call can be resumed."""
    store = Path(store) if store is not None else None
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)

    def path(job):
        return store / ("-".join(str(part) for part in job[1:]) + ".pkl")

    done, todo = {}, []
    for job in jobs:
        if store is not None and path(job).exists():
            done[job] = pd.read_pickle(path(job))
        else:
            todo.append(job)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            done[job] = future.result()
            if store is not None:
                pd.to_pickle(done[job], path(job))
    return [done[job] for job in jobs]


def run_all(data_root: Path, workers: int = 10, store: Path | None = None) -> list[dict]:
    """Every run in RUNS (see `run_saved` for `store`)."""
    jobs = [(str(data_root), name, form) for name, form in RUNS]
    return run_saved(run_one, jobs, workers, store)


def summary(runs: list[dict]) -> pd.DataFrame:
    """Per run, on the scored weekdays of the model-selection months."""
    rows = []
    for run in runs:
        days = selection_days(run["days"])
        row = {
            "arm": run["name"],
            "form": run["form"],
            "rule": days["rule"].iloc[0] if len(days) else run["name"],
            "days": len(days),
            "size_mean_kwh": float(days["size_kwh"].mean()),
            "size_median_kwh": float(days["size_kwh"].median()),
            "zero_offer_share": float((days["size_kwh"] <= 1e-9).mean()),
            "fixed_delivered": float(days["fixed_delivered"].astype(float).mean()),
            "hindsight_delivered": float(days["hindsight_delivered"].astype(float).mean()),
            "all_bounds_held": float(days["all_bounds_held"].astype(float).mean()),
            "exact_true_kwh": float(days["exact_true_kwh"].mean()),
            "exact_learned_kwh": float(days["exact_learned_kwh"].mean()),
            "solve_seconds_max": float(days["solve_seconds"].max()),
        }
        if "log" in run:
            log = run["log"]
            log = log[log["fitted"] & (log["decision_time"] >= SELECTION_START)]
            row["level_median"] = float(log["level"].median())
            row["level_max"] = float(log["level"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def spreads(runs: list[dict]) -> dict:
    """Learned against population (arm 4 against arm 5) for each form, with the rule on
    delivery: the week-to-week spread report of `devchecks.spread`."""
    by = {(r["name"], r["form"]): r["days"] for r in runs}
    out = {}
    for form in FORMS:
        if ("4d", form) in by and ("5d", form) in by:
            weekly, report = devchecks.spread(by[("4d", form)], by[("5d", form)])
            report["weekly"] = weekly.to_dict(orient="list")
            out[form] = report
    return out
