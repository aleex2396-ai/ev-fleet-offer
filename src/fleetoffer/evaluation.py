"""The runs after the `analysis-plan` tag (analysis plan, Sections 7 to 9), each once, from the
tagged commit: the frozen development evaluation (Caltech, September 2019 to February 2020),
the sealed Caltech months (from 1 March 2020) and the replication site (JPL).

Everything here is fixed before the tag (`docs/registration.md`):

- **Primary design:** the registered arms with the registered day-level rule, on the
  one-dimensional fallback offer, which check D9 called for on the model-selection months.
- **Secondary analysis, chosen after the May to August 2019 look:** arms 4, 5 and 6 calibrated
  on delivery (`daylevel.DeliveryCalibrated`) with the day-average base set (4d, 5d, 6d).
- **Descriptive:** the level sweep and its diagnostic with the empty cars known (`levels`).

The sealed Caltech run replays the whole Caltech record and makes offers from 1 March 2020; it
adds, for H3, arms whose learning stops on 1 March 2020 (`Frozen`): no refit, no new day
score, the calibration level in force that day. JPL is run in two tracks, both starting from
the pipeline as it stood at the end of the frozen development evaluation (every refit and day
score through February 2020): track 1 (primary) keeps it frozen and shows it no JPL history;
track 2 drops its Caltech car-days and day scores, keeps its fitted models and working level,
and learns from JPL outcomes as they mature, refitting once a JPL window holds enough car-days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer import data, levels, options, percar
from fleetoffer.daylevel import Calibrated
from fleetoffer.engine import Daily, Oracle, fleet_replay
from fleetoffer.features import LOCAL_TZ
from fleetoffer.interface import DayOutcome, DayView, FleetRule
from fleetoffer.offer import CAP_KW, STEPS, BaseSet
from fleetoffer.quantiles import PerCarRule
from fleetoffer.resample import paired_interval, rate_interval

EVALUATION_START = percar.SELECTION_END  # 1 September 2019, Pacific
EVALUATION_END = data.TEST_PERIODS["caltech"]  # 1 March 2020, Pacific
SEALED_START = EVALUATION_END


@dataclass(frozen=True)
class Window:
    """Where and when a run makes offers. `kind` "jpl" means the pipeline is first trained on
    Caltech up to `train_until` and then run on the other site's table."""

    kind: str  # "caltech" or "jpl"
    until: pd.Timestamp | None  # decisions stop before this
    offers_from: pd.Timestamp | None
    freeze_at: pd.Timestamp | None = None  # the H3 arms stop learning here
    train_until: pd.Timestamp | None = None  # JPL: the end of Caltech training
    reopening_year: int | None = None


def _local(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="America/Los_Angeles")


# The rehearsal runs every path of the three runs on Caltech development data only (June to
# August 2019; nothing from September 2019 is read), so that the tagged commit has run end to
# end before any sealed period opens. Its numbers are not results.
REHEARSAL_OTHER_SITE_FROM = _local("2019-06-01")
WINDOWS = {
    "evaluation": Window("caltech", EVALUATION_END, EVALUATION_START),
    "sealed": Window("caltech", None, SEALED_START, SEALED_START, reopening_year=2021),
    "jpl": Window("jpl", None, None, train_until=EVALUATION_END),
    "rehearsal-evaluation": Window("caltech", _local("2019-07-01"), _local("2019-06-01")),
    "rehearsal-sealed": Window(
        "caltech", percar.SELECTION_END, _local("2019-07-01"), _local("2019-07-01"), None, 2019
    ),
    "rehearsal-jpl": Window(
        "jpl", percar.SELECTION_END, None, train_until=REHEARSAL_OTHER_SITE_FROM
    ),
}
PERIODS = tuple(WINDOWS)
OFFERS_FROM = {name: w.offers_from for name, w in WINDOWS.items()}


def is_rehearsal(period: str) -> bool:
    return period.startswith("rehearsal")


def stage(period: str) -> str:
    """The run a period stands for: "evaluation", "sealed" or "jpl"."""
    return period.removeprefix("rehearsal-")


REGISTERED_FORM = "segment"  # D9 triggered on May to August 2019 (docs/registration.md)
SECONDARY_FORM = "average"
REGISTERED = ("1", "2", "3", "4", "5", "6", "4s", "5s", "6s")
SECONDARY = ("1a", "4d", "5d", "6d")  # "1a": the oracle on the day-average base set
FROZEN_ARMS = ("4f", "5f", "4df", "5df")  # H3: learning stops on 1 March 2020
JPL_TRACKS = ("frozen", "updating")
SWEEP_MODELS = ("4", "5", "6", "4k", "5k", "6k")
SWEEP_FORMS = {"evaluation": options.FORMS, "sealed": options.FORMS, "jpl": ("segment",)}
# JPL's sweep is on the registered (fallback) form only: programme (24) at JPL's fleet size, at
# every level of every day, does not fit the time available.

# Sensitivities of the analysis plan (Section 3) that change only a setting of the engine:
# the decision at 09:00 (horizon 09:00 to 19:00) and power caps of 3.3 and 11 kW. Each is run
# on the frozen evaluation and the sealed months for the oracle and arms 4 and 5 under both
# designs; the arm is named "<arm>~<setting>".
SENSITIVITIES = {
    "at0900": {"daily": Daily(at="09:00")},
    "cap3.3": {"cap": 3.3},
    "cap11": {"cap": 11.0},
}
SENSITIVITY_ARMS = (
    ("1", REGISTERED_FORM),
    ("4", REGISTERED_FORM),
    ("5", REGISTERED_FORM),
    ("1a", SECONDARY_FORM),
    ("4d", SECONDARY_FORM),
    ("5d", SECONDARY_FORM),
)
NEVER = pd.Timestamp("2100-01-01", tz="UTC")  # offers_from for a run that only learns
# The oracle learns nothing, so its days can be run apart. At JPL its programme (24) on the
# day-average base set, with about twice Caltech's cars, is run in half-year pieces at once.
ORACLE_PIECES = {
    "jpl": ["2000-01-01"]
    + [f"{y}-{m:02d}-01" for y in (2018, 2019, 2020, 2021) for m in (1, 7)][1:]
    + ["2100-01-01"],
    "rehearsal-jpl": ["2000-01-01"]
    + [f"2019-{m}" for m in ("06-10", "06-20", "07-01", "07-10", "07-20", "08-01", "08-15")]
    + ["2100-01-01"],
}
BASE_FILE = Path(__file__).with_name("h0.json")  # committed at the tag, loaded, never recomputed


def base_set(path: Path = BASE_FILE) -> BaseSet:
    stored = json.loads(Path(path).read_text())
    return BaseSet(np.asarray(stored["power_kw"], dtype=float), float(stored["energy_kwh"]))


def write_base_set(base: BaseSet, path: Path = BASE_FILE) -> None:
    if len(base.power) != STEPS:
        raise ValueError(f"the base set must have {STEPS} steps")
    stored = {
        "source": "engine.training_base_set over Caltech weekdays at 10:00 through April 2019",
        "energy_kwh": float(base.energy),
        "power_kw": [float(x) for x in base.power],
    }
    Path(path).write_text(json.dumps(stored, indent=2) + "\n")


def load(data_root, period: str, *, unseal: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sessions and app entries of the table a period makes offers on: Caltech through
    February 2020 for the frozen evaluation; the whole Caltech record for the sealed run; the
    whole JPL record. A rehearsal reads Caltech development data only (its "other site" is
    Caltech from June 2019)."""
    if period not in WINDOWS:
        raise ValueError(f"unknown period {period}")
    if is_rehearsal(period):
        sessions, entries = percar.load(Path(data_root))
        if WINDOWS[period].kind == "jpl":
            sessions = sessions[sessions["connect_time"] >= REHEARSAL_OTHER_SITE_FROM]
            sessions = sessions.reset_index(drop=True)
            entries = entries[entries["session_id"].isin(set(sessions["session_id"]))]
        return sessions, entries.reset_index(drop=True)
    if period == "evaluation":
        return percar.load(Path(data_root), until=EVALUATION_END)
    if period == "sealed":
        return percar.load(Path(data_root), until=None, unseal=unseal)
    return percar.load(Path(data_root), until=None, site="jpl", unseal=unseal)


def arms(period: str) -> list[tuple[str, str]]:
    out = [(a, REGISTERED_FORM) for a in REGISTERED] + [(a, SECONDARY_FORM) for a in SECONDARY]
    if stage(period) == "sealed":
        out += [(a, REGISTERED_FORM if "d" not in a else SECONDARY_FORM) for a in FROZEN_ARMS]
    return out


def _per_car_of(rule: FleetRule) -> PerCarRule | None:
    if isinstance(rule, PerCarRule):
        return rule
    return getattr(rule, "per_car", None)


class Frozen(FleetRule):
    """A rule whose learning stops: from `when` (or from `freeze()`), its models are not
    refitted, its day-level level is the one in force at `when`, and it is told no outcome.
    With `hide_history`, its inputs also see no other session's history (JPL, track 1)."""

    def __init__(self, rule: FleetRule, when: pd.Timestamp, *, hide_history: bool = False):
        self.rule, self.when, self.hide_history = rule, when, hide_history
        self.name = f"{rule.name} | frozen"
        self.is_frozen = False

    @property
    def log(self) -> list[dict]:
        return getattr(self.rule, "log", [])

    @property
    def per_car(self) -> PerCarRule | None:
        return _per_car_of(self.rule)

    def freeze(self) -> None:
        per_car = _per_car_of(self.rule)
        if per_car is not None:
            per_car.frozen = True
            per_car.hide_history = self.hide_history
        if isinstance(self.rule, Calibrated):
            self.rule.frozen_at = self.when
        self.is_frozen = True

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_frozen and day.decision_time >= self.when:
            self.freeze()
        return self.rule.bounds(day)

    def observe(self, outcome: DayOutcome) -> None:
        if not self.is_frozen:
            self.rule.observe(outcome)


def make_rule(
    name: str, form: str, base: BaseSet, freeze_at=SEALED_START, cap: float = CAP_KW
) -> FleetRule:
    """An arm of `arms`: "f" marks the H3 arm whose learning stops at `freeze_at`."""
    if name.endswith("f"):
        return Frozen(options.make_run(name[:-1], form, base, cap), freeze_at)
    return options.make_run("1" if name == "1a" else name, form, base, cap)


def _rows(rule: FleetRule, result, name: str, form: str) -> dict:
    out = {"name": name, "form": form, "days": result.days.assign(arm=name, form=form)}
    out["cars"] = result.cars.assign(arm=name, form=form)
    log = getattr(rule, "log", None)
    if log:
        out["log"] = pd.DataFrame(log).assign(arm=name, form=form)
    per_car = _per_car_of(rule)
    if per_car is not None and name in ("4", "5"):
        out["predictions"] = percar.scored(per_car, result.cars)
    return out


def run_arm(
    data_root: str, period: str, name: str, form: str, unseal: bool = False, setting: str = ""
) -> dict:
    """One arm on the frozen evaluation or the sealed Caltech months, at the registered
    settings or at one of `SENSITIVITIES` (then named "<arm>~<setting>")."""
    window = WINDOWS[period]
    sessions, entries = load(data_root, period, unseal=unseal)
    base = base_set()
    changed = SENSITIVITIES[setting] if setting else {}
    rule = make_rule(name, form, base, window.freeze_at, cap=changed.get("cap", CAP_KW))
    result = fleet_replay(
        sessions,
        rule,
        entries=entries,
        until=window.until,
        offers_from=window.offers_from,
        **options.engine_options(form, base),
        **changed,
    )
    rows = _rows(rule, result, name, form)
    if setting:
        label = f"{name}~{setting}"
        rows["name"] = label
        for key in ("days", "cars", "log"):
            if key in rows:
                rows[key] = rows[key].assign(arm=label)
        rows.pop("predictions", None)
    return rows


def trained_on_caltech(data_root: str, rule: FleetRule, until=EVALUATION_END) -> FleetRule:
    """Run `rule` over Caltech before `until`, learning only (no offer is made)."""
    if isinstance(rule, Oracle):
        return rule
    if until > percar.SELECTION_END:
        sessions, entries = load(data_root, "evaluation")
    else:
        sessions, entries = percar.load(Path(data_root))
    fleet_replay(sessions, rule, entries=entries, until=until, method="bounds")
    return rule


def for_jpl(rule: FleetRule, track: str, trained_until=EVALUATION_END) -> FleetRule:
    """The Caltech-trained rule in one of the two JPL tracks."""
    if isinstance(rule, Oracle):
        return rule
    if track == "frozen":
        frozen = Frozen(rule, trained_until, hide_history=True)
        frozen.freeze()
        frozen.name = f"{rule.name} | jpl frozen"
        return frozen
    if track == "updating":
        if hasattr(rule, "forget_history"):
            rule.forget_history()
        else:
            _per_car_of(rule).forget_history()
        rule.name = f"{rule.name} | jpl updating"
        return rule
    raise ValueError(f"unknown JPL track {track}")


def run_jpl(
    data_root: str,
    period: str,
    name: str,
    form: str,
    track: str,
    unseal: bool = False,
    piece: int | None = None,
) -> dict:
    """One arm at the other site in one track; the oracle may be run in `ORACLE_PIECES`."""
    window = WINDOWS[period]
    base = base_set()
    made = make_rule(name, form, base)
    if piece is not None and not isinstance(made, Oracle):
        raise ValueError("only the oracle, which learns nothing, is run in pieces")
    trained = trained_on_caltech(data_root, made, window.train_until)
    rule = for_jpl(trained, track, window.train_until)
    sessions, entries = load(data_root, period, unseal=unseal)
    until, first = window.until, None
    if piece is not None:
        edges = [_local(day) for day in ORACLE_PIECES[period]]
        first, last = edges[piece], edges[piece + 1]
        until = last if until is None else min(until, last)
    result = fleet_replay(
        sessions,
        rule,
        entries=entries,
        until=until,
        offers_from=first,
        **options.engine_options(form, base),
    )
    out = _rows(rule, result, name, form)
    for key in ("days", "cars", "log"):
        if key in out:
            frame = out[key].assign(track=track)
            if first is not None and "decision_time" in frame.columns:
                frame = frame[frame["decision_time"] >= first].reset_index(drop=True)
            out[key] = frame
    return out


def run_sweep(data_root: str, period: str, model: str, form: str, unseal: bool = False) -> dict:
    """The level sweep of one model and form on a period (JPL: track 1, the frozen pipeline)."""
    window = WINDOWS[period]
    base = base_set()
    if window.kind == "jpl":
        rule = levels.sweep_rule(model, form, base, offers_from=NEVER)
        trained_on_caltech(data_root, rule, window.train_until)
        rule.per_car.frozen = True
        rule.per_car.hide_history = True
        rule.offers_from = None
    else:
        rule = levels.sweep_rule(model, form, base, offers_from=window.offers_from)
    sessions, entries = load(data_root, period, unseal=unseal)
    fleet_replay(
        sessions,
        rule,
        entries=entries,
        until=window.until,
        method="bounds",
        reveal_empty=rule.drop_empty,
    )
    return {"model": model, "form": form, "sweep": levels.sweep_rows(rule, model)}


def jobs(period: str, data_root: str, unseal: bool) -> list[tuple]:
    """Every run of a period, as arguments of `run_job`."""
    out = []
    if WINDOWS[period].kind == "jpl":
        for track in JPL_TRACKS:
            for name, form in arms(period):
                if name.startswith("1") and track == "updating":
                    continue  # the oracle learns nothing; one run serves both tracks
                if name == "1a":
                    pieces = range(len(ORACLE_PIECES[period]) - 1)
                    out += [
                        (data_root, "jpl", period, name, form, track, unseal, k) for k in pieces
                    ]
                    continue
                out.append((data_root, "jpl", period, name, form, track, unseal))
    else:
        out += [(data_root, "arm", period, name, form, unseal) for name, form in arms(period)]
        for setting in SENSITIVITIES:
            out += [
                (data_root, "arm", period, name, form, unseal, setting)
                for name, form in SENSITIVITY_ARMS
            ]
    forms = SWEEP_FORMS[stage(period)]
    for model in SWEEP_MODELS:
        out += [(data_root, "sweep", period, model, form, unseal) for form in forms]
    return out


def run_job(data_root: str, kind: str, *rest):
    return {"arm": run_arm, "jpl": run_jpl, "sweep": run_sweep}[kind](data_root, *rest)


def run_period(
    period: str, data_root: Path, *, unseal: bool = False, workers: int = 10, store=None
) -> list[dict]:
    """Every run of a period in parallel; with `store`, resumable (`options.run_saved`)."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period}")
    return options.run_saved(run_job, jobs(period, str(data_root), unseal), workers, store)


# Hypotheses and reports (analysis plan, Section 9).


def scored_days(days: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Scored weekdays on which an offer was made, from `start`."""
    keep = days["size_kwh"].notna() & (days["cars_with_energy"] >= levels.D3_MIN_CARS)
    if start is not None:
        keep &= days["decision_time"] >= start
    return days[keep].reset_index(drop=True)


def arm_summary(days: pd.DataFrame) -> pd.DataFrame:
    """Per arm (and track): size and delivery over the scored weekdays."""
    keys = [k for k in ("track", "arm", "form") if k in days.columns]
    rows = days.assign(
        fixed=days["fixed_delivered"].astype(float),
        hindsight=days["hindsight_delivered"].astype(float),
        held=days["all_bounds_held"].astype(float),
        zero=(days["size_kwh"] <= 1e-9).astype(float),
    )
    return (
        rows.groupby(keys)
        .agg(
            days=("decision_time", "size"),
            size_mean_kwh=("size_kwh", "mean"),
            zero_offer_share=("zero", "mean"),
            fixed_delivered=("fixed", "mean"),
            hindsight_delivered=("hindsight", "mean"),
            all_bounds_held=("held", "mean"),
            exact_true_kwh=("exact_true_kwh", "mean"),
        )
        .reset_index()
    )


def monthly(days: pd.DataFrame) -> pd.DataFrame:
    """Per arm and month: mean size and share of days delivered, with that share's interval
    from resampling the month's whole weeks (Figures 2 and 3)."""
    month = days["decision_time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.to_period("M")
    keys = [k for k in ("track", "arm", "form") if k in days.columns]
    rows = days.assign(month=month.astype(str), fixed=days["fixed_delivered"].astype(float))
    out = []
    for key, group in rows.groupby([*keys, "month"], sort=True):
        rate = rate_interval(group, "fixed")
        out.append(
            dict(zip([*keys, "month"], key, strict=True))
            | {
                "days": len(group),
                "size_mean_kwh": float(group["size_kwh"].mean()),
                "fixed_delivered": float(group["fixed"].mean()),
                "fixed_delivered_low": rate["low"],
                "fixed_delivered_high": rate["high"],
            }
        )
    return pd.DataFrame(out)


def paired(days: pd.DataFrame, a: str, b: str, *, weeks: int = 1) -> dict:
    """H1's contrast: arm a against arm b, relative difference of mean size, paired by day,
    with a week-block interval; both arms' delivery rates beside it."""
    wide = days.pivot_table(
        index="decision_time", columns="arm", values="size_kwh", aggfunc="first"
    ).reset_index()
    delivered = days.pivot_table(
        index="decision_time", columns="arm", values="fixed_delivered", aggfunc="first"
    )
    both = wide.dropna(subset=[a, b])
    delivered = delivered.loc[both["decision_time"]]  # the same days as the sizes
    out = {
        "a": a,
        "b": b,
        "days": len(both),
        "mean_a_kwh": float(both[a].mean()),
        "mean_b_kwh": float(both[b].mean()),
        "delivered_a": float(delivered[a].astype(float).mean()),
        "delivered_b": float(delivered[b].astype(float).mean()),
    }
    if out["mean_b_kwh"] <= 1e-9 and out["mean_a_kwh"] <= 1e-9:
        return out | {"reading": "not testable: both arms offer zero"}
    absolute = paired_interval(both, a, b, weeks=weeks)
    if out["mean_b_kwh"] <= 1e-9:
        # the relative difference is undefined; the absolute one is reported
        return out | {
            "absolute_kwh": absolute,
            "reading": "not testable on the relative scale: the second arm offers zero",
        }
    relative = paired_interval(both, a, b, weeks=weeks, relative=True)
    passed = relative["estimate"] >= 0.10 and relative["low"] > 0
    return out | {
        "relative": relative,
        "absolute_kwh": absolute,
        "reading": "pass" if passed else "fail",
    }


def h2(days: pd.DataFrame) -> dict:
    """H2: arm 3 delivered on at least 98% of days and at most half the size of arm 4."""
    three, four = (days[days["arm"] == a] for a in ("3", "4"))
    delivered = float(three["fixed_delivered"].astype(float).mean())
    size3, size4 = float(three["size_kwh"].mean()), float(four["size_kwh"].mean())
    return {
        "delivered_3": delivered,
        "size_3_kwh": size3,
        "size_4_kwh": size4,
        "reading": "pass" if delivered >= 0.98 and size3 <= 0.5 * size4 + 1e-12 else "fail",
    }


def reopening(days: pd.DataFrame, year: int = 2021, window: int = 10, cars: float = 5) -> str:
    """The reopening date (analysis plan, Section 7): the first weekday of `year` on which the
    running average of cars present at 10:00 over the last `window` weekdays, that day included,
    is at least `cars`, applied in time order (each day's average uses that day and earlier
    weekdays only)."""
    local = days["decision_time"].dt.tz_convert(LOCAL_TZ)
    rows = days.assign(local=local).sort_values("decision_time")
    rows = rows[rows["local"].dt.year == year]
    average = rows["cars"].astype(float).rolling(window, min_periods=window).mean()
    hit = rows[average >= cars]
    if hit.empty:
        return ""
    return str(hit["local"].iloc[0].date())


def h3(days: pd.DataFrame, start: str, arm: str, frozen_arm: str) -> dict:
    """H3 from the reopening date: (a) the frozen arm delivered on fewer than 80% of the first
    20 scored weekdays; (b) the adaptive arm reaches a 90% running average of delivered days
    over 20 scored weekdays within 40 scored weekdays."""
    if not start:
        return {"reading": "not run: no reopening date in 2021"}
    since = pd.Timestamp(start, tz=LOCAL_TZ)
    out = {"reopening": start, "arm": arm, "frozen_arm": frozen_arm}
    adaptive = scored_days(days[days["arm"] == arm], since).sort_values("decision_time")
    frozen = scored_days(days[days["arm"] == frozen_arm], since).sort_values("decision_time")
    first20 = frozen["fixed_delivered"].astype(float).head(20)
    out["frozen_first20_delivered"] = float(first20.mean()) if len(first20) == 20 else np.nan
    out["h3a"] = "pass" if len(first20) == 20 and first20.mean() < 0.80 else "fail"
    delivered = adaptive["fixed_delivered"].astype(float).to_numpy()
    running = pd.Series(delivered).rolling(20, min_periods=20).mean().to_numpy()
    reached = np.flatnonzero(running >= 0.90 - 1e-12)
    out["adaptive_reached_at_scored_weekday"] = int(reached[0] + 1) if len(reached) else None
    out["h3b"] = "pass" if len(reached) and reached[0] + 1 <= 40 else "fail"
    out["adaptive_scored_weekdays"] = len(delivered)
    return out


def h4(
    predictions_4: pd.DataFrame,
    predictions_5: pd.DataFrame,
    revealed: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> dict:
    """H4: the learned per-car stay bound keeps at least 0.5 hours more than the population
    bound at matched 90% reliability (`percar.matched`), on the period's car-days on which
    both models gave a bound."""
    keys = ["decision_time", "session_id"]
    own = [c for c in predictions_5.columns if c.startswith("population:")]
    frame = predictions_4.merge(predictions_5[keys + own], on=keys, validate="1:1")
    names = {"gbdt": "learned", "population": "population"}
    matched = percar.matched(frame, revealed, names)
    keep = matched["decision_time"] >= start
    if end is not None:
        keep &= matched["decision_time"] < end
    period = matched[keep]
    both = period.pivot_table(index=keys, columns="rule", values="bound_h").dropna().index
    period = period.set_index(keys).loc[both].reset_index()
    by = period.groupby("rule").agg(hours=("bound_h", "mean"), coverage=("held", "mean"))
    gain = float(by.loc["learned", "hours"] - by.loc["population", "hours"])
    return {
        "car_days": len(both),
        "learned_hours": float(by.loc["learned", "hours"]),
        "population_hours": float(by.loc["population", "hours"]),
        "learned_coverage": float(by.loc["learned", "coverage"]),
        "population_coverage": float(by.loc["population", "coverage"]),
        "gain_hours": gain,
        "reading": "pass" if gain >= 0.5 else "fail",
    }


H3_PAIRS = (("4", "4f"), ("5", "5f"), ("4d", "4df"), ("5d", "5df"))


def contrasts(scored: pd.DataFrame) -> dict:
    """H1's contrast under both designs, with the two-week-block sensitivity, and arm 4
    against arm 6 (what individual and learned information adds beyond the time of day)."""
    out = {}
    for label, (a, b) in {
        "h1_registered": ("4", "5"),
        "h1_secondary": ("4d", "5d"),
        "learned_vs_stratified_registered": ("4", "6"),
        "learned_vs_stratified_secondary": ("4d", "6d"),
        **{
            f"h1_{design}~{setting}": (f"{a}~{setting}", f"{b}~{setting}")
            for setting in SENSITIVITIES
            for design, a, b in (("registered", "4", "5"), ("secondary", "4d", "5d"))
        },
    }.items():
        if {a, b} <= set(scored["arm"]):
            out[label] = {
                "one_week": paired(scored, a, b),
                "two_week": paired(scored, a, b, weeks=2),
            }
    return out


def hypotheses(period: str, scored: pd.DataFrame, days: pd.DataFrame, predictions: dict) -> dict:
    """The readings of the plan's Section 9 for one period. H1 and H4 are confirmatory on the
    frozen evaluation only; elsewhere every contrast is descriptive."""
    out: dict = {"period": period}
    window = WINDOWS[period]
    if window.kind == "jpl":
        for track in JPL_TRACKS:
            frame = scored[(scored["track"] == track) | (scored["arm"] == "1")]
            out[track] = contrasts(frame) | {"h2": h2(frame)}
        return out
    out |= contrasts(scored)
    out["h2"] = h2(scored)
    if "4" in predictions and "5" in predictions:
        revealed = days.drop_duplicates("decision_time")[["decision_time", "revealed_at"]]
        out["h4"] = h4(
            predictions["4"], predictions["5"], revealed, window.offers_from, window.until
        )
    if window.reopening_year is not None:
        start = reopening(days[days["arm"] == "1"], year=window.reopening_year)
        out["reopening"] = start
        out["h3"] = {f"{a} against {f}": h3(days, start, a, f) for a, f in H3_PAIRS}
    return out


def failures(days: pd.DataFrame, cars: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Figure 4's table: per scored day of one arm, the share of present cars whose deadline,
    energy, or either bound failed, beside whether the offer was delivered under each check."""
    mine = cars[cars["arm"] == arm]
    mine = mine.assign(
        deadline_failed=~mine["deadline_held"].astype(bool),
        energy_failed=~mine["energy_held"].astype(bool),
    )
    mine = mine.assign(any_failed=mine["deadline_failed"] | mine["energy_failed"])
    shares = (
        mine.groupby("decision_time")[["deadline_failed", "energy_failed", "any_failed"]]
        .mean()
        .rename(
            columns={
                "deadline_failed": "share_deadline_failed",
                "energy_failed": "share_energy_failed",
                "any_failed": "share_mispredicted",
            }
        )
        .reset_index()
    )
    keep = ["decision_time", "fixed_delivered", "hindsight_delivered", "size_kwh"]
    return days[days["arm"] == arm][keep].merge(shares, on="decision_time", how="inner")


def failure_summary(per_day: pd.DataFrame) -> dict:
    """How often each check delivers despite failed bounds (Figure 4's reported numbers)."""
    failed = per_day["share_mispredicted"] > 0
    out = {"days": len(per_day), "days_with_a_failed_bound": int(failed.sum())}
    for check in ("fixed_delivered", "hindsight_delivered"):
        delivered = per_day[check].astype(bool)
        out[f"{check}_given_a_failed_bound"] = (
            float(delivered[failed].mean()) if failed.any() else np.nan
        )
        not_delivered = per_day[~delivered]
        out[f"median_mispredicted_when_not_{check}"] = (
            float(not_delivered["share_mispredicted"].median()) if len(not_delivered) else np.nan
        )
        out[f"median_mispredicted_when_{check}"] = float(
            per_day[delivered]["share_mispredicted"].median()
        )
    return out
