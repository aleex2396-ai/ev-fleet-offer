"""End-to-end runs and the command line. Each run writes its tables, figures and a manifest.

Scores for a set of per-car decisions, always reported together, because a rule can always
hold more often by bounding less:

- ``held_share``: share of the bounds given that held.
- ``mean_bound_h``: average bound per decision, counting a decision with no bound as 0
  hours (the aggregator could not count on that car).
- ``given_share``: share of decisions where a bound was given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer import __version__, acn, calibrate, data
from fleetoffer.engine import Daily, energy_in_window, replay
from fleetoffer.figures import plot_monthly
from fleetoffer.models import FrozenQuantile, RollingQuantile
from fleetoffer.synthetic import Regime, demo_regimes, make_sessions

DEMO_TZ = "America/Los_Angeles"
DEMO_LEVEL = 0.1
DECISION = Daily(at="10:00", tz=DEMO_TZ)

# The months each site has sessions in the public API (checked 4 Nov 2026).
SITE_MONTHS = {"caltech": ("2018-04", "2021-09"), "jpl": ("2018-09", "2021-09")}

# Check D1 (analysis plan, Section 10a), fixed before any time series was fetched.
D1_SAMPLE = 300
D1_SEED = 20261105
D1_NAME = "d1-sample"
D1_MIN_USABLE = 0.95  # share of sampled sessions with a usable current series
D1_MAX_MEDIAN_ERROR = 0.05  # median |integrated - recorded| / recorded, sessions with energy


def _summarise(decisions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = decisions.assign(
        _held=decisions["held"].astype("Float64"),
        _bound=decisions["bound_h"].fillna(0.0),
        _given=decisions["given"].astype(int),
    )
    if not keys:
        d, keys = d.assign(_all="all"), ["_all"]
    out = d.groupby(keys, sort=True).agg(
        decisions=("_given", "size"),
        given=("_given", "sum"),
        held=("_held", "sum"),
        mean_bound_h=("_bound", "mean"),
    )
    out["held_share"] = (out["held"] / out["given"].where(out["given"] > 0)).astype(float)
    out["given_share"] = out["given"] / out["decisions"]
    out["held"] = out["held"].astype(int)
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


def fleet_days(decisions: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Cars present at each daily decision time (one rule's decisions; the roster is shared)."""
    one_rule = decisions[decisions["rule"] == decisions["rule"].iloc[0]]
    days = one_rule.groupby("decision_time").size().rename("cars_present").reset_index()
    days.insert(0, "day", days["decision_time"].dt.tz_convert(tz).dt.date)
    return days


def run_demo(
    out: Path,
    *,
    seed: int = 0,
    regimes: list[Regime] | None = None,
    end: str = "2003-10-01",
    fit_until: str = "2002-01-01",
) -> dict:
    """Run the pipeline on made-up fleets. This checks the plumbing; it is not a result.

    Every weekday at 10:00, each car plugged in is given a lower bound on its remaining stay
    by each rule, and the bound is scored by replay.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = make_sessions(regimes or demo_regimes(), end, tz=DEMO_TZ, seed=seed)

    rules = [
        FrozenQuantile(pd.Timestamp(fit_until, tz=DEMO_TZ), level=DEMO_LEVEL),
        RollingQuantile(30, level=DEMO_LEVEL),
        RollingQuantile(120, level=DEMO_LEVEL),
    ]
    decisions = pd.concat([replay(sessions, rule, daily=DECISION) for rule in rules])
    decisions = decisions.reset_index(drop=True)
    monthly = monthly_summary(decisions, tz=DEMO_TZ)
    overall = overall_summary(decisions)
    days = fleet_days(decisions, DEMO_TZ)

    sessions.to_csv(out / "sessions.csv", index=False)
    decisions.to_csv(out / "decisions.csv.gz", index=False)
    monthly.to_csv(out / "monthly.csv", index=False)
    overall.to_csv(out / "overall.csv", index=False)
    days.to_csv(out / "fleet_days.csv", index=False)
    figure = plot_monthly(
        monthly,
        DEMO_LEVEL,
        out / "monthly.png",
        rules=[rule.name for rule in rules],
        title="Made-up data: pipeline check, not a result",
        note=(
            f"Synthetic sessions (seed {seed}): a stable year, a near-empty half year, then a new "
            "population with shorter stays. Bounds given at 10:00 on weekdays to every car "
            "plugged in. Months with under 20 bounds are left blank."
        ),
    )

    manifest = {
        "run": "demo",
        "made_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "code": _git_state(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "parameters": {
            "seed": seed,
            "end": end,
            "fit_until": fit_until,
            "level": DEMO_LEVEL,
            "decision": {"at": DECISION.at, "time_zone": DECISION.tz, "weekdays_only": True},
            "rules": [rule.name for rule in rules],
        },
        "inputs": {"sessions.csv": _sha256(out / "sessions.csv"), "sessions": len(sessions)},
        "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json"),
        "figure": figure.name,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def d1_sample(sessions: pd.DataFrame) -> pd.DataFrame:
    """The seeded sample of development sessions for check D1, in plug-in order."""
    ids = np.sort(sessions["session_id"].astype(str).to_numpy())
    chosen = np.random.default_rng(D1_SEED).choice(ids, size=D1_SAMPLE, replace=False)
    picked = sessions[sessions["session_id"].astype(str).isin(set(chosen))]
    return picked.sort_values(["connect_time", "session_id"]).reset_index(drop=True)


def d1_table(sessions: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """One row per sampled session: usable or why not, recorded and integrated energy."""
    by_id = {str(r.get("sessionID")): r for r in records}
    rows = []
    for row in sessions.itertuples(index=False):
        series, reason = data.parse_series(by_id[str(row.session_id)])
        recorded = float(row.kwh_delivered)
        integrated = data.energy_kwh(series) if series is not None else np.nan
        rows.append(
            {
                "session_id": row.session_id,
                "reason": reason,
                "samples": len(series.times_h) if series is not None else 0,
                "negative_samples": int((series.amps < 0).sum()) if series is not None else 0,
                "kwh_recorded": recorded,
                "kwh_integrated": integrated,
                "abs_rel_error": abs(integrated - recorded) / recorded
                if series is not None and recorded > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def d1_summary(table: pd.DataFrame) -> dict:
    usable = table["reason"] == "usable"
    errors = table["abs_rel_error"].dropna()
    with_energy = table[usable & (table["kwh_recorded"] > 0)]
    usable_share = float(usable.mean())
    median_error = float(errors.median()) if len(errors) else float("nan")
    return {
        "sessions": len(table),
        "usable": int(usable.sum()),
        "usable_share": usable_share,
        "reasons": table["reason"].value_counts().to_dict(),
        "usable_with_recorded_energy": len(with_energy),
        "usable_with_zero_recorded_energy": int((usable & (table["kwh_recorded"] <= 0)).sum()),
        "median_abs_rel_error": median_error,
        "share_within_5_percent": float((errors <= 0.05).mean()) if len(errors) else float("nan"),
        "median_ratio_integrated_to_recorded": float(
            (with_energy["kwh_integrated"] / with_energy["kwh_recorded"]).median()
        )
        if len(with_energy)
        else float("nan"),
        "sessions_with_negative_samples": int((table["negative_samples"] > 0).sum()),
        "rule": {
            "min_usable_share": D1_MIN_USABLE,
            "max_median_abs_rel_error": D1_MAX_MEDIAN_ERROR,
            "volts": data.VOLTS,
        },
        "passes": bool(usable_share >= D1_MIN_USABLE and median_error <= D1_MAX_MEDIAN_ERROR),
    }


def d1_readings(sample: pd.DataFrame, records: list[dict], table: pd.DataFrame) -> dict:
    """Readings taken after check D1 failed, to say why. They change neither its rule nor result.

    - Sessions whose timestamps were not strictly increasing: whether only repeated times
      (no time going backwards) caused it, and how much of each series was repeated.
    - The ratio of integrated to recorded energy among usable sessions: its spread, the line
      voltage at which the median ratio would be 1, and the error left once that one common
      scale is removed.
    - Peak current and peak pilot signal among usable sessions.
    - The fallback estimate of energy between 10:00 and 20:00 against the series (scaled to
      the recorded total, so only the timing is compared), for usable sessions plugged in
      before 10:00 on a weekday and still plugged in at 10:00 that day.
    """
    by_id = {str(r.get("sessionID")): r for r in records}
    t = table.assign(session_id=table["session_id"].astype(str)).set_index("session_id")

    repeats_only, repeated_share = 0, []
    for sid in t.index[t["reason"] == "not_increasing"]:
        stamps = pd.Series(by_id[sid]["chargingCurrent"]["timestamps"], dtype=object)
        step = np.diff(data.to_hours(pd.to_datetime(stamps, format=data._TIME, utc=True)))
        repeats_only += bool((step >= 0).all())
        repeated_share.append(float((step == 0).mean()))

    usable = t[(t["reason"] == "usable") & (t["kwh_recorded"] > 0)]
    ratio = usable["kwh_integrated"] / usable["kwh_recorded"]
    scale = float(ratio.median())
    rescaled_error = (ratio / scale - 1.0).abs()
    peak_amps = [max(map(float, by_id[s]["chargingCurrent"]["current"])) for s in usable.index]
    pilots = [(by_id[s].get("pilotSignal") or {}).get("pilot") or [] for s in usable.index]
    peak_pilot = [max(map(float, p)) for p in pilots if p]

    s = sample.assign(session_id=sample["session_id"].astype(str)).set_index("session_id")
    s = s.loc[usable.index]
    local = s["connect_time"].dt.tz_convert(DECISION.tz)
    ten = (local.dt.normalize() + pd.Timedelta(hours=10)).dt.tz_convert("UTC")
    present = (local.dt.day_of_week < 5) & (s["connect_time"] < ten) & (s["disconnect_time"] >= ten)
    s, ten_h = s[present], data.to_hours(ten[present])
    by_series = []
    for sid, start in zip(s.index, ten_h, strict=True):
        series, _ = data.parse_series(by_id[sid])
        share = data.energy_kwh(series, start, start + 10.0) / data.energy_kwh(series)
        by_series.append(float(s.at[sid, "kwh_delivered"]) * share)
    by_series = np.array(by_series)
    estimate = energy_in_window(
        data.to_hours(s["connect_time"]),
        data.to_hours(s["done_charging_time"]),
        data.to_hours(s["disconnect_time"]),
        s["kwh_delivered"].to_numpy(dtype=float),
        ten_h,
        ten_h + 10.0,
    )
    gap = np.abs(estimate - by_series)
    some = by_series > 0.5

    return {
        "note": "read after the check failed; the rule and its result are unchanged",
        "not_increasing": {
            "sessions": len(repeated_share),
            "only_repeated_times": repeats_only,
            "median_share_of_samples_repeated": float(np.median(repeated_share))
            if repeated_share
            else None,
        },
        "no_series_recorded_kwh": sorted(
            round(float(v), 2) for v in t.loc[t["reason"] == "no_series", "kwh_recorded"]
        ),
        "ratio_integrated_to_recorded": {
            "sessions": len(ratio),
            "quantiles": {str(q): float(ratio.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "volts_for_median_ratio_1": data.VOLTS / scale,
            "median_abs_rel_error_after_common_scale": float(rescaled_error.median()),
            "share_within_5_percent_after_common_scale": float((rescaled_error <= 0.05).mean()),
        },
        "peak_current_amps_max": max(peak_amps),
        "sessions_peak_current_above_36_amps": int(sum(a > 36.0 for a in peak_amps)),
        "peak_pilot_amps_max": max(peak_pilot) if peak_pilot else None,
        "fallback_10_to_20": {
            "cars": len(s),
            "kwh_by_series_median": float(np.median(by_series)),
            "abs_gap_kwh_median": float(np.median(gap)),
            "abs_gap_kwh_90th_percentile": float(np.quantile(gap, 0.9)),
            "rel_gap_median_where_series_above_half_kwh": float(
                np.median(gap[some] / by_series[some])
            ),
            "kwh_total_by_series": float(by_series.sum()),
            "kwh_total_by_fallback": float(estimate.sum()),
            "cars_above_half_kwh_by_series": int(some.sum()),
            "cars_above_half_kwh_by_fallback": int((estimate > 0.5).sum()),
        },
    }


def charging_done_counts(sessions: pd.DataFrame) -> dict:
    """How often the charging-done time, which D1's fallback spreads energy up to, is unusable."""
    done, start, end = (
        sessions["done_charging_time"],
        sessions["connect_time"],
        sessions["disconnect_time"],
    )
    return {
        "sessions": len(sessions),
        "missing": int(done.isna().sum()),
        "not_after_plug_in": int((done <= start).sum()),
        "after_unplug": int((done > end).sum()),
    }


def run_d1(data_root: Path, out: Path) -> dict:
    """Check D1 on the stored sample: share usable, and integrated against recorded energy."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sessions, _ = data.load_sessions(data_root, "caltech")
    sample = d1_sample(sessions)
    records = data.load_series(data_root, "caltech", D1_NAME)
    if {str(r.get("sessionID")) for r in records} != set(sample["session_id"].astype(str)):
        raise ValueError("the stored D1 series are not the seeded sample")
    table = d1_table(sample, records)
    summary = d1_summary(table)
    table.to_csv(out / "d1_sessions.csv", index=False)
    (out / "d1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passes"]:
        readings = d1_readings(sample, records, table)
        readings["charging_done_in_development"] = charging_done_counts(sessions)
        (out / "d1_readings.json").write_text(json.dumps(readings, indent=2) + "\n")
    manifest = {
        "run": "d1",
        "made_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "code": _git_state(),
        "parameters": {"sample": D1_SAMPLE, "seed": D1_SEED, "volts": data.VOLTS},
        "inputs": {"series": acn.fingerprint(records), "sessions": len(sessions)},
        "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    """The commit the code came from, and whether it had uncommitted changes."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def main(argv: list[str] | None = None) -> int:
    """Command line: ``fleetoffer <command>``."""
    parser = argparse.ArgumentParser(
        prog="fleetoffer",
        description="A fleet charging offer from learned per-car bounds, scored by replay.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo", help="run the pipeline on made-up fleets (a plumbing check, not a result)"
    )
    demo.add_argument("--out", type=Path, default=Path("results/demo"))
    demo.add_argument("--seed", type=int, default=0)

    fetch = commands.add_parser(
        "fetch", help="download one site's sessions from ACN-Data, month by month (needs a token)"
    )
    fetch.add_argument("site", choices=sorted(SITE_MONTHS))
    fetch.add_argument("--data", type=Path, default=Path("data"))
    fetch.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

    verify = commands.add_parser(
        "verify", help="check downloaded months against the committed fingerprints"
    )
    verify.add_argument("site", choices=sorted(SITE_MONTHS))
    verify.add_argument("--series", action="store_true", help="the time series, not the sessions")
    verify.add_argument("--data", type=Path, default=Path("data"))
    verify.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

    series = commands.add_parser(
        "series",
        help="download Caltech's per-session time series for the development period (token)",
    )
    series.add_argument("site", choices=["caltech"])
    series.add_argument("--sample", action="store_true", help="only the seeded sample of check D1")
    series.add_argument("--data", type=Path, default=Path("data"))
    series.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

    d1 = commands.add_parser("d1", help="check D1: are the time series usable for energy?")
    d1.add_argument("--data", type=Path, default=Path("data"))
    d1.add_argument("--out", type=Path, default=Path("results/d1"))

    snap = commands.add_parser(
        "snapshot-check",
        help="compare the downloaded Caltech sessions with the pinned public SustainGym extracts",
    )
    snap.add_argument("snapshots", type=Path, help="folder holding the four pinned CSV files")
    snap.add_argument("--data", type=Path, default=Path("data"))

    commands.add_parser(
        "requirements",
        help="days the 2022 drift theorem needs for a 90%%-of-days offer, beside the days held",
    )

    args = parser.parse_args(argv)
    if args.command == "demo":
        manifest = run_demo(args.out, seed=args.seed)
        print(f"wrote {len(manifest['outputs'])} files and manifest.json to {args.out}")
    elif args.command == "fetch":
        first, last = SITE_MONTHS[args.site]
        table = acn.fetch_site(
            args.site,
            first,
            last,
            data_root=args.data,
            fingerprint_dir=args.fingerprints,
            get=acn.http_getter(acn.read_token()),
        )
        print(f"{args.site}: {int(table['sessions'].sum())} sessions in {len(table)} months")
    elif args.command == "verify" and args.series:
        failed = 0
        for table in sorted(p.stem for p in args.fingerprints.glob(f"{args.site}-series*.csv")):
            report = acn.verify_series(args.data, args.fingerprints, table, args.site)
            failed += int((~report["matches"]).sum())
            print(f"{table}: {int(report['matches'].sum())} of {len(report)} files match")
        return int(failed > 0)
    elif args.command == "verify":
        report = acn.verify_fingerprints(args.data, args.fingerprints, args.site)
        bad = report[~report["matches"]]
        print(f"{args.site}: {len(report) - len(bad)} of {len(report)} months match")
        if len(bad):
            print(bad[["month", "sessions", "found_sessions"]].to_string(index=False))
            return 1
    elif args.command == "series":
        get = acn.http_getter(acn.read_token())
        until = data.TEST_PERIODS[args.site]
        if args.sample:
            sessions, _ = data.load_sessions(args.data, args.site)
            table = acn.fetch_series_sessions(
                args.site,
                d1_sample(sessions),
                D1_NAME,
                data_root=args.data,
                fingerprint_dir=args.fingerprints,
                get=get,
            )
        else:
            sessions, _ = data.load_sessions(args.data, args.site)
            table = acn.fetch_series_site(
                args.site,
                sessions["connect_time"].min().strftime("%Y-%m-%d"),
                until=until,
                data_root=args.data,
                fingerprint_dir=args.fingerprints,
                get=get,
            )
        print(f"{args.site} series: {int(table['sessions'].sum())} sessions stored")
    elif args.command == "d1":
        summary = run_d1(args.data, args.out)
        print(json.dumps(summary, indent=2))
    elif args.command == "snapshot-check":
        sessions, counts = data.load_sessions(args.data, "caltech")
        print("cleaning:", ", ".join(f"{k} {v}" for k, v in counts.items()))
        sealed_from = data.TEST_PERIODS["caltech"]
        print(acn.check_snapshots(args.snapshots, sessions, sealed_from).to_string(index=False))
    elif args.command == "requirements":
        print(calibrate.requirements_table().to_string(index=False))
        print()
        print(calibrate.days_available().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
