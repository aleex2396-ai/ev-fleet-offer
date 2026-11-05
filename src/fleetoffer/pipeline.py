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

import pandas as pd

from fleetoffer import __version__, acn, calibrate, data
from fleetoffer.engine import Daily, replay
from fleetoffer.figures import plot_monthly
from fleetoffer.models import FrozenQuantile, RollingQuantile
from fleetoffer.synthetic import Regime, demo_regimes, make_sessions

DEMO_TZ = "America/Los_Angeles"
DEMO_LEVEL = 0.1
DECISION = Daily(at="10:00", tz=DEMO_TZ)

# The months each site has sessions in the public API (checked 4 Nov 2026).
SITE_MONTHS = {"caltech": ("2018-04", "2021-09"), "jpl": ("2018-09", "2021-09")}


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
    verify.add_argument("--data", type=Path, default=Path("data"))
    verify.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

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
    elif args.command == "verify":
        report = acn.verify_fingerprints(args.data, args.fingerprints, args.site)
        bad = report[~report["matches"]]
        print(f"{args.site}: {len(report) - len(bad)} of {len(report)} months match")
        if len(bad):
            print(bad[["month", "sessions", "found_sessions"]].to_string(index=False))
            return 1
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
