"""End-to-end runs. Each run writes its tables, figures and a manifest saying how it was made."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from staypromise import __version__
from staypromise.engine import replay
from staypromise.figures import plot_monthly
from staypromise.rules import FrozenQuantile, RollingQuantile
from staypromise.scoring import monthly_summary, overall_summary
from staypromise.synthetic import Regime, demo_regimes, make_sessions

DEMO_TZ = "America/Los_Angeles"
DEMO_LEVEL = 0.1


def run_demo(
    out: Path,
    *,
    seed: int = 0,
    regimes: list[Regime] | None = None,
    end: str = "2003-10-01",
    fit_until: str = "2002-01-01",
) -> dict:
    """Run the whole pipeline on made-up sessions. This checks the plumbing; it is not a result."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = make_sessions(regimes or demo_regimes(), end, tz=DEMO_TZ, seed=seed)

    rules = [
        FrozenQuantile(pd.Timestamp(fit_until, tz=DEMO_TZ), level=DEMO_LEVEL),
        RollingQuantile(30, level=DEMO_LEVEL),
        RollingQuantile(120, level=DEMO_LEVEL),
    ]
    decisions = pd.concat([replay(sessions, rule) for rule in rules], ignore_index=True)
    monthly = monthly_summary(decisions, tz=DEMO_TZ)
    overall = overall_summary(decisions)

    sessions.to_csv(out / "sessions.csv", index=False)
    decisions.to_csv(out / "decisions.csv.gz", index=False)
    monthly.to_csv(out / "monthly.csv", index=False)
    overall.to_csv(out / "overall.csv", index=False)
    figure = plot_monthly(
        monthly,
        DEMO_LEVEL,
        out / "monthly.png",
        rules=[rule.name for rule in rules],
        title="Made-up data: pipeline check, not a result",
        note=(
            f"Synthetic sessions (seed {seed}): a stable year, a near-empty half year, then a new "
            "population with shorter stays. Months with under 20 promises are left blank."
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
            "time_zone": DEMO_TZ,
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
