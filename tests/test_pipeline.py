import json

import numpy as np
import pandas as pd
import pytest

from fleetoffer.pipeline import main, monthly_summary, overall_summary, run_demo
from fleetoffer.synthetic import Regime

SHORT = [
    Regime("2001-01-01", sessions_per_weekday=20, median_stay_h=6.0),
    Regime("2001-03-01", sessions_per_weekday=20, median_stay_h=2.0, returning_share=0.3),
]


def test_demo_writes_every_output_and_a_manifest(tmp_path):
    manifest = run_demo(tmp_path, regimes=SHORT, end="2001-05-01", fit_until="2001-02-01")
    outputs = [
        "sessions.csv",
        "decisions.csv.gz",
        "monthly.csv",
        "overall.csv",
        "fleet_days.csv",
        "monthly.png",
    ]
    for name in outputs:
        assert (tmp_path / name).stat().st_size > 0
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["parameters"]["seed"] == 0
    assert on_disk["inputs"]["sessions"] == manifest["inputs"]["sessions"] > 0

    # After the change, the frozen rule's bounds hold far less often than a rolling one's.
    monthly = pd.read_csv(tmp_path / "monthly.csv")
    april = monthly[monthly["month"] == "2001-04"].set_index("rule")["held_share"]
    assert april["frozen"] < 0.6 < 0.8 < april["rolling 30 d"]

    # One row per weekday with cars present at 10:00.
    days = pd.read_csv(tmp_path / "fleet_days.csv")
    assert days["cars_present"].min() >= 1
    assert pd.to_datetime(days["day"]).dt.day_of_week.max() < 5
    assert days["day"].is_unique


def test_command_line(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    assert "manifest.json" in capsys.readouterr().out


def decisions():
    times = pd.to_datetime(
        ["2001-01-05 12:00", "2001-01-06 12:00", "2001-01-07 12:00", "2001-02-01 03:00"], utc=True
    )
    return pd.DataFrame(
        {
            "rule": "r",
            "decision_time": times,
            "given": [True, True, False, True],
            "bound_h": [2.0, 3.0, np.nan, 1.0],
            "held": pd.array([True, False, pd.NA, True], dtype="boolean"),
        }
    )


def test_monthly_scores():
    jan = monthly_summary(decisions()).iloc[0]
    assert (jan["decisions"], jan["given"], jan["held"]) == (3, 2, 1)
    assert jan["held_share"] == pytest.approx(0.5)  # of the bounds given
    assert jan["mean_bound_h"] == pytest.approx(5.0 / 3.0)  # no bound counts as 0 h
    assert jan["given_share"] == pytest.approx(2.0 / 3.0)


def test_months_follow_the_site_time_zone():
    # 1 Feb 03:00 UTC is still 31 Jan in California.
    by_month = monthly_summary(decisions(), tz="America/Los_Angeles")
    assert list(by_month["month"].astype(str)) == ["2001-01"]
    assert by_month.iloc[0]["decisions"] == 4


def test_overall_scores():
    row = overall_summary(decisions()).iloc[0]
    assert row["rule"] == "r"
    assert row["held_share"] == pytest.approx(2.0 / 3.0)
    assert overall_summary(decisions().drop(columns="rule")).shape[0] == 1
