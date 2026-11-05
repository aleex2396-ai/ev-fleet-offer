import json

import numpy as np
import pandas as pd
import pytest

from fleetoffer.pipeline import (
    D1_SAMPLE,
    charging_done_counts,
    d1_readings,
    d1_sample,
    d1_summary,
    d1_table,
    main,
    monthly_summary,
    overall_summary,
    run_demo,
)
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
        "fleet_offers.csv",
        "monthly.png",
    ]
    for name in outputs:
        assert (tmp_path / name).stat().st_size > 0
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["parameters"]["seed"] == 0

    # The oracle's offer, built from true sets, is delivered every day and within the band.
    check = on_disk["fleet_check"]
    assert check["days"] >= 9
    assert check["delivered_share_fixed"] == check["delivered_share_hindsight"] == 1.0
    assert 0 < check["median_size_over_exact"] <= 1.0
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


def test_d1_sample_is_seeded_and_from_the_given_sessions():
    sessions = pd.DataFrame(
        {
            "session_id": [f"s{k}" for k in range(1000)],
            "connect_time": pd.date_range("2019-01-01", periods=1000, freq="h", tz="UTC"),
        }
    )
    a, b = d1_sample(sessions), d1_sample(sessions.sample(frac=1.0, random_state=3))
    assert len(a) == D1_SAMPLE and a["session_id"].is_unique
    assert list(a["session_id"]) == list(b["session_id"])  # input order does not matter
    assert a["connect_time"].is_monotonic_increasing


def test_d1_summary_applies_the_rule():
    rows = [("usable", 10.0, 10.2)] * 96 + [("too_short", 5.0, np.nan)] * 4
    table = pd.DataFrame(rows, columns=["reason", "kwh_recorded", "kwh_integrated"])
    table["negative_samples"] = 0
    table["abs_rel_error"] = (table["kwh_integrated"] - table["kwh_recorded"]).abs() / 10.0
    summary = d1_summary(table)
    assert summary["usable_share"] == pytest.approx(0.96)
    assert summary["median_abs_rel_error"] == pytest.approx(0.02)
    assert summary["passes"]
    table.loc[:10, "reason"] = "no_series"
    assert not d1_summary(table)["passes"]  # 85 of 100 usable


def test_d1_table_compares_integrated_with_recorded_energy():
    stamps = ["Mon, 03 Jun 2019 15:00:00 GMT", "Mon, 03 Jun 2019 16:00:00 GMT"]
    records = [
        {"sessionID": "a", "chargingCurrent": {"timestamps": stamps, "current": [10.0, 0.0]}},
        {"sessionID": "b"},
    ]
    sessions = pd.DataFrame({"session_id": ["a", "b"], "kwh_delivered": [2.0, 3.0]})
    table = d1_table(sessions, records).set_index("session_id")
    assert table.loc["a", "kwh_integrated"] == pytest.approx(2.08)  # 10 A, 1 h, 208 V
    assert table.loc["a", "abs_rel_error"] == pytest.approx(0.04)
    assert table.loc["b", "reason"] == "no_series" and np.isnan(table.loc["b", "abs_rel_error"])


def test_d1_readings_explain_a_failure_without_changing_it():
    def stamps(*hours):
        return [f"Mon, 03 Jun 2019 {h:02d}:00:00 GMT" for h in hours]

    records = [
        {  # plugged in 08:00 Pacific, present at 10:00; front-loaded charging
            "sessionID": "a",
            "chargingCurrent": {
                "timestamps": stamps(15, 16, 17, 18, 19),
                "current": [10, 10, 5, 5, 0],
            },
            "pilotSignal": {"timestamps": stamps(15, 19), "pilot": [32, 32]},
        },
        {  # a repeated timestamp
            "sessionID": "b",
            "chargingCurrent": {"timestamps": stamps(15, 15, 16), "current": [1, 1, 0]},
        },
        {"sessionID": "c"},
        {  # plugged in 11:00 Pacific, after the decision
            "sessionID": "d",
            "chargingCurrent": {"timestamps": stamps(18, 19, 20), "current": [20, 20, 0]},
        },
    ]
    utc = lambda h: pd.Timestamp("2019-06-03", tz="UTC") + pd.Timedelta(hours=h)  # noqa: E731
    sample = pd.DataFrame(
        {
            "session_id": ["a", "b", "c", "d"],
            "connect_time": [utc(15), utc(15), utc(15), utc(18)],
            "done_charging_time": [utc(19), utc(16), utc(16), utc(20)],
            "disconnect_time": [utc(25), utc(17), utc(17), utc(21)],
            "kwh_delivered": [10.0, 1.0, 3.0, 8.0],
        }
    )
    table = d1_table(sample, records)
    readings = d1_readings(sample, records, table)
    assert readings["not_increasing"] == {
        "sessions": 1,
        "only_repeated_times": 1,
        "median_share_of_samples_repeated": 0.5,
    }
    assert readings["no_series_recorded_kwh"] == [3.0]
    ratio = readings["ratio_integrated_to_recorded"]
    assert ratio["quantiles"]["0.5"] == pytest.approx((0.624 + 1.04) / 2)
    assert ratio["median_abs_rel_error_after_common_scale"] == pytest.approx(0.25)
    assert readings["peak_current_amps_max"] == 20.0
    assert readings["peak_pilot_amps_max"] == 32.0
    fallback = readings["fallback_10_to_20"]
    assert fallback["cars"] == 1  # only "a" is present at 10:00
    assert fallback["kwh_total_by_series"] == pytest.approx(10.0 / 3.0)  # 10 of 30 amp-hours
    assert fallback["kwh_total_by_fallback"] == pytest.approx(5.0)  # 2 of 4 hours to done
    assert d1_summary(table)["passes"] is False


def test_charging_done_counts():
    t = pd.Timestamp("2019-06-03", tz="UTC")
    h = pd.Timedelta(hours=1)
    sessions = pd.DataFrame(
        {
            "connect_time": [t, t, t, t],
            "done_charging_time": [t + h, pd.NaT, t, t + 3 * h],
            "disconnect_time": [t + 2 * h] * 4,
        }
    )
    assert charging_done_counts(sessions) == {
        "sessions": 4,
        "missing": 1,
        "not_after_plug_in": 1,
        "after_unplug": 1,
    }
