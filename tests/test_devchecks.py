import numpy as np
import pandas as pd
import pytest

from fleetoffer import devchecks
from fleetoffer.offer import BaseSet
from fleetoffer.resample import paired_interval, week_of

PT = "America/Los_Angeles"


def at(date: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date} 10:00", tz=PT).tz_convert("UTC")


def test_d3_reads_the_median_and_tenth_percentile():
    counts = pd.DataFrame(
        {
            "decision_time": [at("2019-05-01")] * 10,
            "cars": 20,
            "cars_with_energy": [3, 4, 9, 10, 10, 11, 12, 13, 14, 15],
            "period": "model selection",
        }
    )
    reading = devchecks.d3_reading(counts)
    assert reading["median"] == 10.5 and reading["below_5"] == 2
    assert reading["p10"] == pytest.approx(3.9) and not reading["passes"]


COLUMNS = ["decision_time", "session_id", "linked", "n_past", "typed_kwh", "target_energy_kwh"]


def cardays(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_d4_counts_unclaimed_cars_with_nothing_left_to_take():
    frame = cardays(
        [
            (at("2019-05-01"), "a", 0, np.nan, np.nan, 0.0),
            (at("2019-05-01"), "b", 1, 3, 10.0, 0.0),
            (at("2019-05-01"), "c", 0, np.nan, np.nan, 4.0),
            (at("2019-05-01"), "d", 1, 1, 5.0, 2.0),
        ]
    )
    reading = devchecks.d4(frame)
    assert reading["unclaimed_zero_share"] == 0.25 and not reading["rule_applies"]


def test_d5_counts_weekdays_with_enough_typed_requests_among_cars_with_energy():
    rows = []
    for k, date in enumerate(["2019-05-01", "2019-05-02"]):
        typed = [10.0, np.nan, np.nan, np.nan] if k == 0 else [10.0, 8.0, np.nan, np.nan]
        for i, kwh in enumerate(typed):
            rows.append((at(date), f"{date}{i}", 1, 0, kwh, 3.0))
        rows.append((at(date), f"{date}x", 1, 0, 9.0, 0.2))  # no energy: not counted
    reading = devchecks.d5(cardays(rows))
    assert reading["weekdays_with_cars"] == 2 and reading["weekdays_at_30_percent"] == 1


def test_d6_needs_a_link_and_three_earlier_sessions():
    frame = cardays(
        [
            (at("2019-05-01"), "a", 1, 3, np.nan, 1.0),
            (at("2019-05-01"), "b", 1, 2, np.nan, 1.0),
            (at("2019-05-01"), "c", 0, np.nan, np.nan, 1.0),
            (at("2019-06-03"), "d", 1, 5, np.nan, 1.0),
        ]
    )
    table = devchecks.d6(frame).set_index("month")
    assert table.loc["2019-05", "linked"] == pytest.approx(2 / 3)
    assert table.loc["2019-05", "history"] == pytest.approx(1 / 3)
    assert table.loc["2019-06", "history"] == 1.0


def test_d7_uses_each_session_once():
    frame = cardays(
        [
            (at("2019-05-01"), "a", 1, 0, 10.0, 1.0),
            (at("2019-05-02"), "a", 1, 0, 20.0, 1.0),  # the same session the next morning
            (at("2019-05-01"), "b", 1, 0, 10.0, 1.0),
            (at("2019-05-01"), "c", 1, 0, 30.0, 1.0),
        ]
    )
    sessions = pd.DataFrame({"session_id": ["a", "b", "c"], "kwh_delivered": [9.0, 12.0, 27.0]})
    reading = devchecks.d7(frame, sessions)
    assert reading["sessions"] == 3 and reading["median_ratio"] == pytest.approx(0.9)
    assert reading["adequate_proxy"]  # ranks agree: 9 < 12 < 27 against 10 = 10 < 30


def test_d12_on_a_base_set():
    base = BaseSet(np.array([6.7, 3.0, 0.0]), 5.0)
    reading = devchecks.d12(base)
    assert reading["non_empty_with_width"] and reading["steps_with_power"] == 2


def days_table(rule, alphas, sizes, start="2019-05-01"):
    times = [at(d.strftime("%Y-%m-%d")) for d in pd.bdate_range(start, periods=len(alphas))]
    return pd.DataFrame(
        {
            "rule": rule,
            "decision_time": times,
            "cars_with_energy": 8,
            "alpha": alphas,
            "alpha_day_average": alphas,
            "size_kwh": sizes,
            "size_day_average_kwh": sizes,
            "solve_seconds": 1.0,
            "fixed_delivered": True,
            "hindsight_delivered": True,
            "all_bounds_held": True,
        }
    )


def test_d9_switches_on_too_many_small_oracle_offers():
    oracle = days_table("1 oracle", [0.1, 0.1, 0.5, 0.5], [1.0, 1.0, 5.0, 5.0])
    reading = devchecks.d9(oracle, oracle)
    assert reading["oracle_alpha_below_0_2_share"] == 0.5 and reading["fallback"]


def test_selection_days_drop_warm_up_and_unscored_days():
    days = days_table("x", [1.0] * 3, [2.0, np.nan, 2.0], start="2019-04-30")
    days.loc[2, "cars_with_energy"] = 4
    assert len(devchecks.selection_days(days)) == 0  # 30 Apr: warm-up; 1 May: no offer; 2 May: 4


def test_week_blocks_and_a_paired_interval():
    times = pd.Series([at("2019-05-06"), at("2019-05-10"), at("2019-05-13"), at("2019-05-27")])
    assert week_of(times).tolist() == [0, 0, 1, 3]
    assert week_of(times, weeks=2).tolist() == [0, 0, 0, 1]
    days = pd.DataFrame({"decision_time": times, "a": [2.0, 2.0, 2.0, 2.0], "b": [1.0] * 4})
    reading = paired_interval(days, "a", "b", draws=200)
    assert reading["estimate"] == 1.0 and reading["low"] == reading["high"] == 1.0
    assert paired_interval(days, "a", "b", relative=True, draws=50)["estimate"] == 1.0


def test_feedback_delay_counts_days_known_before_the_next_decision():
    times = [at(d) for d in ("2019-05-01", "2019-05-02", "2019-05-03", "2019-05-06")]
    log = pd.DataFrame(
        {
            "decision_time": times,
            "fitted": True,
            "scored": [True, True, False, True],
            # day 1 known that evening; day 2 only after the weekend; day 4 is the last
            "revealed_at": [
                times[0] + pd.Timedelta(hours=10),
                times[3] + pd.Timedelta(hours=1),
                times[2] + pd.Timedelta(hours=10),
                times[3] + pd.Timedelta(hours=10),
            ],
        }
    )
    reading = devchecks.feedback_delay(log)
    assert reading["scored_days"] == 2 and reading["known_before_next"] == 0.5
    assert reading["most_pending"] == 1  # at the decisions of 3 and 6 May, day 2 is pending
