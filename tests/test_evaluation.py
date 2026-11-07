import numpy as np
import pandas as pd
import pytest

from fleetoffer import evaluation, resample
from fleetoffer.baselines import Population
from fleetoffer.daylevel import Calibrated
from fleetoffer.engine import fleet_replay
from fleetoffer.features import LOCAL_TZ
from fleetoffer.offer import STEPS, BaseSet
from fleetoffer.quantiles import PerCarRule
from fleetoffer.synthetic import Regime, make_sessions, with_app_entries


def no_flags(dates):
    n = len(dates)
    return pd.DataFrame({"in_term": np.zeros(n), "summer": np.zeros(n), "holiday": np.zeros(n)})


def calibrated():
    return Calibrated(PerCarRule([Population()], min_rows=50, calendar=no_flags), gamma=0.05)


@pytest.fixture(scope="module")
def claimed():
    sessions = make_sessions(
        [Regime("2001-01-01", sessions_per_weekday=12, median_stay_h=6.0, drivers=40)],
        "2001-05-20",
        seed=3,
    )
    return with_app_entries(sessions, seed=3)


def test_a_frozen_rule_stops_refitting_scoring_and_moving_its_level(claimed):
    sessions, entries = claimed
    when = pd.Timestamp("2001-04-01", tz=LOCAL_TZ)
    rule = calibrated()
    frozen = evaluation.Frozen(rule, when)
    fleet_replay(sessions, frozen, entries=entries, method="bounds")
    assert frozen.is_frozen
    origins = pd.to_datetime([row["origin"] for row in rule.per_car.fit_log])
    assert (origins < when).all() and len(origins) >= 3
    observed = pd.to_datetime([row["decision_time"] for row in rule.log])
    assert len(observed) > 20
    later = [when + pd.Timedelta(days=d) for d in (1, 20, 45)]
    assert len({rule.level(t) for t in later}) == 1
    assert rule.frozen_at == when


def test_the_updating_jpl_track_forgets_caltech_days_and_keeps_its_fit(claimed):
    sessions, entries = claimed
    rule = calibrated()
    fleet_replay(sessions, rule, entries=entries, method="bounds")
    assert rule.scores and rule.per_car.fitted["population"]
    moved = evaluation.for_jpl(rule, "updating")
    assert moved.scores == [] and moved.per_car.cardays().empty
    other = make_sessions(
        [Regime("2003-01-01", sessions_per_weekday=12, median_stay_h=4.0, drivers=30)],
        "2003-03-10",
        seed=9,
    )
    other, other_entries = with_app_entries(other, seed=9)
    fleet_replay(other, moved, entries=other_entries, method="bounds")
    first = [row for row in moved.per_car.fit_log if row["origin"].year == 2003][0]
    assert first["kept_earlier_fit"] and not first["fitted"]
    assert len(moved.scores) > 10


def test_the_frozen_jpl_track_hides_history_and_learns_nothing(claimed):
    sessions, entries = claimed
    rule = calibrated()
    fleet_replay(sessions, rule, entries=entries, method="bounds")
    scores = list(rule.scores)
    frozen = evaluation.for_jpl(rule, "frozen")
    assert rule.per_car.frozen and rule.per_car.hide_history
    other = make_sessions(
        [Regime("2003-01-01", sessions_per_weekday=12, median_stay_h=4.0, drivers=30)],
        "2003-02-20",
        seed=9,
    )
    other, other_entries = with_app_entries(other, seed=9)
    fleet_replay(other, frozen, entries=other_entries, method="bounds")
    assert rule.scores == scores


def weekdays(start, n, cars):
    times = pd.bdate_range(start, periods=n, tz=LOCAL_TZ) + pd.Timedelta(hours=10)
    return pd.DataFrame({"decision_time": times.tz_convert("UTC"), "cars": cars})


def test_the_reopening_is_the_first_2021_weekday_whose_ten_day_average_reaches_five():
    cars = [0] * 15 + [10] * 20
    days = weekdays("2021-01-01", len(cars), cars)
    # the trailing average first reaches 5 on the fifth weekday with 10 cars
    expected = days["decision_time"].iloc[19].tz_convert(LOCAL_TZ).date()
    assert evaluation.reopening(days) == str(expected)
    assert evaluation.reopening(weekdays("2021-01-01", 30, [3] * 30)) == ""


def test_the_reopening_window_reaches_back_into_the_year_before():
    # ten busy weekdays at the end of 2020: the first weekday of 2021 already qualifies
    days = weekdays("2020-12-17", 12, [10] * 12)
    first_2021 = days["decision_time"].dt.tz_convert(LOCAL_TZ)
    first_2021 = first_2021[first_2021.dt.year == 2021].iloc[0].date()
    assert evaluation.reopening(days) == str(first_2021)
    # a day of 2020 never counts as the reopening, however busy
    assert evaluation.reopening(weekdays("2020-12-01", 30, [10] * 30)) == "2021-01-01"


def h3_days(adaptive, frozen):
    n = len(adaptive)
    base = weekdays("2021-03-01", n, [20] * n).assign(
        size_kwh=1.0, cars_with_energy=10, exact_true_kwh=5.0
    )
    rows = [
        base.assign(arm="4", fixed_delivered=adaptive),
        base.assign(arm="4f", fixed_delivered=frozen),
    ]
    return pd.concat(rows, ignore_index=True)


def test_h3_reads_the_frozen_and_adaptive_arms():
    adaptive = [False] * 10 + [True] * 40
    frozen = [True] * 10 + [False] * 40
    out = evaluation.h3(h3_days(adaptive, frozen), "2021-03-01", "4", "4f")
    assert out["frozen_first20_delivered"] == pytest.approx(0.5)
    assert out["h3a"] == "pass"
    assert out["adaptive_reached_at_scored_weekday"] == 28  # 18 of the last 20 delivered
    assert out["h3b"] == "pass"
    late = [False] * 30 + [True] * 20
    assert evaluation.h3(h3_days(late, frozen), "2021-03-01", "4", "4f")["h3b"] == "fail"


def test_h1_with_both_arms_at_zero_is_not_testable():
    days = h3_days([True] * 30, [True] * 30).assign(size_kwh=0.0)
    days["arm"] = days["arm"].map({"4": "4", "4f": "5"})
    assert evaluation.paired(days, "4", "5")["reading"] == "not testable: both arms offer zero"


def test_the_base_set_is_written_and_read_back_exactly(tmp_path):
    base = BaseSet(np.linspace(5.0, 0.2, STEPS), 5.774)
    evaluation.write_base_set(base, tmp_path / "h0.json")
    back = evaluation.base_set(tmp_path / "h0.json")
    assert np.array_equal(back.power, base.power) and back.energy == base.energy


def test_every_period_runs_the_registered_and_secondary_arms_and_the_sweep():
    for period in evaluation.PERIODS:
        jobs = evaluation.jobs(period, "data", False)
        kinds = {job[1] for job in jobs}
        assert "sweep" in kinds
        names = {job[3] for job in jobs if job[1] != "sweep"}
        assert set(evaluation.REGISTERED) | set(evaluation.SECONDARY) <= names
    for period in ("sealed", "rehearsal-sealed"):
        sealed = {job[3] for job in evaluation.jobs(period, "data", False) if job[1] == "arm"}
        assert set(evaluation.FROZEN_ARMS) <= sealed
    jpl = evaluation.jobs("jpl", "data", False)
    assert ("data", "jpl", "jpl", "1", "segment", "updating", False) not in jpl
    for period in evaluation.PERIODS:
        rehearsal = f"rehearsal-{evaluation.stage(period)}"
        assert len(evaluation.jobs(period, "d", False)) == len(
            evaluation.jobs(rehearsal, "d", False)
        )


def test_the_committed_base_set_is_the_one_printed_in_the_registration_page():
    import re
    from pathlib import Path

    base = evaluation.base_set()
    page = (Path(__file__).parents[1] / "docs" / "registration.md").read_text()
    table = page[page.index("**h₀") :]
    printed = [float(x) for x in re.findall(r"\| (\d+\.\d{3}) ", table)]
    assert len(printed) == STEPS
    assert np.allclose(np.round(base.power, 3), printed)
    assert f"{base.energy:.3f} kWh" in page


def test_the_sensitivities_change_one_setting_on_the_caltech_runs_only():
    for period in ("evaluation", "sealed"):
        jobs = evaluation.jobs(period, "d", False)
        runs = [job for job in jobs if job[1] == "arm" and len(job) == 7]
        assert {job[6] for job in runs} == set(evaluation.SENSITIVITIES)
        assert len(runs) == len(evaluation.SENSITIVITIES) * len(evaluation.SENSITIVITY_ARMS)
    settings = set(evaluation.SENSITIVITIES)
    assert not any(settings & set(job) for job in evaluation.jobs("jpl", "d", False))
    base = BaseSet(np.linspace(5.0, 0.2, STEPS), 5.774)
    assert evaluation.make_rule("4d", "average", base, cap=11.0).cap == 11.0
    assert evaluation.make_rule("5d", "average", base).cap == 6.7


def test_the_oracle_run_in_pieces_gives_the_days_of_one_whole_run(claimed, monkeypatch):
    sessions, entries = claimed
    monkeypatch.setattr(evaluation, "load", lambda *args, **kwargs: (sessions, entries))
    edges = ["2000-01-01", "2001-02-15", "2001-04-01", "2100-01-01"]
    monkeypatch.setitem(evaluation.ORACLE_PIECES, "rehearsal-jpl", edges)
    whole = evaluation.run_jpl("d", "rehearsal-jpl", "1", "segment", "frozen")
    parts = [
        evaluation.run_jpl("d", "rehearsal-jpl", "1", "segment", "frozen", piece=k)
        for k in range(len(edges) - 1)
    ]
    for key in ("days", "cars"):
        joined = pd.concat([p[key] for p in parts], ignore_index=True)
        expected = whole[key]
        assert len(joined) == len(expected) > 50
        drop = [c for c in ("solve_seconds",) if c in expected.columns]
        pd.testing.assert_frame_equal(
            joined.drop(columns=drop), expected.drop(columns=drop), check_dtype=False
        )
    with pytest.raises(ValueError):
        evaluation.run_jpl("d", "rehearsal-jpl", "5", "segment", "frozen", piece=0)
    jobs = evaluation.jobs("jpl", "d", True)
    assert sum(job[3] == "1a" for job in jobs) == len(evaluation.ORACLE_PIECES["jpl"]) - 1


def test_the_0900_sensitivity_decides_at_nine_and_learns_from_nine_oclock_days(claimed):
    sessions, entries = claimed
    rule = calibrated()
    result = fleet_replay(
        sessions, rule, entries=entries, method="segment", **evaluation.SENSITIVITIES["at0900"]
    )
    local = result.days["decision_time"].dt.tz_convert(LOCAL_TZ)
    assert (local.dt.hour == 9).all() and (local.dt.minute == 0).all()
    offered = result.days["size_kwh"].notna()
    assert offered.sum() > 20
    assert len(rule.scores) > 20 and rule.per_car.fitted["population"]


def test_the_monthly_delivery_rate_comes_with_a_week_block_interval():
    days = weekdays("2021-03-01", 40, [10] * 40).assign(
        arm="4", size_kwh=1.0, fixed_delivered=[True, False, True, True] * 10
    )
    table = evaluation.monthly(days).set_index("month")
    for _, row in table.iterrows():
        assert row["fixed_delivered_low"] <= row["fixed_delivered"] <= row["fixed_delivered_high"]
    assert table["days"].sum() == 40
    full = evaluation.monthly(days.assign(fixed_delivered=True))
    assert (full["fixed_delivered_low"] == 1).all() and (full["fixed_delivered_high"] == 1).all()


def test_the_jpl_sweep_with_the_empty_cars_known_trains_and_runs(monkeypatch):
    from fleetoffer import percar

    sessions = make_sessions(
        [Regime("2019-01-07", sessions_per_weekday=12, median_stay_h=6.0, drivers=40)],
        "2019-05-20",
        seed=4,
    )
    sessions, entries = with_app_entries(sessions, seed=4)
    monkeypatch.setattr(evaluation, "load", lambda *args, **kwargs: (sessions, entries))
    monkeypatch.setattr(percar, "load", lambda *args, **kwargs: (sessions, entries))
    out = evaluation.run_sweep("d", "rehearsal-jpl", "5k", "segment")
    assert len(out["sweep"]) > 20 and (out["sweep"]["dropped_cars"] >= 0).all()


def test_h1_reads_its_interval_on_the_kwh_scale_when_the_second_arm_is_often_zero():
    n = 40
    population = [0.0] * n
    population[3] = 0.5  # one positive day: most resamples of weeks have a zero total
    days = h3_days([True] * n, [True] * n)
    days["arm"] = days["arm"].map({"4": "4", "4f": "5"})
    days.loc[days["arm"] == "4", "size_kwh"] = 2.0
    days.loc[days["arm"] == "5", "size_kwh"] = population
    out = evaluation.paired(days, "4", "5")
    assert 0 < out["relative"]["draws_defined"] < resample.DRAWS
    assert out["absolute_kwh"]["low"] > 0
    assert out["reading"] == "pass"
