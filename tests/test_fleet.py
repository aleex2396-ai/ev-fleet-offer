import numpy as np
import pandas as pd
import pytest
from helpers import FleetOracle, FleetPast, FleetPeekEnergy

from fleetoffer.checks import LookaheadError, check_no_lookahead_days
from fleetoffer.data import to_hours
from fleetoffer.engine import MORNING, Oracle, energy_in_window, fleet_replay, training_base_set
from fleetoffer.interface import FleetRule
from fleetoffer.offer import BaseSet
from fleetoffer.synthetic import Regime, make_sessions

REGIMES = [Regime("2001-01-01", sessions_per_weekday=9, median_stay_h=6.0)]


@pytest.fixture(scope="module")
def sessions():
    return make_sessions(REGIMES, "2001-01-27", seed=3)


def test_made_up_energy_is_charged_at_a_constant_rate_so_the_estimate_is_exact(sessions):
    hours = (sessions["done_charging_time"] - sessions["connect_time"]) / pd.Timedelta(hours=1)
    rate = sessions["kwh_delivered"] / hours
    assert rate.between(3.0 - 1e-9, 6.6 + 1e-9).all()
    assert (sessions["done_charging_time"] <= sessions["disconnect_time"]).all()
    s = sessions.iloc[:50]
    t = to_hours(s["connect_time"]) + 0.5  # half an hour after plug-in
    got = energy_in_window(
        to_hours(s["connect_time"]),
        to_hours(s["done_charging_time"]),
        to_hours(s["disconnect_time"]),
        s["kwh_delivered"].to_numpy(),
        t,
        t + 10.0,
    )
    exact = rate.to_numpy()[:50] * np.clip(hours.to_numpy()[:50] - 0.5, 0, 10)
    assert got == pytest.approx(exact)


def test_true_bounds_are_delivered_every_day(sessions):
    result = fleet_replay(sessions, FleetOracle(sessions))
    scored = result.days[result.days["cars"] > 0]
    assert len(scored) >= 15
    assert scored["all_bounds_held"].all()
    assert scored["fixed_delivered"].all() and scored["hindsight_delivered"].all()
    assert (scored["size_kwh"] <= scored["exact_true_kwh"] + 1e-6).all()
    assert (result.cars["learned_deadline"] == result.cars["true_deadline"]).all()


def test_bounds_of_zero_offer_nothing_and_that_is_delivered(sessions):
    class Nothing(FleetRule):
        name = "nothing"

        def bounds(self, day):
            n = len(day.present)
            return np.zeros(n), np.zeros(n)

    days = fleet_replay(sessions, Nothing()).days
    scored = days[days["cars"] > 0]
    assert (scored["size_kwh"] == 0).all() and scored["fixed_delivered"].all()


def test_a_day_is_revealed_only_after_its_horizon_and_its_last_car_have_gone(sessions):
    rule = FleetPast()
    result = fleet_replay(sessions, rule)
    days = result.days.set_index("decision_time")
    last_unplug = (
        result.cars.merge(sessions[["session_id", "disconnect_time"]], on="session_id")
        .groupby("decision_time")["disconnect_time"]
        .max()
    )
    for when, revealed in days["revealed_at"].dropna().items():
        assert revealed >= when + pd.Timedelta(hours=10)
        assert revealed >= last_unplug[when]
    for decision_time, seen in rule.seen:
        assert all(o.revealed_at < decision_time for o in seen)
    assert any(len(seen) for _, seen in rule.seen)


def test_cars_present_are_shown_without_anything_recorded_later(sessions):
    captured = []

    class Look(FleetRule):
        def bounds(self, day):
            captured.append(day)
            n = len(day.present)
            return np.zeros(n), np.zeros(n)

    fleet_replay(sessions, Look())
    day = next(d for d in captured if len(d.present))
    assert set(day.present.columns) == {
        "session_id",
        "connect_time",
        "station_id",
        "user_id",
        "elapsed_h",
    }
    assert (day.ended["disconnect_time"] < day.decision_time).all()
    linked_late = sessions.assign(linked_time=sessions["disconnect_time"])
    captured.clear()
    fleet_replay(linked_late, Look())
    assert all(d.present["user_id"].isna().all() for d in captured)


def test_the_training_base_set_averages_the_true_sets(sessions):
    until = pd.Timestamp("2001-01-13", tz="UTC")
    base = training_base_set(sessions, until)
    oracle = fleet_replay(sessions[sessions["connect_time"] < until], FleetOracle(sessions))
    early = oracle.cars[oracle.cars["decision_time"] + pd.Timedelta(hours=10) <= until]
    assert base.energy == pytest.approx(early["true_energy_kwh"].mean())
    limit = np.minimum(6.7, early["true_energy_kwh"] / 0.25)
    power = [(limit * (early["true_deadline"] > k)).mean() for k in range(40)]
    assert base.power == pytest.approx(np.array(power))


def test_the_day_level_scramble_passes_an_honest_rule_through_the_programme(sessions):
    base = training_base_set(sessions, pd.Timestamp("2001-01-10", tz="UTC"))
    compared = check_no_lookahead_days(lambda table: FleetPast(), sessions, base=base)
    assert compared > 10


def test_the_day_level_scramble_catches_a_rule_that_reads_later_energy(sessions):
    with pytest.raises(LookaheadError):
        check_no_lookahead_days(FleetPeekEnergy, sessions)


def test_the_day_level_scramble_catches_a_rule_that_reads_departures(sessions):
    with pytest.raises(LookaheadError):
        check_no_lookahead_days(FleetOracle, sessions)


def test_the_fallback_offer_runs_through_the_day_engine(sessions):
    days = fleet_replay(sessions, FleetOracle(sessions), method="segment").days
    scored = days[days["cars"] > 0]
    assert scored["fixed_delivered"].all() and (scored["size_kwh"] >= 0).all()


def test_a_fixed_base_set_is_used_as_given(sessions):
    base = BaseSet(np.full(40, 1.0), 5.0)
    days = fleet_replay(sessions, FleetOracle(sessions), base=base).days
    scored = days[days["cars"] > 0]
    widths = [base.width(t) for t in (16, 24, 32)]
    assert scored["size_kwh"].to_numpy() == pytest.approx(
        scored["alpha"].to_numpy() * np.mean(widths)
    )
    assert MORNING.at == "10:00"


def test_the_engines_oracle_matches_a_rule_that_reads_the_truth(sessions):
    a = fleet_replay(sessions, Oracle()).days
    b = fleet_replay(sessions, FleetOracle(sessions)).days
    assert a["digest"].equals(b["digest"])
