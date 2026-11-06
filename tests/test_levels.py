import numpy as np
import pandas as pd
import pytest

from fleetoffer import levels, options
from fleetoffer.baselines import Population
from fleetoffer.checks import check_no_lookahead_days
from fleetoffer.daylevel import DeliveryCalibrated, LevelSweep
from fleetoffer.engine import fleet_replay, training_base_set
from fleetoffer.offer import BaseSet, solve, solve_segment
from fleetoffer.quantiles import GBDT, PerCarRule
from fleetoffer.synthetic import Regime, make_sessions, with_app_entries


def no_flags(dates):
    n = len(dates)
    return pd.DataFrame({"in_term": np.zeros(n), "summer": np.zeros(n), "holiday": np.zeros(n)})


def population():
    return PerCarRule([Population()], min_rows=50, calendar=no_flags)


@pytest.fixture(scope="module")
def claimed():
    sessions = make_sessions(
        [Regime("2001-01-01", sessions_per_weekday=12, median_stay_h=6.0, drivers=40)],
        "2001-03-20",
        seed=5,
    )
    return with_app_entries(sessions, seed=5)


@pytest.mark.parametrize("form", options.FORMS)
def test_the_sweep_finds_the_same_day_score_as_the_delivery_scan(claimed, form):
    # The delivery-calibrated rule stops at the first level whose offer was not delivered;
    # the sweep makes every level's offer. On the same bounds, the last level before the
    # first failure in the sweep is that rule's day score.
    sessions, entries = claimed
    base = training_base_set(sessions, pd.Timestamp("2001-01-20", tz="UTC"))
    make = options.make_offer(form, base)
    some = np.array([0.0, 0.05, 0.2, 0.5, 0.95])  # a few of DELIVERY_LEVELS, to keep it quick
    scan = DeliveryCalibrated(population(), make, levels=some, gamma=0.05)
    fleet_replay(sessions, scan, entries=entries, **options.engine_options(form, base))
    sweep = LevelSweep(population(), {form: make}, levels=some)
    fleet_replay(sessions, sweep, entries=entries, method="bounds")

    log = pd.DataFrame(scan.log)
    log = log[log["fitted"]].set_index("decision_time")
    rows = pd.DataFrame(sweep.log)
    assert len(log) > 20
    compared = 0
    for when, day in rows.groupby("decision_time"):
        if when not in log.index:
            continue
        delivered = day.sort_values("level")["fixed_delivered"].to_numpy(bool)
        first_failure = int(np.argmin(delivered)) if not delivered.all() else len(delivered)
        expected = some[max(first_failure - 1, 0)]
        assert log.loc[when, "score"] == pytest.approx(expected)
        compared += 1
    assert compared == len(log)


def test_at_level_zero_every_car_gets_nothing_and_the_offer_is_empty_and_delivered(claimed):
    sessions, entries = claimed
    sweep = LevelSweep(population(), {"segment": solve_segment}, levels=[0.0, 0.5])
    fleet_replay(sessions, sweep, entries=entries, method="bounds")
    rows = pd.DataFrame(sweep.log)
    zero = rows[rows["level"] == 0.0]
    assert len(zero) > 20
    assert (zero["zero_ceiling_cars"] == zero["cars"]).all()
    assert (zero["zero_stay_cars"] == zero["cars"]).all()
    assert (zero["size_kwh"] == 0).all()
    assert zero["fixed_delivered"].all() and zero["hindsight_delivered"].all()
    half = rows[rows["level"] == 0.5]
    assert (half["zero_ceiling_cars"] <= half["cars"]).all()
    assert (half["zero_ceiling_cars_with_energy"] <= half["cars_with_energy"]).all()


def test_the_level_sweep_does_not_look_ahead(claimed):
    sessions, entries = claimed
    first = sessions["connect_time"].min()
    short = sessions[sessions["connect_time"] < first + pd.Timedelta(days=50)]
    made = []

    def factory(table):
        per_car = PerCarRule([GBDT(max_iter=10)], min_rows=50, calendar=no_flags)
        offers = {"average": lambda f: solve(f, BaseSet.average(f)), "segment": solve_segment}
        rule = LevelSweep(per_car, offers, levels=[0.0, 0.1, 0.5])
        made.append(rule)
        return rule

    cutoff = first + pd.Timedelta(days=47, hours=19)
    compared = check_no_lookahead_days(factory, short, [cutoff], entries=entries, method="bounds")
    assert compared > 25
    real, scrambled = (pd.DataFrame(rule.made) for rule in made)
    before = cutoff.tz_convert("UTC")
    real, scrambled = (
        t[t["decision_time"] < before].reset_index(drop=True) for t in (real, scrambled)
    )
    assert len(real) >= 3 * 2 * 10  # three levels, two forms, at least ten days
    assert real.equals(scrambled)


def test_the_matched_comparison_takes_the_largest_offer_delivered_often_enough():
    curve = pd.DataFrame(
        {
            "model": "4",
            "model_name": "learned",
            "form": "average",
            "level": [0.0, 0.1, 0.2, 0.3],
            "size_mean_kwh": [0.0, 1.0, 2.0, 3.0],
            "fixed_delivered": [1.0, 0.96, 0.91, 0.80],
            "hindsight_delivered": [1.0, 1.0, 1.0, 0.99],
        }
    )
    out = levels.matched(curve).set_index(["measure", "rate"])
    assert out.loc[("fixed_delivered", 0.90), "size_mean_kwh"] == 2.0
    assert out.loc[("fixed_delivered", 0.95), "level"] == 0.1
    assert out.loc[("hindsight_delivered", 0.95), "size_mean_kwh"] == 3.0


def test_every_planned_sweep_run_names_a_known_model_and_form():
    for model, form in levels.RUNS:
        assert form in levels.FORMS
        assert model == "1" or model.removesuffix("k") in levels.MODELS


def test_the_engine_reveals_the_empty_cars_only_when_asked(claimed):
    sessions, entries = claimed
    seen = []

    class Spy(LevelSweep):
        def bounds(self, day):
            seen.append(day.known_empty)
            return super().bounds(day)

    plain = PerCarRule([Population()], min_rows=50, calendar=no_flags)
    fleet_replay(sessions, Spy(plain, {}), entries=entries, method="bounds")
    assert all(flag is None for flag in seen)
    seen.clear()
    per_car = PerCarRule(
        [Population(energy_left_only=True)], min_rows=50, calendar=no_flags, energy_left_only=True
    )
    result = fleet_replay(
        sessions, Spy(per_car, {}), entries=entries, method="bounds", reveal_empty=True
    )
    cars = result.cars
    for flag, (_, day) in zip(seen, cars.groupby("decision_time", sort=True), strict=True):
        assert np.array_equal(flag, day["true_energy_kwh"].to_numpy() <= 1e-9)
    assert any(flag.any() for flag in seen)


def test_with_the_empty_cars_known_they_are_dropped_and_the_models_train_without_them(claimed):
    sessions, entries = claimed
    per_car = PerCarRule(
        [Population(energy_left_only=True)], min_rows=50, calendar=no_flags, energy_left_only=True
    )
    sweep = LevelSweep(per_car, {"segment": solve_segment}, levels=[0.1], drop_empty=True)
    result = fleet_replay(sessions, sweep, entries=entries, method="bounds", reveal_empty=True)
    rows = pd.DataFrame(sweep.log).set_index("decision_time")
    empty = (result.cars["true_energy_kwh"] <= 1e-9).groupby(result.cars["decision_time"]).sum()
    total = result.cars.groupby("decision_time").size()
    assert (rows["dropped_cars"] == empty.loc[rows.index]).all()
    assert (rows["cars"] == (total - empty).loc[rows.index]).all()
    assert rows["dropped_cars"].sum() > 0
    with pytest.raises(ValueError):
        LevelSweep(population(), {}, drop_empty=True)


def test_the_accounting_reads_each_step_at_its_own_delivery_rate():
    def curve(model, sizes, fixed, hindsight):
        return pd.DataFrame(
            {
                "model": model,
                "model_name": model,
                "form": "average",
                "level": [0.0, 0.1, 0.2],
                "size_mean_kwh": sizes,
                "fixed_delivered": fixed,
                "hindsight_delivered": hindsight,
            }
        )

    table = pd.concat(
        [
            curve("4", [0.0, 1.0, 2.0], [1.0, 0.92, 0.80], [1.0, 1.0, 0.95]),
            curve("4k", [0.0, 5.0, 9.0], [1.0, 0.95, 0.91], [1.0, 1.0, 1.0]),
        ]
    )
    oracle = pd.DataFrame(
        {
            "form": ["average"],
            "size_mean_kwh": [40.0],
            "fixed_delivered": [1.0],
            "exact_true_kwh": [90.0],
        }
    )
    online = pd.DataFrame(
        {"arm": ["4d"], "form": ["average"], "size_mean_kwh": [0.5], "fixed_delivered": [0.94]}
    )
    out = levels.accounting(table, oracle, online).set_index(["step", "who"])
    assert out.loc[("exact band of the true sets", "1"), "size_mean_kwh"] == 90.0
    assert out.loc[("perfect knowledge", "1"), "share_of_perfect"] == 1.0
    assert out.loc[("forecasts at 90% delivery", "4"), "size_mean_kwh"] == 1.0
    assert out.loc[("forecasts at 90% delivery", "4"), "delivered"] == 0.92
    assert out.loc[("empty cars known", "4k"), "size_mean_kwh"] == 9.0
    assert out.loc[("hindsight split at 90%", "4"), "size_mean_kwh"] == 2.0
    assert out.loc[("day-level rule as run", "4d"), "share_of_perfect"] == 0.5 / 40.0
