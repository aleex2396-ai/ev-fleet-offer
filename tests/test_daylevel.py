import numpy as np
import pandas as pd
import pytest

from fleetoffer.baselines import Population
from fleetoffer.checks import check_no_lookahead_days
from fleetoffer.daylevel import (
    DELIVERY_LEVELS,
    GRID,
    Calibrated,
    DeliveryCalibrated,
    UnionBound,
    at_levels,
    conformal_level,
    day_score,
    delivery_score,
    held_level,
    is_scored,
    supported,
)
from fleetoffer.engine import fleet_replay
from fleetoffer.offer import STEP_H, STEPS, BaseSet, Fleet, solve, solve_segment
from fleetoffer.quantiles import GBDT, LEVELS, PerCarRule, at_level
from fleetoffer.synthetic import Regime, make_sessions, with_app_entries


def no_flags(dates):
    n = len(dates)
    return pd.DataFrame({"in_term": np.zeros(n), "summer": np.zeros(n), "holiday": np.zeros(n)})


def grids(n, rng):
    stay = np.sort(rng.uniform(0, 12, (n, len(LEVELS))), axis=1)
    energy = np.sort(rng.uniform(0, 25, (n, len(LEVELS))), axis=1)
    return stay, energy


def engine_holds(stay_h, energy, true_deadline, true_energy):
    """What the engine counts as held (engine.fleet_replay)."""
    steps = np.clip(np.floor(np.minimum(stay_h, STEPS * STEP_H) / STEP_H + 1e-9), 0, STEPS)
    return (steps <= true_deadline) & (energy <= true_energy + 1e-9)


def test_reading_levels_car_by_car_matches_reading_one_level():
    rng = np.random.default_rng(0)
    stay, _ = grids(30, rng)
    for level in [0.0, 0.001, 0.002, 0.003, 0.05, 0.1, 0.125, 0.5, 0.93, 0.95, 0.99]:
        assert np.array_equal(at_levels(stay, np.full(30, level)), at_level(stay, level))


def test_the_held_level_is_the_last_grid_level_at_which_the_bound_held():
    rng = np.random.default_rng(1)
    stay, energy = grids(200, rng)
    deadline = rng.integers(0, STEPS + 1, 200)
    taken = rng.uniform(0, 25, 200)

    def holds(grid, level, i, target):
        bound = at_level(grid[i : i + 1], level)
        if target == "stay":
            return engine_holds(bound, 0.0, deadline[i], 1.0)[0]
        return engine_holds(0.0, bound, STEPS, taken[i])[0]

    for grid, target in ((stay, "stay"), (energy, "energy")):
        level = held_level(grid, deadline if target == "stay" else taken, target)
        index = np.searchsorted(GRID, level)
        assert np.array_equal(GRID[index], level)
        for i in range(200):
            assert holds(grid, level[i], i, target)
            if index[i] + 1 < len(GRID):
                assert not holds(grid, GRID[index[i] + 1], i, target)


def test_every_bound_holds_at_a_level_exactly_when_the_level_is_at_most_the_day_score():
    rng = np.random.default_rng(2)
    for _ in range(20):
        stay, energy = grids(15, rng)
        deadline, taken = rng.integers(8, STEPS + 1, 15), rng.uniform(2, 25, 15)
        score = day_score(stay, energy, deadline, taken)
        for q in GRID[::7]:
            held = engine_holds(at_level(stay, q), at_level(energy, q), deadline, taken).all()
            assert held == (q <= score)


def test_a_car_without_a_prediction_never_limits_the_day():
    rng = np.random.default_rng(3)
    stay, energy = grids(4, rng)
    stay[0] = energy[0] = np.nan
    assert held_level(stay, np.zeros(4), "stay")[0] == GRID[-1]


def test_the_conformal_rank():
    scores = np.arange(1, 20) / 100  # 19 scores
    assert conformal_level(scores, 0.10) == 0.02  # k = floor(0.1 * 20) = 2
    assert conformal_level(scores[:8], 0.10) == 0.0  # k = 0: trivial
    assert conformal_level(scores, 0.0) == 0.0
    assert conformal_level(scores, 1.0) == GRID[-1]
    assert conformal_level(np.array([]), 0.5) == 0.0


def test_a_day_is_scored_with_five_cars_taking_more_than_half_a_kwh():
    assert is_scored(np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.0]))
    assert not is_scored(np.array([0.6, 0.6, 0.6, 0.6, 0.5]))  # 0.5 is not above 0.5


def test_d10_support_between_grid_levels():
    levels = (0.002, 0.005)
    assert supported(0.002, levels) and supported(0.003, levels) and supported(0.005, levels)
    assert not supported(0.006, levels) and not supported(0.001, levels)


@pytest.fixture(scope="module")
def claimed():
    sessions = make_sessions(
        [Regime("2001-01-01", sessions_per_weekday=12, median_stay_h=6.0, drivers=40)],
        "2001-04-20",
        seed=5,
    )
    return with_app_entries(sessions, seed=5)


def _known_scores(log: pd.DataFrame, now, window_days=90):
    past = log[
        log["scored"].fillna(False).astype(bool)
        & (log["revealed_at"] < now)
        & (log["decision_time"] >= now - pd.Timedelta(days=window_days))
    ]
    return past["score"].to_numpy(float)


def test_adaptive_conformal_rule_in_the_engine(claimed):
    sessions, entries = claimed
    rule = Calibrated(
        PerCarRule([Population()], min_rows=50, calendar=no_flags), rule="aci", gamma=0.05
    )
    days = fleet_replay(sessions, rule, method="bounds", entries=entries).days
    log = pd.DataFrame(rule.log)
    scored = log[log["fitted"]]
    assert len(scored) > 40
    # the level used each day is the rule applied to the scores known that morning
    for _, row in scored.iterrows():
        expect = conformal_level(_known_scores(log, row["decision_time"]), row["miss_at_decision"])
        assert row["level"] == pytest.approx(min(expect, GRID[-1]))
    # a day failed by its score exactly when the engine saw a bound fail
    merged = scored.merge(days[["decision_time", "all_bounds_held"]], on="decision_time")
    assert (merged["failed"] == ~merged["all_bounds_held"].astype(bool)).all()
    # the recursion: the final working level is the start plus gamma times the summed errors
    counted = log[log["scored"].fillna(False).astype(bool)]
    expected = 0.10 + 0.05 * (0.10 - counted["failed"].astype(float)).sum()
    assert rule.miss == pytest.approx(expected)


def test_split_rule_runs_only_on_a_long_enough_window(claimed):
    sessions, entries = claimed
    rule = Calibrated(
        PerCarRule([Population()], min_rows=50, calendar=no_flags), rule="split", min_days=10
    )
    fleet_replay(sessions, rule, method="bounds", entries=entries)
    log = pd.DataFrame(rule.log)
    log = log[log["fitted"]]
    assert set(log["rule"]) == {"aci", "split"}
    for _, row in log.iterrows():
        known = _known_scores(pd.DataFrame(rule.log), row["decision_time"])
        assert (row["rule"] == "split") == (len(known) >= 10)
        if row["rule"] == "split":
            assert row["level"] == conformal_level(known, 0.10)


def test_union_bound_gives_the_trivial_bound_where_d10_does_not_support_the_level(claimed):
    sessions, entries = claimed
    rule = UnionBound(
        PerCarRule([Population()], min_rows=50, calendar=no_flags),
        {"stay": (0.002, 0.005), "energy": ()},
    )
    result = fleet_replay(sessions, rule, method="bounds", entries=entries)
    log = pd.DataFrame(rule.log)
    assert (log["level"] == 0.10 / (2 * log["cars"])).all()
    cars = result.cars.merge(log[["decision_time", "stay_supported"]], on="decision_time")
    assert (cars["learned_energy_kwh"] == 0).all()  # energy never supported here
    assert (cars.loc[~cars["stay_supported"], "learned_deadline"] == 0).all()


def test_the_calibrated_offer_does_not_look_ahead(claimed):
    sessions, entries = claimed
    first = sessions["connect_time"].min()
    short = sessions[sessions["connect_time"] < first + pd.Timedelta(days=50)]

    def factory(table):
        per_car = PerCarRule([GBDT(max_iter=10)], min_rows=50, calendar=no_flags)
        return Calibrated(per_car, rule="aci", gamma=0.2)

    cutoff = first + pd.Timedelta(days=47, hours=19)
    assert check_no_lookahead_days(factory, short, [cutoff], entries=entries) > 25


def test_the_step_is_the_smallest_that_keeps_failures_within_five_points_over_116_days():
    from fleetoffer.daylevel import ACI_GAMMA, ACI_HORIZON_DAYS, aci_failure_bound, aci_step

    assert ACI_HORIZON_DAYS == 116 and np.ceil(aci_step() * 100) / 100 == ACI_GAMMA
    assert aci_failure_bound(116) <= 0.15 < aci_failure_bound(116, gamma=ACI_GAMMA - 0.01)


def test_failed_days_stay_within_the_bound_even_against_the_worst_days():
    """Whatever the days do, the recursion caps the share of failed days: here every day
    fails whenever the rule offers anything above the trivial bound."""
    from fleetoffer.daylevel import ACI_GAMMA, aci_failure_bound

    rng = np.random.default_rng(4)
    miss, scores, failed, levels = 0.10, [], [], []
    for day in range(260):
        levels.append(miss)
        # friendly days first (every bound holds), so the working level climbs; then the worst
        level = conformal_level(np.array(scores[-60:]), miss) if miss < 1 else GRID[-1]
        if day < 100:
            score = GRID[-1]
        else:
            score = max(level - 0.001, 0.0) if level > 0 else rng.uniform(0, 0.5)
        fail = level > score
        failed.append(fail)
        scores.append(score)
        miss += ACI_GAMMA * (0.10 - fail)
    assert np.mean(failed) <= aci_failure_bound(260)
    late = slice(100, 216)  # a later stretch of 116 days obeys the bound from its own start
    assert np.mean(failed[late]) <= aci_failure_bound(116, start=levels[100])
    assert np.mean(failed[late]) > aci_failure_bound(116)  # ... which a_1 alone would not give
    assert np.mean(failed) > 0.10  # the worst case does push it above the target


# Calibration on delivery (measured on the model-selection months; not the registered rule).


def one_car(true_energy=4.0):
    """One car staying past 20:00 whose energy bound at level q is 10 q kWh."""
    stay = np.full((1, len(LEVELS)), 12.0)
    energy = 10.0 * np.asarray(LEVELS, dtype=float)[None, :]
    return stay, energy, Fleet.of([STEPS], [true_energy])


def test_with_the_cars_own_set_as_template_delivery_fails_once_a_bound_fails():
    stay, energy, true = one_car()
    score, made = delivery_score(stay, energy, true, lambda f: solve(f, BaseSet.average(f)))
    assert score == pytest.approx(0.4)  # 10 q <= 4 kWh
    assert made == list(DELIVERY_LEVELS).index(0.5) + 1  # stops at the first failure
    assert day_score(stay, energy, true.deadline, true.energy) == pytest.approx(0.4)


def test_the_fallback_offer_is_delivered_past_a_failed_energy_bound():
    # The fallback's reference charges half the ceiling and its shift moves no energy, so
    # with one car it asks 5 q kWh: delivered up to q = 0.8, though the bound fails above 0.4.
    stay, energy, true = one_car()
    score, _ = delivery_score(stay, energy, true, solve_segment)
    assert score == pytest.approx(0.8)
    assert day_score(stay, energy, true.deadline, true.energy) == pytest.approx(0.4)


def test_an_offer_delivered_at_every_level_scores_the_top():
    stay, energy, true = one_car(true_energy=20.0)
    assert delivery_score(stay, energy, true, solve_segment) == (DELIVERY_LEVELS[-1], 17)


def test_delivery_calibrated_rule_in_the_engine(claimed):
    sessions, entries = claimed
    rule = DeliveryCalibrated(
        PerCarRule([Population()], min_rows=50, calendar=no_flags), solve_segment, gamma=0.05
    )
    days = fleet_replay(sessions, rule, method="segment", entries=entries).days
    log = pd.DataFrame(rule.log)
    scored = log[log["fitted"]]
    assert len(scored) > 40
    assert scored["score"].isin(DELIVERY_LEVELS).all()
    for _, row in scored.iterrows():
        expect = conformal_level(_known_scores(log, row["decision_time"]), row["miss_at_decision"])
        assert row["level"] == pytest.approx(min(expect, GRID[-1]))
    # every day the engine's offer was not delivered counts as failed (the converse need not
    # hold: a lower level may have failed while the level used did not)
    merged = scored.merge(days[["decision_time", "fixed_delivered"]], on="decision_time")
    failed = merged["failed"].astype(bool)
    assert failed[~merged["fixed_delivered"].astype(bool)].all()
    assert ((merged["level"] <= merged["score"]) == ~failed).all()
    counted = log[log["scored"].fillna(False).astype(bool)]
    expected = 0.10 + 0.05 * (0.10 - counted["failed"].astype(float)).sum()
    assert rule.miss == pytest.approx(expected)


def test_the_delivery_calibrated_offer_does_not_look_ahead(claimed):
    sessions, entries = claimed
    first = sessions["connect_time"].min()
    short = sessions[sessions["connect_time"] < first + pd.Timedelta(days=50)]

    def factory(table):
        per_car = PerCarRule([GBDT(max_iter=10)], min_rows=50, calendar=no_flags)
        return DeliveryCalibrated(per_car, solve_segment, gamma=0.2)

    cutoff = first + pd.Timedelta(days=47, hours=19)
    compared = check_no_lookahead_days(factory, short, [cutoff], entries=entries, method="segment")
    assert compared > 25
