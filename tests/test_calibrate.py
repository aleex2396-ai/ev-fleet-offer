"""What the day-level guarantees require, against hand arithmetic and the plan's numbers."""

import math

import pytest

from fleetoffer import calibrate


def test_corollary1_matches_hand_arithmetic():
    # ln(10 / 0.05) / (2 * 0.025^2) = 5.2983 / 0.00125 = 4238.65, rounded up.
    assert calibrate.corollary1_sample_size(0.10, 0.05, 0.025, 0.05, 10) == 4239


@pytest.mark.parametrize(
    ("alpha", "theta"),
    [(0.0, 0.05), (0.10, 0.01), (0.05, 0.05), (0.05, 0.0)],
)
def test_corollary1_refuses_settings_outside_its_statement(alpha, theta):
    with pytest.raises(ValueError):
        calibrate.corollary1_sample_size(0.10, alpha, theta, 0.05, 10)


@pytest.mark.parametrize(
    ("k", "n", "p"), [(0, 5, 0.3), (2, 10, 0.075), (7, 40, 0.2), (12, 12, 0.5)]
)
def test_binomial_cdf_matches_direct_sum(k, n, p):
    direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert calibrate.binomial_cdf(k, n, p) == pytest.approx(direct, rel=1e-12)


def test_binomial_cdf_survives_large_n():
    assert 0 < calibrate.binomial_cdf(400, 10_000, 0.075) < 1e-20


def test_theorem4_bound_with_no_discards_is_a_power():
    # alpha = 0: every sample must be kept, so the bound is |X| (1 - (eps - theta))^N.
    assert calibrate.theorem4_bound(50, 0.10, 0.0, 0.05, 7) == pytest.approx(7 * 0.95**50)


def test_theorem4_sample_size_is_the_first_n_that_stays_below_delta():
    n = calibrate.theorem4_sample_size(0.10, 0.05, 0.025, 0.05, 10, n_max=10_000)
    assert n is not None
    assert all(
        calibrate.theorem4_bound(m, 0.10, 0.05, 0.025, 10) <= 0.05 for m in range(n, n + 300)
    )
    assert calibrate.theorem4_bound(n - 1, 0.10, 0.05, 0.025, 10) > 0.05


def test_theorem4_needs_no_more_samples_than_corollary1():
    for alpha, theta in [(0.05, 0.025), (0.02, 0.04), (0.05, 0.01)]:
        exact = calibrate.theorem4_sample_size(0.10, alpha, theta, 0.05, 100, n_max=20_000)
        assert exact <= calibrate.corollary1_sample_size(0.10, alpha, theta, 0.05, 100)


def test_radius_is_drift_over_theta():
    assert calibrate.radius(0.07, 0.025) == pytest.approx(2.8)
    with pytest.raises(ValueError):
        calibrate.radius(0.07, 0.0)


def test_the_plan_numbers_for_days_as_samples():
    # Analysis plan, Section 6: |X| = 101 scaling levels, eps = 0.10, delta = 0.05.
    table = calibrate.requirements_table().set_index(["alpha", "theta"])
    assert table.loc[(0.02, 0.04)].to_dict() == {
        "ball_over_drift_budget": 25.0,
        "theorem4_first_below": 285,
        "theorem4_from": 312,
        "corollary1": 2379,
        "corollary1_over_theorem4": 7.6,
    }
    assert table.loc[(0.05, 0.025)].to_dict() == {
        "ball_over_drift_budget": 40.0,
        "theorem4_first_below": 1057,
        "theorem4_from": 1121,
        "corollary1": 6089,
        "corollary1_over_theorem4": 5.4,
    }
    # The development record's 483 weekdays meet the first setting by a wide margin.
    assert calibrate.theorem4_bound(483, 0.10, 0.02, 0.04, 101) < 0.001


def test_weekdays_the_record_holds():
    days = calibrate.days_available().set_index("span")["weekdays"]
    assert days.iloc[0] == 483  # 25 April 2018 to 29 February 2020
    assert days.iloc[2] == 182  # 1 January to 13 September 2021
    assert calibrate.weekdays("2019-06-03", "2019-06-09") == 5


def test_union_level_splits_the_miss_over_two_bounds_per_car():
    assert calibrate.union_level(40) == pytest.approx(0.00125)
    assert calibrate.union_level(1) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        calibrate.union_level(0)


def test_adaptive_conformal_gap_and_smallest_step():
    # (max{0.1, 0.9} + gamma) / (gamma T): with T = 182 and gamma = 0.1, (1.0) / 18.2.
    assert calibrate.aci_gap(182, 0.1) == pytest.approx(1.0 / 18.2)
    gamma = calibrate.aci_smallest_step(182, 0.05)
    assert gamma == pytest.approx(0.9 / 8.1)
    assert round(gamma, 2) == 0.11
    assert calibrate.aci_gap(182, gamma) == pytest.approx(0.05)
    assert calibrate.aci_smallest_step(20, 0.05) is None  # no step reaches 5 points in 20 days
    with pytest.raises(ValueError):
        calibrate.aci_gap(182, 0.1, start_level=1.5)
