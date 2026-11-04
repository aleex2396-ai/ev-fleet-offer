"""The guarantee's requirements, against values worked out by hand."""

import math

import pytest

from staypromise import theory


def test_corollary1_matches_hand_arithmetic():
    # ln(10 / 0.05) / (2 * 0.025^2) = 5.2983 / 0.00125 = 4238.65, rounded up.
    assert theory.corollary1_sample_size(0.10, 0.05, 0.025, 0.05, 10) == 4239


@pytest.mark.parametrize(
    ("alpha", "theta"),
    [(0.0, 0.05), (0.10, 0.01), (0.05, 0.05), (0.05, 0.0)],
)
def test_corollary1_refuses_settings_outside_its_statement(alpha, theta):
    with pytest.raises(ValueError):
        theory.corollary1_sample_size(0.10, alpha, theta, 0.05, 10)


@pytest.mark.parametrize(
    ("k", "n", "p"), [(0, 5, 0.3), (2, 10, 0.075), (7, 40, 0.2), (12, 12, 0.5)]
)
def test_binomial_cdf_matches_direct_sum(k, n, p):
    direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert theory.binomial_cdf(k, n, p) == pytest.approx(direct, rel=1e-12)


def test_binomial_cdf_survives_large_n():
    assert 0 < theory.binomial_cdf(400, 10_000, 0.075) < 1e-20


def test_theorem4_bound_with_no_discards_is_a_power():
    # alpha = 0: every sample must be kept, so the bound is |X| (1 - (eps - theta))^N.
    assert theory.theorem4_bound(50, 0.10, 0.0, 0.05, 7) == pytest.approx(7 * 0.95**50)


def test_theorem4_sample_size_is_the_first_n_that_stays_below_delta():
    n = theory.theorem4_sample_size(0.10, 0.05, 0.025, 0.05, 10, n_max=10_000)
    assert n is not None
    assert all(theory.theorem4_bound(m, 0.10, 0.05, 0.025, 10) <= 0.05 for m in range(n, n + 300))
    assert theory.theorem4_bound(n - 1, 0.10, 0.05, 0.025, 10) > 0.05


def test_theorem4_needs_no_more_samples_than_corollary1():
    for alpha, theta in [(0.05, 0.025), (0.02, 0.04), (0.05, 0.01)]:
        exact = theory.theorem4_sample_size(0.10, alpha, theta, 0.05, 100, n_max=20_000)
        assert exact <= theory.corollary1_sample_size(0.10, alpha, theta, 0.05, 100)


def test_radius_is_drift_over_theta():
    assert theory.radius(0.07, 0.025) == pytest.approx(2.8)
    with pytest.raises(ValueError):
        theory.radius(0.07, 0.0)
