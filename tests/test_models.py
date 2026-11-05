import math

import numpy as np
import pytest

from fleetoffer.models import km_curve, lower_quantile


def test_kaplan_meier_by_hand():
    # Stays of 1, 2, 2+, 3 and 4+ hours (+ = still plugged in).
    times, survival = km_curve([1, 2, 2, 3, 4], [True, True, False, True, False])
    assert list(times) == [1, 2, 3]
    # At risk: 5, then 4, then 2.  S = 4/5, then 4/5 * 3/4, then that * 1/2.
    assert list(survival) == pytest.approx([0.8, 0.6, 0.3])


@pytest.mark.parametrize("n", [5, 10, 20, 37, 100])
@pytest.mark.parametrize("level", [0.05, 0.1, 0.2])
def test_without_censoring_it_is_the_usual_lower_quantile(n, level):
    stays = np.random.default_rng(n).lognormal(1.0, 0.6, n)
    bound = lower_quantile(*km_curve(stays, np.ones(n, bool)), level)
    # The largest bound that fails for at most a `level` share of the sample.
    assert bound == np.sort(stays)[math.floor(level * n + 1e-9)]
    assert np.mean(stays < bound) <= level


def test_a_car_already_plugged_in_gets_a_conditional_bound():
    stays = np.arange(1.0, 11.0)  # 1, 2, ..., 10 hours, all ended
    curve = km_curve(stays, np.ones(10, bool))
    # Among stays of at least 3.5 h (4..10), staying at least 0.5 h more fails 0 of 7 times;
    # 1.5 h more would fail 1 of 7 (14%), above the 10% allowed.
    assert lower_quantile(*curve, 0.1, elapsed_h=3.5) == pytest.approx(0.5)


def test_cars_still_plugged_in_are_used_not_dropped():
    rng = np.random.default_rng(0)
    stays = rng.lognormal(np.log(4.0), 0.6, 40_000)
    seen_for = rng.uniform(0.0, 12.0, stays.size)  # how long each car had been watched
    ended = stays <= seen_for
    observed = np.where(ended, stays, seen_for)
    true_q = 4.0 * math.exp(-1.2815516 * 0.6)

    with_km = lower_quantile(*km_curve(observed, ended), 0.1)
    dropping_unfinished = np.quantile(observed[ended], 0.1)
    assert with_km == pytest.approx(true_q, rel=0.02)
    assert dropping_unfinished < 0.9 * true_q  # the bias the estimator avoids


def test_returns_none_when_the_data_cannot_say():
    assert lower_quantile(*km_curve([1.0, 2.0], [False, False]), 0.1) is None
    assert lower_quantile(*km_curve([], []), 0.1) is None
    # Only one of 20 stays has ended, so a 10% failure level is never reached.
    observed = np.r_[0.5, np.full(19, 3.0)]
    ended = np.r_[True, np.zeros(19, bool)]
    assert lower_quantile(*km_curve(observed, ended), 0.1) is None


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1])
def test_level_must_be_a_proper_share(level):
    with pytest.raises(ValueError):
        lower_quantile(*km_curve([1.0], [True]), level)
