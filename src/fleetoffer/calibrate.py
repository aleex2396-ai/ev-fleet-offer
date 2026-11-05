"""Day-level calibration: what each rule for a 90%-of-days target requires and gives.

Three routes to the target are compared in the analysis plan (Section 5): a union bound
across every present car's two bounds, adaptive conformal inference with days as steps, and
a split conformal rule on past days. This module holds the arithmetic of their guarantees,
computed from the exact statements quoted in `docs/theory.md`.

Union bound. If each of the 2n per-car bounds (deadline and energy for n cars) misses with
probability at most q, all hold at once with probability at least 1 - 2nq, with no assumption
about independence between cars. A 90% day needs q = 0.10 / (2n).

Gibbs and Candes, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021,
Proposition 4.1: with the working level updated as a_{t+1} = a_t + gamma (alpha - err_t),
the average miss rate over T steps is within (max{a_1, 1 - a_1} + gamma) / (gamma T) of the
target alpha, with no constraint on the data-generating distribution.

Yan, Parise and Bitar, "Data-Driven Approximations of Chance Constrained Programs in
Nonstationary Environments", IEEE Control Systems Letters 6 (2022), arXiv:2205.03748,
Section III:

- Condition (9): the 1-Wasserstein distance between the distributions of samples i and i + k
  is at most rho(k), for every i >= 1 and k >= 0, with rho a known function, rho(0) = 0.
- Theorem 4 (finite set X of candidate decisions):
  Pr(the robust sampled solution set is not inside the true eps-feasible set)
  <= |X| Psi(alpha N; p_1..p_N), p_i = (eps - rho(N + 1 - i) / r_i)_+, with Psi the
  distribution function of a sum of independent Bernoulli variables (Definition 2).
- Remark 4, (14): radii r_i = rho(N + 1 - i) / theta. Then every p_i = eps - theta, and Psi
  is the binomial distribution function with N trials and success chance eps - theta.
- Corollary 1, (15): with (14), delta in (0, 1), alpha in (0, eps), theta in (0, eps - alpha),
  N >= ln(|X| / delta) / (2 (eps - alpha - theta)^2) gives confidence at least 1 - delta. It
  follows from Theorem 4 through Hoeffding's inequality, so it is sufficient, not necessary.

Here a sample is a whole day, the decision is one of 101 scaling levels of the offer, and
eps = 0.10 for a 90%-of-days target. Each paper's alpha keeps its own meaning above.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# The settings of the analysis plan, Section 6.
TARGET_MISS = 0.10  # eps: the offer may fail on at most 10% of days
CONFIDENCE_DELTA = 0.05
SCALE_LEVELS = 101  # |X|: a grid of 101 scaling levels of the offer
SETTINGS = ((0.02, 0.04), (0.05, 0.025))  # (alpha, theta) pairs inside Corollary 1's ranges

# Weekdays the record holds (calendar weekdays, both ends included, Pacific dates).
WEEKDAY_SPANS = {
    "development, 25 April 2018 to 29 February 2020": ("2018-04-25", "2020-02-29"),
    "one rolling year (a calendar year)": ("2019-01-01", "2019-12-31"),
    "2021, 1 January to 13 September": ("2021-01-01", "2021-09-13"),
}


def union_level(n_cars: int, target_miss: float = TARGET_MISS) -> float:
    """Per-bound miss level q = target / (2n) for n cars with two bounds each."""
    if n_cars < 1:
        raise ValueError("need at least one car")
    return target_miss / (2 * n_cars)


def aci_gap(steps: int, gamma: float, start_level: float = TARGET_MISS) -> float:
    """Proposition 4.1: the long-run miss rate is within this of the target after `steps`."""
    if not 0.0 <= start_level <= 1.0:
        raise ValueError("the starting level must be in [0, 1]")
    if gamma <= 0 or steps < 1:
        raise ValueError("need a positive step and at least one step")
    return (max(start_level, 1.0 - start_level) + gamma) / (gamma * steps)


def aci_smallest_step(steps: int, within: float, start_level: float = TARGET_MISS) -> float | None:
    """Smallest gamma for which `aci_gap` is at most `within`; None if no gamma reaches it.

    (m + gamma) / (gamma T) <= w  is  gamma >= m / (w T - 1), with m = max{a_1, 1 - a_1}.
    """
    m = max(start_level, 1.0 - start_level)
    if within * steps <= 1.0:
        return None
    return m / (within * steps - 1.0)


def _check_corollary1(eps: float, alpha: float, theta: float, delta: float, n_candidates: int):
    if not 0 < delta < 1:
        raise ValueError("Corollary 1 needs delta in (0, 1)")
    if not 0 < alpha < eps:
        raise ValueError("Corollary 1 needs alpha in (0, eps)")
    if not 0 < theta < eps - alpha:
        raise ValueError("Corollary 1 needs theta in (0, eps - alpha)")
    if n_candidates < 1:
        raise ValueError("the set of candidate decisions must be non-empty")


def corollary1_sample_size(
    eps: float, alpha: float, theta: float, delta: float, n_candidates: int
) -> int:
    """Smallest N meeting (15): ln(|X| / delta) / (2 (eps - alpha - theta)^2), rounded up."""
    _check_corollary1(eps, alpha, theta, delta, n_candidates)
    return math.ceil(math.log(n_candidates / delta) / (2 * (eps - alpha - theta) ** 2))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Pr(Binomial(n, p) <= k), summed in log space so large n does not underflow."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    i = np.arange(k + 1)
    log_ratio = np.log(n - i[:-1]) - np.log(i[:-1] + 1) + math.log(p) - math.log1p(-p)
    log_pmf = n * math.log1p(-p) + np.concatenate([[0.0], np.cumsum(log_ratio)])
    top = log_pmf.max()
    return float(min(1.0, math.exp(top) * np.exp(log_pmf - top).sum()))


def theorem4_bound(n: int, eps: float, alpha: float, theta: float, n_candidates: int) -> float:
    """Theorem 4's bound with the radii (14): |X| times Pr(Binomial(N, eps - theta) <= alpha N)."""
    return n_candidates * binomial_cdf(math.floor(alpha * n), n, eps - theta)


def _check_theorem4(eps, alpha, theta, delta, n_candidates):
    if not 0 < delta < 1 or not 0 <= alpha < eps or not 0 < theta <= eps or n_candidates < 1:
        raise ValueError("need delta in (0, 1), 0 <= alpha < eps, 0 < theta <= eps, |X| >= 1")


def theorem4_first_below(
    eps: float, alpha: float, theta: float, delta: float, n_candidates: int, n_max: int = 50_000
) -> int | None:
    """Smallest N at which Theorem 4's bound is at most delta (it may rise above it again)."""
    _check_theorem4(eps, alpha, theta, delta, n_candidates)
    for n in range(1, n_max + 1):
        if theorem4_bound(n, eps, alpha, theta, n_candidates) <= delta:
            return n
    return None


def theorem4_sample_size(
    eps: float, alpha: float, theta: float, delta: float, n_candidates: int, n_max: int = 50_000
) -> int | None:
    """Smallest N from which Theorem 4's bound stays at most delta for every N up to n_max.

    The bound is not monotone in N (alpha N is rounded down), so "from which it stays" is
    the honest reading. Returns None if the bound is still above delta at n_max.
    """
    _check_theorem4(eps, alpha, theta, delta, n_candidates)
    last_failure = 0
    for n in range(1, n_max + 1):
        if theorem4_bound(n, eps, alpha, theta, n_candidates) > delta:
            last_failure = n
    return None if last_failure == n_max else last_failure + 1


def radius(rho: float, theta: float) -> float:
    """Remark 4, (14): the ball around a sample whose distribution may sit rho away."""
    if theta <= 0:
        raise ValueError("theta must be positive")
    return rho / theta


def weekdays(first: str, last: str) -> int:
    """Calendar weekdays from `first` to `last`, both included."""
    end = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return int(np.busday_count(first, end))


def requirements_table(eps: float = TARGET_MISS, delta: float = CONFIDENCE_DELTA) -> pd.DataFrame:
    """Days needed by Theorem 4 and by Corollary 1 at the plan's two settings."""
    rows = []
    for alpha, theta in SETTINGS:
        first = theorem4_first_below(eps, alpha, theta, delta, SCALE_LEVELS)
        stays = theorem4_sample_size(eps, alpha, theta, delta, SCALE_LEVELS)
        corollary = corollary1_sample_size(eps, alpha, theta, delta, SCALE_LEVELS)
        rows.append(
            {
                "alpha": alpha,
                "theta": theta,
                "ball_over_drift_budget": round(1 / theta, 1),
                "theorem4_first_below": first,
                "theorem4_from": stays,
                "corollary1": corollary,
                "corollary1_over_theorem4": round(corollary / stays, 1),
            }
        )
    return pd.DataFrame(rows)


def days_available() -> pd.DataFrame:
    """Calendar weekdays in each span the plan sets against the requirement."""
    return pd.DataFrame(
        [{"span": name, "weekdays": weekdays(*span)} for name, span in WEEKDAY_SPANS.items()]
    )
