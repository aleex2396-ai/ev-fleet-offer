"""What the published guarantees require, computed from their exact statements.

Yan, Parise and Bitar, "Data-Driven Approximations of Chance Constrained Programs in
Nonstationary Environments", arXiv:2205.03748v1 (2022), Section III:

- Condition (9): the 1-Wasserstein distance between the distributions of samples i and i + k
  is at most rho(k), for every i >= 1 and k >= 0, with rho a known function, rho(0) = 0.
- Robust sample approximation (10)-(12): a ball of radius r_i around each past sample; a
  decision is kept if it satisfies the constraint on the whole ball for at least
  ceil((1 - alpha) N) of the N samples.
- Theorem 4 (finite set X of candidate decisions):
  Pr(the kept set is not inside the target's feasible set) <= |X| Psi(alpha N; p_1..p_N),
  p_i = (eps - rho(N + 1 - i) / r_i)_+, with Psi the Poisson-binomial distribution function
  (Definition 2).
- Remark 4, (14): radii r_i = rho(N + 1 - i) / theta. Then every p_i = eps - theta, and
  Psi is the binomial distribution function with N trials and success chance eps - theta.
- Corollary 1, (15): with (14), delta in (0, 1), alpha in (0, eps), theta in (0, eps - alpha),
  N >= ln(|X| / delta) / (2 (eps - alpha - theta)^2) gives confidence at least 1 - delta.
  It follows from Theorem 4 through Hoeffding's inequality, so it is sufficient, not necessary.

Here a promise "the car stays at least x hours" is the decision x, the stay is the sample,
and the constraint is x <= stay; eps = 0.10 for a 90% promise.
"""

import math

import numpy as np


def _check_corollary1(eps: float, alpha: float, theta: float, delta: float, n_candidates: int):
    if not 0 < delta < 1:
        raise ValueError("Corollary 1 needs delta in (0, 1)")
    if not 0 < alpha < eps:
        raise ValueError("Corollary 1 needs alpha in (0, eps)")
    if not 0 < theta < eps - alpha:
        raise ValueError("Corollary 1 needs theta in (0, eps - alpha)")
    if n_candidates < 1:
        raise ValueError("the set of candidate promises must be non-empty")


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


def theorem4_sample_size(
    eps: float,
    alpha: float,
    theta: float,
    delta: float,
    n_candidates: int,
    n_max: int = 50_000,
) -> int | None:
    """Smallest N from which Theorem 4's bound stays at most delta for every N up to n_max.

    The bound is not monotone in N (alpha N is rounded down), so "from which it stays" is
    the honest reading. Returns None if the bound is still above delta at n_max.
    """
    if not 0 < delta < 1 or not 0 <= alpha < eps or not 0 < theta <= eps or n_candidates < 1:
        raise ValueError("need delta in (0, 1), 0 <= alpha < eps, 0 < theta <= eps, |X| >= 1")
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
