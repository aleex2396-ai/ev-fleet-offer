"""The fleet's offer at the decision time, its allocation to cars, and whether it was delivered.

Notation follows `docs/theory.md`. The horizon has K steps of delta hours (40 steps of 0.25 h,
10:00 to 20:00). Steps are counted from 1 in the text and from 0 in the code. A car's set is

    U(d, e) = { u : 0 <= u_k <= c for k <= d;  u_k = 0 for k > d;  delta * sum(u) <= e },

with deadline d (a number of steps), energy ceiling e (kWh) and power cap c (kW).

The inner approximation is programme (24) of Al Taha, Vincent and Bitar (2025): maximise
alpha subject to the conditions of their Theorem 2 with P = alpha * I. Those conditions are
Lemma 1 of the paper applied to each car, and Lemma 1 is necessary and sufficient for
gamma_i + Gamma_i U0 to lie in U_i whatever half-space description is used for the two sets.
So the programme is written here with irredundant descriptions (a box and one energy row for
each set, instead of the 4K rows of their H), which leaves the feasible values of p, alpha,
gamma_i and Gamma_i, and hence the optimal alpha, exactly as in (24) and makes the programme
about five times smaller. Lemma 1 assumes U0 nonempty; it always holds the zero profile. (24)
asks alpha > 0; here alpha >= 0, so a day on which nothing can be offered gives alpha = 0
rather than no solution. `tests/test_offer.py` solves (24) as printed, with the same
relaxation, on small fleets and checks that the two give the same alpha.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog

STEPS = 40  # K: 10:00 to 20:00
STEP_H = 0.25  # delta, hours
CAP_KW = 6.7  # c: the primary assumption of the analysis plan (Section 3)
CHECKPOINTS = (8, 16, 24, 32)  # cumulative energy after 12:00, 14:00, 16:00 and 18:00
SIZE_CHECKPOINTS = (16, 24, 32)  # the offer's size is its band width at 14:00, 16:00, 18:00
# A stylised time-of-use tariff, relative prices per step: a midday low 11:00-15:00, shoulders
# 10:00-11:00 and 15:00-16:00, an evening peak 16:00-20:00. Only the order of prices matters.
TARIFF = np.array([2.0] * 4 + [1.0] * 16 + [2.0] * 4 + [4.0] * 16)
DELIVERY_TOL_KWH = 1e-4  # a profile counts as delivered if at most this much is missing
TRIVIAL_BASE_KWH = 1e-7  # a base set with a smaller energy ceiling is treated as empty
# Programme (24) is solved by HiGHS's interior-point method with crossover to a vertex; on
# made-up fleets of 20 to 40 cars it is five times faster than its simplex method and gives
# the same alpha. Which optimal gamma_i and Gamma_i it returns fixes the allocation of (31).
# On a rare badly scaled day the interior-point method stops without an answer (HiGHS model
# status unknown); the same programme is then solved by the dual simplex method, and the
# offer's status says so.
SOLVER = (
    f"HiGHS (scipy {scipy.__version__} linprog), interior point with crossover, default tolerances;"
    " dual simplex if the interior-point method returns no answer"
)


@dataclass(frozen=True)
class Fleet:
    """The cars of one day: deadline (steps), energy ceiling (kWh), power cap (kW)."""

    deadline: np.ndarray
    energy: np.ndarray
    cap: np.ndarray
    steps: int = STEPS
    step_h: float = STEP_H

    @classmethod
    def of(cls, deadline, energy, cap=CAP_KW, steps=STEPS, step_h=STEP_H) -> Fleet:
        deadline = np.asarray(deadline, dtype=int).reshape(-1)
        energy = np.asarray(energy, dtype=float).reshape(-1)
        cap = np.broadcast_to(np.asarray(cap, dtype=float), deadline.shape).copy()
        if deadline.shape != energy.shape:
            raise ValueError("need one deadline and one energy ceiling per car")
        if ((deadline < 0) | (deadline > steps)).any():
            raise ValueError(f"deadlines must be whole steps from 0 to {steps}")
        if (energy < 0).any() or (cap < 0).any() or not np.isfinite(energy).all():
            raise ValueError("energy ceilings and caps must be finite and non-negative")
        return cls(deadline, energy, cap, steps, step_h)

    def __len__(self) -> int:
        return len(self.deadline)

    def upper(self) -> np.ndarray:
        """Each car's power limit at each step (cars by steps), up to its deadline.

        The limit is min(c_i, e_i / delta), the tightest the set implies: no step can take
        more than the whole ceiling. This describes the same set as the cap alone, so it
        changes nothing in programme (24) or in the checks, but it is the vector averaged in
        the base set (11): with the cap alone, a car with no energy left that stays late
        would give the base set charging capacity at hours no car can use, and alpha would be
        0 (docs/theory.md).
        """
        steps = np.arange(self.steps)
        limit = np.minimum(self.cap, self.energy / self.step_h)
        return np.where(steps[None, :] < self.deadline[:, None], limit[:, None], 0.0)

    def band_width(self, t: int) -> float:
        """Exact width of the fleet's cumulative-energy band after step t (docs/theory.md):
        sum over cars of min(e_i, c_i delta min(t, d_i)); the band's lower end is 0."""
        reach = self.cap * self.step_h * np.minimum(t, self.deadline)
        return float(np.minimum(self.energy, reach).sum())


@dataclass(frozen=True)
class BaseSet:
    """U0 = {u : 0 <= u <= power, delta * sum(u) <= energy}, the form of equation (11).

    With every car's set in the generalised battery model of equation (2) (power between 0 and
    its limit, net energy between 0 and its ceiling at every step), the average of the cars'
    vectors h_i is a power limit equal to the average limit and an energy ceiling equal to the
    average ceiling; because power is non-negative, the ceiling binds only on the total.
    """

    power: np.ndarray
    energy: float
    step_h: float = STEP_H

    @classmethod
    def average(cls, fleet: Fleet) -> BaseSet:
        """Equation (11): the average of the given cars' vectors."""
        if len(fleet) == 0:
            raise ValueError("the base set needs at least one car")
        return cls(fleet.upper().mean(axis=0), float(fleet.energy.mean()), fleet.step_h)

    def is_trivial(self) -> bool:
        """True if U0 holds only the zero profile, so no scaling of it offers anything. An
        energy ceiling of at most `TRIVIAL_BASE_KWH` counts as zero: such a base set offers
        less than the delivery tolerance at any scale the programme returns, and the solver
        reports the programme on it as unbounded."""
        return self.energy <= TRIVIAL_BASE_KWH or not (self.power > 0).any()

    def width(self, t: int) -> float:
        """Max minus min over U0 of the cumulative energy after step t."""
        return float(min(self.energy, self.step_h * self.power[:t].sum()))

    def fill(self, order: np.ndarray, skip: int = 0) -> np.ndarray:
        """The profile of U0 that charges at full power in the given order of steps until the
        energy runs out, leaving the first `skip` steps at zero."""
        u = np.zeros_like(self.power)
        left = self.energy
        for k in order:
            if k < skip or left <= 0:
                continue
            u[k] = min(self.power[k], left / self.step_h)
            left -= u[k] * self.step_h
        return u


@dataclass(frozen=True)
class Offer:
    """p + alpha U0, with the per-car maps gamma_i + Gamma_i u0 of Section IV."""

    alpha: float
    p: np.ndarray  # (K,)
    gamma: np.ndarray  # (cars, K)
    big_gamma: np.ndarray  # (cars, K, K)
    base: BaseSet
    seconds: float
    status: str

    def size(self, checkpoints=SIZE_CHECKPOINTS) -> float:
        """Average band width of the offer at the checkpoints, kWh."""
        return float(np.mean([self.alpha * self.base.width(t) for t in checkpoints]))

    def profile(self, u0: np.ndarray) -> np.ndarray:
        return self.p + self.alpha * u0

    def allocate(self, u0: np.ndarray) -> np.ndarray:
        """Equation (31): car i's profile gamma_i + Gamma_i u0 (cars by steps)."""
        return self.gamma + self.big_gamma @ u0

    def checked(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The nine checked profiles, each with its allocation to the cars."""
        return {
            name: (self.profile(u0), self.allocate(u0))
            for name, u0 in checked_profiles(self.base).items()
        }

    def digest(self) -> str:
        """A fingerprint of everything the programme returned, to compare two runs exactly."""
        parts = [np.atleast_1d(self.alpha), self.p, self.gamma.ravel(), self.big_gamma.ravel()]
        return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest()


def solve(fleet: Fleet, base: BaseSet) -> Offer:
    """Programme (24) on the given sets, in the irredundant form described above."""
    start = time.perf_counter()
    n, steps = len(fleet), fleet.steps
    empty = Offer(
        0.0, np.zeros(steps), np.zeros((n, steps)), np.zeros((n, steps, steps)), base, 0.0, ""
    )
    if base.is_trivial() or n == 0:
        status = "trivial base set" if base.is_trivial() else "no cars"
        return _replace(empty, seconds=time.perf_counter() - start, status=status)

    cols = np.flatnonzero(base.power > 0)  # J: steps where U0 can charge
    q, delta = len(cols), fleet.step_h
    rows_ub, cols_ub, vals_ub, rhs_ub = [], [], [], []
    lower, upper = [0.0], [np.inf]  # alpha >= 0
    layout = []  # per car: (first gamma column, first Gamma column, m)
    nvar, nrow = 1, 0

    def add_var(count: int, lo: float, hi: float) -> int:
        nonlocal nvar
        first = nvar
        nvar += count
        lower.extend([lo] * count)
        upper.extend([hi] * count)
        return first

    def support_at_most(terms_g: list[list[tuple[int, float]]], terms_rhs, rhs: float) -> None:
        """sigma_U0(g) + (terms_rhs . x) <= rhs, where g_b = sum of terms_g[b]; by Lemma 1's
        dual: there are w >= 0 (one per column of J) and s >= 0 with w >= g - s delta and
        power_J . w + energy s + (terms_rhs . x) <= rhs."""
        nonlocal nrow
        w0 = add_var(q, 0.0, np.inf)
        s = add_var(1, 0.0, np.inf)
        for b in range(q):  # g_b - w_b - delta s <= 0
            for col, val in terms_g[b]:
                rows_ub.append(nrow)
                cols_ub.append(col)
                vals_ub.append(val)
            rows_ub.extend([nrow, nrow])
            cols_ub.extend([w0 + b, s])
            vals_ub.extend([-1.0, -delta])
            rhs_ub.append(0.0)
            nrow += 1
        for b in range(q):
            rows_ub.append(nrow)
            cols_ub.append(w0 + b)
            vals_ub.append(base.power[cols[b]])
        rows_ub.append(nrow)
        cols_ub.append(s)
        vals_ub.append(base.energy)
        for col, val in terms_rhs:
            rows_ub.append(nrow)
            cols_ub.append(col)
            vals_ub.append(val)
        rhs_ub.append(rhs)
        nrow += 1

    for i in range(n):
        m = int(fleet.deadline[i])  # the car can charge in steps 0 .. m-1
        g0 = add_var(m, -np.inf, np.inf)
        big0 = add_var(m * q, -np.inf, np.inf)  # Gamma_i[a, cols[b]] at big0 + a q + b
        layout.append((g0, big0, m))
        for a in range(m):
            row = [[(big0 + a * q + b, 1.0)] for b in range(q)]
            support_at_most(row, [(g0 + a, 1.0)], fleet.cap[i])  # u_a <= c_i
            neg = [[(big0 + a * q + b, -1.0)] for b in range(q)]
            support_at_most(neg, [(g0 + a, -1.0)], 0.0)  # -u_a <= 0
        energy_g = [[(big0 + a * q + b, delta) for a in range(m)] for b in range(q)]
        support_at_most(energy_g, [(g0 + a, delta) for a in range(m)], fleet.energy[i])

    # sum_i Gamma_i = alpha I, on the columns J (other columns of U0 are always zero).
    rows_eq, cols_eq, vals_eq, neq = [], [], [], 0
    for k in range(steps):
        cars = [i for i in range(n) if layout[i][2] > k]
        for b, j in enumerate(cols):
            if not cars and k != j:
                continue
            for i in cars:
                rows_eq.append(neq)
                cols_eq.append(layout[i][1] + k * q + b)
                vals_eq.append(1.0)
            if k == j:
                rows_eq.append(neq)
                cols_eq.append(0)
                vals_eq.append(-1.0)
            neq += 1

    objective = np.zeros(nvar)
    objective[0] = -1.0
    a_ub = sparse.csr_array((vals_ub, (rows_ub, cols_ub)), shape=(nrow, nvar))
    a_eq = sparse.csr_array((vals_eq, (rows_eq, cols_eq)), shape=(neq, nvar)) if neq else None
    problem = {
        "A_ub": a_ub,
        "b_ub": np.asarray(rhs_ub),
        "A_eq": a_eq,
        "b_eq": np.zeros(neq) if neq else None,
        "bounds": np.column_stack([lower, upper]),
    }
    status = "optimal"
    result = linprog(objective, method="highs-ipm", **problem)
    if result.status != 0:
        result = linprog(objective, method="highs-ds", **problem)
        status = "optimal (dual simplex)"
    if result.status != 0:
        raise RuntimeError(f"programme (24) did not solve: {result.message}")

    x = result.x
    gamma = np.zeros((n, steps))
    big_gamma = np.zeros((n, steps, steps))
    for i, (g0, big0, m) in enumerate(layout):
        gamma[i, :m] = x[g0 : g0 + m]
        big_gamma[i, :m, cols] = x[big0 : big0 + m * q].reshape(m, q).T
    return Offer(
        alpha=float(x[0]),
        p=gamma.sum(axis=0),
        gamma=gamma,
        big_gamma=big_gamma,
        base=base,
        seconds=time.perf_counter() - start,
        status=status,
    )


def _replace(offer: Offer, **changes) -> Offer:
    return Offer(**{**offer.__dict__, **changes})


def checked_profiles(
    base: BaseSet, checkpoints=CHECKPOINTS, tariff=TARIFF
) -> dict[str, np.ndarray]:
    """The nine points u0 of U0 whose profiles p + alpha u0 are checked (analysis plan,
    Section 3).

    For each checkpoint t, a profile with the most cumulative energy by step t and one with
    the least. Many profiles reach each extreme; the tie is broken so that the pair moves as
    much energy across t as possible: the first charges as much as it can before t, earliest
    first, and nothing after; the second charges nothing before t and as much as it can
    after, earliest first. The ninth delivers U0's largest total energy at least cost under
    the tariff, cheapest steps first and earliest first among equal prices (without the
    condition on total energy the least-cost point of U0 is zero, whose profile is p).
    """
    steps = np.arange(len(base.power))
    out = {}
    for t in checkpoints:
        out[f"most_by_step_{t}"] = base.fill(steps[:t])
        out[f"least_by_step_{t}"] = base.fill(steps, skip=t)
    out["least_cost"] = base.fill(np.argsort(tariff[: len(steps)], kind="stable"))
    return out


def fixed_shortfall(allocation: np.ndarray, true: Fleet) -> np.ndarray:
    """Energy each car cannot take of the profile allocated to it at 10:00 (kWh).

    The car follows its allocation as far as its true set allows: at each step no more than
    its true power limit, and in total no more than its true ceiling. What is left over is
    the shortfall. It is zero exactly when the allocation lies in the true set (to rounding).
    """
    delta = true.step_h
    asked = delta * allocation.sum(axis=1)
    can = np.minimum(
        true.energy, delta * np.minimum(np.maximum(allocation, 0), true.upper()).sum(1)
    )
    return np.maximum(asked - can, 0.0)


def hindsight_shortfall(profile: np.ndarray, true: Fleet) -> float:
    """Energy of the profile that no split among the cars actually present could deliver.

    The largest energy the cars can take with, at every step, their total power no more than
    the profile's, each car within its true set. The profile can be split exactly among the
    cars if and only if that largest energy equals the profile's energy. This uses the true
    sets, which no aggregator knows at 10:00, so it is an upper bound on delivery.
    """
    n, steps = len(true), true.steps
    delta = true.step_h
    if n == 0:
        return float(delta * np.maximum(profile, 0).sum())
    upper = true.upper().reshape(-1)
    by_step = sparse.kron(sparse.csr_array(np.ones((1, n))), sparse.eye_array(steps))
    by_car = sparse.kron(sparse.eye_array(n), sparse.csr_array(np.full((1, steps), delta)))
    result = linprog(
        -delta * np.ones(n * steps),
        A_ub=sparse.vstack([by_step, by_car]).tocsr(),
        b_ub=np.concatenate([np.maximum(profile, 0), true.energy]),
        bounds=np.column_stack([np.zeros(n * steps), upper]),
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"the hindsight split did not solve: {result.message}")
    return float(max(delta * profile.sum() + result.fun, 0.0))


@dataclass(frozen=True)
class Delivery:
    """The two checks of one day's offer against the cars' true sets."""

    fixed_kwh: dict[str, float]  # shortfall of the fixed allocation, per checked profile
    hindsight_kwh: dict[str, float]  # shortfall of the best split in hindsight, per profile
    energy_kwh: dict[str, float]  # energy of each checked profile

    @property
    def fixed_delivered(self) -> bool:
        return max(self.fixed_kwh.values()) <= DELIVERY_TOL_KWH

    @property
    def hindsight_delivered(self) -> bool:
        return max(self.hindsight_kwh.values()) <= DELIVERY_TOL_KWH


def check_delivery(offer: Offer | SegmentOffer, true: Fleet) -> Delivery:
    """The primary and secondary checks of each checked profile (analysis plan, Section 3)."""
    fixed, hindsight, energy = {}, {}, {}
    for name, (profile, allocation) in offer.checked().items():
        fixed[name] = float(fixed_shortfall(allocation, true).sum())
        hindsight[name] = hindsight_shortfall(profile, true)
        energy[name] = float(true.step_h * profile.sum())
    return Delivery(fixed, hindsight, energy)


# The one-dimensional fallback offer (analysis plan, Section 3; used only if check D9 says so).
SHIFT_SHAPE = np.array([-1.0] * 4 + [0.5] * 8 + [0.0] * 28)  # kW per kW of amplitude


@dataclass(frozen=True)
class SegmentOffer:
    """The segment {b + a v : -A <= a <= A}, allocated as b_i + (a / A) w_i.

    The shape v moves charging out of 10:00-11:00 and into 11:00-13:00 (it sums to zero).
    The reference b is the sum of each car's flat half profile, b_ik = min(c_i, e_i /
    (delta d_i)) / 2 up to its deadline, which sits inside its set with room to move either
    way. A, in kW, is the largest amplitude for which some w_i keep b_i + w_i and b_i - w_i in
    every car's set while the w_i add up to A v; by convexity every a between is then covered.
    Its size is A: the energy, in kWh, that the offer can move out of 10:00-11:00 or into it.
    """

    amplitude: float
    b: np.ndarray  # (cars, K)
    w: np.ndarray  # (cars, K)
    seconds: float

    def size(self) -> float:
        return float(self.amplitude)

    def checked(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for name, sign in (("shift_out", 1.0), ("reference", 0.0), ("shift_in", -1.0)):
            cars = self.b + sign * self.w
            out[name] = (cars.sum(axis=0), cars)
        return out


def solve_segment(fleet: Fleet, shape: np.ndarray = SHIFT_SHAPE) -> SegmentOffer:
    """The largest amplitude A of the one-dimensional fallback, by a linear programme."""
    start = time.perf_counter()
    n, steps, delta = len(fleet), fleet.steps, fleet.step_h
    upper = fleet.upper()
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = np.where(fleet.deadline > 0, fleet.energy / (delta * fleet.deadline), 0.0)
    b = np.where(upper > 0, np.minimum(fleet.cap, flat)[:, None] / 2.0, 0.0)
    if n == 0:
        return SegmentOffer(0.0, b, np.zeros_like(b), time.perf_counter() - start)
    # Variables: A, then w_i (K each). Rows: both ends inside each car's box and energy ceiling.
    nvar = 1 + n * steps
    eye = sparse.eye_array(n * steps)
    rows = [
        sparse.hstack([sparse.csr_array((n * steps, 1)), eye]),  # b + w <= upper
        sparse.hstack([sparse.csr_array((n * steps, 1)), -eye]),  # b - w <= upper
        sparse.hstack([sparse.csr_array((n * steps, 1)), -eye]),  # -(b + w) <= 0
        sparse.hstack([sparse.csr_array((n * steps, 1)), eye]),  # -(b - w) <= 0
    ]
    per_car = sparse.kron(sparse.eye_array(n), sparse.csr_array(np.full((1, steps), delta)))
    rows += [
        sparse.hstack([sparse.csr_array((n, 1)), per_car]),  # energy of b + w
        sparse.hstack([sparse.csr_array((n, 1)), -per_car]),  # energy of b - w
    ]
    flat_b, flat_u = b.reshape(-1), upper.reshape(-1)
    energy_b = delta * b.sum(axis=1)
    rhs = [flat_u - flat_b, flat_u - flat_b, flat_b, flat_b]
    rhs += [fleet.energy - energy_b, fleet.energy - energy_b]
    coupling = sparse.hstack(
        [
            sparse.csr_array(-shape.reshape(-1, 1)),
            sparse.kron(sparse.csr_array(np.ones((1, n))), sparse.eye_array(steps)),
        ]
    )
    objective = np.zeros(nvar)
    objective[0] = -1.0
    result = linprog(
        objective,
        A_ub=sparse.vstack(rows).tocsr(),
        b_ub=np.concatenate(rhs),
        A_eq=coupling.tocsr(),
        b_eq=np.zeros(steps),
        bounds=[(0, None)] + [(None, None)] * (n * steps),
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"the fallback programme did not solve: {result.message}")
    w = result.x[1:].reshape(n, steps)
    return SegmentOffer(float(result.x[0]), b, w, time.perf_counter() - start)
