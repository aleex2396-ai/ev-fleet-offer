import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

from fleetoffer.offer import (
    CHECKPOINTS,
    DELIVERY_TOL_KWH,
    SHIFT_SHAPE,
    BaseSet,
    Fleet,
    check_delivery,
    checked_profiles,
    fixed_shortfall,
    hindsight_shortfall,
    solve,
    solve_segment,
)

DELTA = 0.25


def programme_24_as_printed(fleet: Fleet, base: BaseSet) -> float:
    """Programme (24) of Al Taha, Vincent and Bitar (2024) with their H = (L, -L, I, -I),
    h_i = (upper energy, -lower energy, upper power, -lower power) and Lambda_i of size 4T x 4T."""
    n, t = len(fleet), fleet.steps
    ell = np.tril(np.full((t, t), fleet.step_h))
    h = np.vstack([ell, -ell, np.eye(t), -np.eye(t)])
    zeros = np.zeros(t)
    h0 = np.concatenate([np.full(t, base.energy), zeros, base.power, zeros])
    his = [
        np.concatenate([np.full(t, e), zeros, up, zeros])
        for e, up in zip(fleet.energy, fleet.upper(), strict=True)
    ]
    m = 4 * t
    # Variables: alpha, p (t), then per car gamma (t), Gamma (t*t, row-major), Lambda (m*m).
    per_car = t + t * t + m * m
    nvar = 1 + t + n * per_car
    eq, eq_rhs, ub, ub_rhs = [], [], [], []

    def block(i):
        first = 1 + t + i * per_car
        return first, first + t, first + t + t * t

    # [p, alpha I] = sum_i [gamma_i, Gamma_i]   (20)
    a = sparse.lil_array((t + t * t, nvar))
    a[np.arange(t), 1 + np.arange(t)] = 1.0
    a[t + np.arange(t) * (t + 1), 0] = -1.0
    for i in range(n):
        g, big, _ = block(i)
        a[np.arange(t), g + np.arange(t)] = -1.0
        a[t + np.arange(t * t), big + np.arange(t * t)] = 1.0
    eq.append(a.tocsr())
    eq_rhs.append(np.zeros(t + t * t))
    for i in range(n):
        g, big, lam = block(i)
        # Lambda_i H = H Gamma_i   (22); row-major vec:
        # kron(I, H^T) vec(Lambda) - kron(H, I) vec(Gamma) = 0
        a = sparse.lil_array((m * t, nvar))
        a[:, lam : lam + m * m] = sparse.kron(sparse.eye_array(m), h.T)
        a[:, big : big + t * t] = -sparse.kron(h, sparse.eye_array(t))
        eq.append(a.tocsr())
        eq_rhs.append(np.zeros(m * t))
        # Lambda_i h0 + H gamma_i <= h_i   (23)
        a = sparse.lil_array((m, nvar))
        a[:, lam : lam + m * m] = sparse.kron(sparse.eye_array(m), h0[None, :])
        a[:, g : g + t] = h
        ub.append(a.tocsr())
        ub_rhs.append(his[i])
    bounds = [(0, None)] + [(None, None)] * t  # alpha >= 0 in place of (24)'s alpha > 0
    for _ in range(n):
        bounds += [(None, None)] * (t + t * t) + [(0, None)] * (m * m)  # Lambda_i >= 0   (21)
    c = np.zeros(nvar)
    c[0] = -1.0
    result = linprog(
        c,
        A_ub=sparse.vstack(ub),
        b_ub=np.concatenate(ub_rhs),
        A_eq=sparse.vstack(eq),
        b_eq=np.concatenate(eq_rhs),
        bounds=bounds,
        method="highs",
    )
    assert result.status == 0, result.message
    return float(result.x[0])


def random_fleet(rng, n, steps):
    return Fleet.of(rng.integers(0, steps + 1, n), rng.uniform(0.0, 6.0, n), steps=steps)


def point_of(base: BaseSet, rng) -> np.ndarray:
    u0 = rng.uniform(0.0, 1.0, len(base.power)) * base.power
    total = base.step_h * u0.sum()
    return u0 * min(1.0, base.energy / total) if total > 0 else u0


def test_identical_cars_give_alpha_equal_to_their_number():
    # With identical sets U_i = U0, the aggregate set is exactly N U0 (the paper's (12)).
    fleet = Fleet.of([20] * 4, [8.0] * 4)
    offer = solve(fleet, BaseSet.average(fleet))
    assert offer.alpha == pytest.approx(4.0, abs=1e-6)
    assert offer.size() == pytest.approx(np.mean([fleet.band_width(t) for t in (16, 24, 32)]))


@pytest.mark.parametrize("seed", range(6))
def test_reduced_programme_matches_programme_24_as_printed(seed):
    rng = np.random.default_rng(seed)
    steps = 5
    fleet = random_fleet(rng, 3, steps)
    if fleet.deadline.max() == 0:
        fleet = Fleet.of(fleet.deadline + 2, fleet.energy, steps=steps)
    for base in (BaseSet.average(fleet), BaseSet.average(random_fleet(rng, 4, steps))):
        if base.is_trivial():
            continue
        assert solve(fleet, base).alpha == pytest.approx(
            programme_24_as_printed(fleet, base), abs=1e-6
        )


def test_a_step_the_base_set_needs_but_no_car_covers_makes_alpha_zero():
    steps = 5
    base = BaseSet(np.full(steps, 2.0), 3.0)  # can charge at every step
    fleet = Fleet.of([3, 4], [5.0, 5.0], steps=steps)  # nobody can charge at step 5
    assert solve(fleet, base).alpha == pytest.approx(0.0, abs=1e-9)
    assert programme_24_as_printed(fleet, base) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_every_point_of_the_offer_is_split_among_the_cars_by_the_affine_map(seed):
    rng = np.random.default_rng(seed)
    fleet = random_fleet(rng, 6, 40)
    base = BaseSet.average(fleet)
    offer = solve(fleet, base)
    upper = fleet.upper()
    for u0 in [point_of(base, rng) for _ in range(30)] + list(checked_profiles(base).values()):
        cars = offer.allocate(u0)
        assert cars.sum(axis=0) == pytest.approx(offer.profile(u0), abs=1e-6)
        assert (cars >= -1e-6).all() and (cars <= upper + 1e-6).all()
        assert (DELTA * cars.sum(axis=1) <= fleet.energy + 1e-6).all()
    for t in range(1, 41):
        assert offer.alpha * base.width(t) <= fleet.band_width(t) + 1e-6  # inner approximation


def test_exact_band_width_is_the_sum_of_each_cars_largest_cumulative_energy():
    rng = np.random.default_rng(4)
    fleet = random_fleet(rng, 5, 40)
    for t in (1, 8, 16, 24, 40):
        best = 0.0
        for d, e, c in zip(fleet.deadline, fleet.energy, fleet.cap, strict=True):
            r = linprog(
                -DELTA * (np.arange(40) < t),
                A_ub=np.full((1, 40), DELTA),
                b_ub=[e],
                bounds=[(0, c if k < d else 0) for k in range(40)],
                method="highs",
            )
            best -= r.fun
        assert fleet.band_width(t) == pytest.approx(best, abs=1e-9)


def test_the_nine_profiles_reach_the_extremes_at_their_checkpoints():
    base = BaseSet(np.linspace(1.0, 3.0, 40), 5.0)  # less than 11:00-15:00 can take
    profiles = checked_profiles(base)
    assert len(profiles) == 9
    for u0 in profiles.values():
        assert (u0 >= 0).all() and (u0 <= base.power + 1e-12).all()
        assert DELTA * u0.sum() <= base.energy + 1e-12
    for t in CHECKPOINTS:
        assert DELTA * profiles[f"most_by_step_{t}"][:t].sum() == pytest.approx(base.width(t))
        assert profiles[f"most_by_step_{t}"][t:].sum() == 0.0
        assert profiles[f"least_by_step_{t}"][:t].sum() == 0.0
    cheapest = profiles["least_cost"]
    assert DELTA * cheapest.sum() == pytest.approx(5.0)
    assert cheapest[4:20].sum() > 0 and cheapest[:4].sum() == 0  # the midday low first


def test_fixed_shortfall_counts_what_the_true_set_cannot_take():
    true = Fleet.of([4, 8], [1.0, 10.0], cap=4.0, steps=8)
    allocation = np.array([[2.0] * 6 + [0, 0], [4.0] * 8])
    # car 1: asks 3 kWh; after its deadline (step 4) 1 kWh is lost; its ceiling allows 1 kWh.
    # car 2: asks 8 kWh within its set.
    assert fixed_shortfall(allocation, true) == pytest.approx([2.0, 0.0])


def test_hindsight_split_moves_energy_between_cars_but_not_past_their_limits():
    true = Fleet.of([2, 4], [2.0, 2.0], cap=4.0, steps=4)
    assert hindsight_shortfall(np.array([4.0, 4.0, 0.0, 0.0]), true) == pytest.approx(0.0)
    # 12 kW at step 1 is more than the two caps together: 4 kW for 0.25 h is missing.
    assert hindsight_shortfall(np.array([12.0, 0.0, 0.0, 0.0]), true) == pytest.approx(1.0)
    # 4 kWh in total is the most the two ceilings allow.
    assert hindsight_shortfall(np.full(4, 6.0), true) == pytest.approx(6.0 - 4.0)


def test_learned_sets_equal_to_the_true_sets_are_always_delivered():
    rng = np.random.default_rng(7)
    fleet = random_fleet(rng, 8, 40)
    delivery = check_delivery(solve(fleet, BaseSet.average(fleet)), fleet)
    assert delivery.fixed_delivered and delivery.hindsight_delivered
    assert len(delivery.fixed_kwh) == 9


def test_a_failed_deadline_bound_shows_as_a_shortfall_and_hindsight_is_an_upper_bound():
    learned = Fleet.of([40, 40, 40], [5.0, 5.0, 5.0])
    true = Fleet.of([40, 40, 8], [5.0, 5.0, 5.0])  # the third car left at 12:00
    delivery = check_delivery(solve(learned, BaseSet.average(learned)), true)
    assert not delivery.fixed_delivered
    for name, short in delivery.fixed_kwh.items():
        assert delivery.hindsight_kwh[name] <= short + DELIVERY_TOL_KWH


def test_the_fallback_segment_lies_in_every_cars_set_and_sums_to_the_shape():
    rng = np.random.default_rng(11)
    fleet = random_fleet(rng, 6, 40)
    fleet = Fleet.of(np.maximum(fleet.deadline, 12), fleet.energy + 1.0)
    segment = solve_segment(fleet)
    assert segment.amplitude > 0
    upper = fleet.upper()
    for profile, cars in segment.checked().values():
        assert cars.sum(axis=0) == pytest.approx(profile, abs=1e-6)
        assert (cars >= -1e-6).all() and (cars <= upper + 1e-6).all()
        assert (DELTA * cars.sum(axis=1) <= fleet.energy + 1e-6).all()
    ends = segment.checked()
    moved = ends["reference"][0] - ends["shift_out"][0]
    assert moved == pytest.approx(segment.amplitude * -SHIFT_SHAPE, abs=1e-6)
    assert check_delivery(segment, fleet).fixed_delivered


def test_the_fallback_amplitude_is_limited_by_the_caps():
    # One car, cap 4 kW, flat half profile 2 kW all day: it can move at most 2 kW either way
    # in 10:00-11:00 and, at half that per step, 4 kW in 11:00-13:00; so A = 2.
    fleet = Fleet.of([40], [100.0], cap=4.0)
    assert solve_segment(fleet).amplitude == pytest.approx(2.0, abs=1e-6)


def test_a_car_with_no_energy_left_adds_no_capacity_to_the_average_base_set():
    # The second car stays all day but can take nothing; the first leaves at 14:00.
    fleet = Fleet.of([16, 40], [8.0, 0.0])
    base = BaseSet.average(fleet)
    assert (base.power[16:] == 0).all()
    assert solve(fleet, base).alpha > 0
