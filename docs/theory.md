# Theory: the statements the project reads against

This note states the published results behind the design exactly, with their numbering, the
one elementary proposition the project adds, and what each result requires set against what
the data provide. Numbers marked *computed* come from `src/fleetoffer/calibrate.py`, are
printed by `fleetoffer requirements`, and are checked in `tests/test_calibrate.py`.

Each paper's symbols are kept within its section. In particular α is the scaling factor of
programme (24) in the first two sections, the share of discarded samples in the 2022
theorem, and the target miss rate in Gibbs and Candès; γᵢ is a per-car translation in the
2024 paper and γ the step size of adaptive conformal inference.

## The per-car set

The decision is made at t₀ = 10:00 for the horizon 10:00 to 20:00, in K = 40 steps of
δ = 0.25 hours. For a car with deadline d (a number of steps, 0 ≤ d ≤ K), energy ceiling e
(kWh after 10:00) and power cap c (kW), its set of charging profiles u = (u₁, …, u_K) is

U(d, e) = { u : 0 ≤ u_k ≤ c for k ≤ d;  u_k = 0 for k > d;  δ (u₁ + … + u_K) ≤ e }.

Because u ≥ 0, the ceiling on the total also bounds the cumulative energy at every step. This
is a charging-only special case of the generalised battery model of Al Taha, Vincent and
Bitar (2024, equation (2)): no discharge, and an energy ceiling in place of the energy
requirement of their equation (7). The zero profile lies in every U(d, e), so "the fleet can
follow a profile" never means that each driver received the energy they needed.

The **true set** of a car present at 10:00 uses its actual unplug time (20:00 if it is still
plugged in then), rounded down to the 15-minute grid, and the energy it actually took
between 10:00 and 20:00. A **learned set** uses a deadline no later than the car is predicted
to leave (a lower quantile of remaining stay, rounded down to the grid) and an energy ceiling
no higher than it is predicted to take (a lower quantile of remaining energy), with the same
cap.

## Aggregation: Al Taha, Vincent and Bitar (2024)

Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate
flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4),
3142–3154, 2025 (online April 2024); arXiv 2207.07067.

- Each car's flexibility set is a polytope Uᵢ = {u : Hu ≤ hᵢ} (the generalised battery
  model, equations (2)–(3)); the fleet's aggregate flexibility is the Minkowski sum Σ Uᵢ,
  which is intractable to compute exactly in general.
- **Theorem 2.** For a base set U₀ = {u : Hu ≤ h₀}, the set p + P U₀ lies inside Σ Uᵢ if there
  exist γᵢ and Γᵢ (unrestricted) and Λᵢ ≥ 0 (entrywise) with [p, P] = Σ [γᵢ, Γᵢ]
  (equation (20)), ΛᵢH = HΓᵢ (22) and Λᵢh₀ ≤ hᵢ − Hγᵢ (23). The conditions are sufficient
  and, in the paper's own word, conservative. They are linear, so the inner approximation is
  a linear programme: programme (24) for the structure-preserving case P = αI, with α
  maximised.
- **Section IV** allocates an aggregate profile u = p + Pu₀ to car i as uᵢ = γᵢ + Γᵢu₀
  (equation (31)), an affine map obtained as a by-product of the programme, which the paper
  presents as the way to avoid solving a disaggregation linear programme.
- **Example 1** (equations (4)–(7)) builds a car's set with a power range (which may include
  discharge), a capacity ceiling and a lower energy limit in force after the deadline, the
  energy the driver requires (equation (7)). The sets above drop that requirement.
- The paper's base set (equation (11)) is the average of the present cars' vectors; any
  H-polytope base set is allowed. Here h₀ is fixed from the training period and is the same
  for every arm (analysis plan, Section 3); the paper's choice is a sensitivity.

## What this project adds: containment under one-sided bounds

**Proposition.** If d′ ≤ d and e′ ≤ e, then U(d′, e′) ⊆ U(d, e).

*Proof.* Take u in U(d′, e′). For k ≤ d′, 0 ≤ u_k ≤ c. For d′ < k ≤ d, u_k = 0, which lies in
[0, c]. For k > d ≥ d′, u_k = 0. And δ Σ u_k ≤ e′ ≤ e. So u is in U(d, e). ∎

**Consequences.** Write E for the event that, on a given day, every present car's learned
deadline is no later than its true one (both on the 15-minute grid, the learned one rounded
down) and its learned energy ceiling is no higher than the energy it took, with the same cap
in both sets. On E:

1. Each learned set lies inside the car's true set, and Minkowski sums preserve inclusion, so
   any inner approximation of the sum of learned sets lies inside the sum of true sets: every
   offered profile can be split among the cars actually present.
2. The Section IV allocation computed on the learned sets gives each car a profile in its
   learned set, hence in its true set: the allocation fixed at 10:00 also delivers.

By the union bound, Pr(E) ≥ 1 − (sum of the miss probabilities of the 2n bounds), with no
assumption about independence between cars. Beyond that, how often E holds is an empirical
quantity, measured by day. E is sufficient for delivery, not necessary.

The inclusion fails for an **energy requirement** (a lower limit: a smaller predicted
requirement gives a larger set, so containment would need an upper quantile) and for an
**exact energy target** (no one-sided bound gives containment). The proposition is
elementary; it is stated so that the project claims what the theorem gives and no more.

## The exact band, for measuring the inner approximation's slack

Cumulative energy by step t, C_t(u) = δ (u₁ + … + u_t), is linear in u. The maximum of a
linear function over a Minkowski sum is the sum of its maxima over the summands, and its
maximum over U(d, e) is min(e, c δ min(t, d)); its minimum is 0, at the zero profile. Each
set is convex, so every value between is reached. The fleet's exact band of cumulative energy
at step t is therefore

[ 0,  Σᵢ min(eᵢ, c δ min(t, dᵢ)) ].

For the offer p + αU₀ the band's width at step t is α (max over U₀ of C_t − min over U₀ of
C_t): for a fixed base set, the width is α times a constant. The offer's width and the exact
width are reported side by side at 14:00, 16:00 and 18:00, so the conservatism of the inner
approximation is measured. The exact aggregate set for this class of sets, not only its band,
is a generalised polymatroid (Mukhi, Loho and Abate, IEEE Transactions on Smart Grid 17(3),
2026).

## Day-level calibration under drift: Yan, Parise and Bitar (2022)

Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in
nonstationary environments", IEEE Control Systems Letters 6, 2022; arXiv 2205.03748,
Section III.

- **Assumptions.** Samples are independent, not necessarily identically distributed, and the
  1-Wasserstein distance between the distributions of samples i and i + k is at most ρ(k),
  for every i ≥ 1 and k ≥ 0, with ρ a known function and ρ(0) = 0 (condition (9)).
- **Method** ((10)–(12)). Each past sample is replaced by a norm ball around it; a decision is
  admissible if it satisfies the constraint on the whole ball for at least ⌈(1 − α)N⌉ of the
  N samples. With the radii of **Remark 4**, rᵢ = ρ(N + 1 − i)/θ, each ball is the drift
  budget at the sample's age divided by θ.
- **Theorem 4.** For a finite decision set X, the probability that the robust sampled
  solution set is not inside the true ε-feasible set is at most |X| · Ψ(αN; p₁, …, p_N), with
  pᵢ = (ε − ρ(N + 1 − i)/rᵢ)₊ and Ψ the distribution function of a sum of independent
  Bernoulli variables (Definition 2). With the radii of Remark 4, every pᵢ = ε − θ and Ψ is a
  binomial distribution function.
- **Corollary 1.** For δ in (0, 1), α in (0, ε) and θ in (0, ε − α),
  N ≥ ln(|X|/δ) / (2(ε − α − θ)²) gives confidence at least 1 − δ. It follows from Theorem 4
  through Hoeffding's inequality, so it is sufficient, not necessary.

**With days as samples** (computed; ε = 0.10 for a 90%-of-days target, δ = 0.05, X a grid of
101 scaling levels of the offer):

| α | θ | Each ball is the drift budget times | Theorem 4 bound first at most δ | Theorem 4 bound at most δ for every N from | Corollary 1 |
|---|---|---|---|---|---|
| 0.02 | 0.040 | 25 | 285 days | 312 days | 2,379 days (7.6 times) |
| 0.05 | 0.025 | 40 | 1,057 days | 1,121 days | 6,089 days (5.4 times) |

The bound is not monotone in N, because αN is rounded down; "for every N from" is the
reading that does not depend on a lucky N. The paper suggests no value for θ, α or δ.

## Adaptive conformal inference: Gibbs and Candès (2021)

Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS
2021; arXiv 2106.00170.

- **Update** (equation (2)): the working level moves as αₜ₊₁ = αₜ + γ(α − errₜ), where errₜ
  is 1 when the outcome falls outside the prediction set.
- **Proposition 4.1** (equation (5)): with probability one, for every T, the average miss
  rate over T steps is within (max{α₁, 1 − α₁} + γ)/(γT) of the target α. The paper states
  that this puts no constraints on the data-generating distribution.
- **What it assumes:** the stated recursion with an initial level in [0, 1], a positive step,
  a quantile rule extended to infinite endpoints, and each outcome known before the next
  prediction.

With days as steps the result certifies a long-run share of failed days, not a probability
for each day. Computed: starting at α₁ = 0.10, over the 182 weekdays of 2021 the bound is
within 5 points of 10% only if γ ≥ 0.9/8.1 ≈ 0.11; over 20 days no step reaches 5 points.

## Robust aggregation: Mukhi, Qu, You and Abate (2025)

Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility",
HSCC 2025; arXiv 2405.08232. The robust aggregate flexibility set at confidence 1 − β is the
set of aggregate profiles that the realised fleet can satisfy with probability at least
1 − β (Definition 2.3), under charging requirements drawn independently from one known
distribution. Theorem 4.4 gives a generalised polymatroid contained in that set, built as the
intersection of the aggregate sets of all populations within a Wasserstein ball that contains
the realised population with probability at least 1 − β (Lemma 4.3 and Theorem 4.2). The
paper names per-vehicle distributions as a limit of its independent-draws model. This
project's target differs: the realised day, with the distribution unknown, changing, and
sessions not independent, scored on nine profiles. It reports empirical delivery, and names
this result as the one that would apply if its assumptions held.

## What each day-level rule gives

| Rule | What it gives | Under which assumption |
|---|---|---|
| Union bound across the 2n per-car bounds, each at level 0.10/(2n) (0.00125 with 40 cars, computed) | Delivery on at least 90% of days, for each day | Every per-car bound valid at its level, given what is known at 10:00 |
| Adaptive conformal inference, days as steps | A long-run share of failed days within (max{α₁, 1 − α₁} + γ)/(γT) of 10% | None on the data; the day's outcome known before the next decision |
| Split conformal rule on a window of past day scores | At least 90% for the next day | Exchangeable days |
| Theorem 4 of Yan, Parise and Bitar, days as samples | At most δ chance that the kept decisions break the 90% target | Independent days, a known drift bound ρ on whole-day configurations, a fixed finite decision set, and the day counts above |

## The requirements against the data

| Requirement | What the data provide | Status |
|---|---|---|
| Independent days (2022 theorem; also the reading of a long-run share) | Cars on one day share the day's events, prices and calendar; drivers return | Not given; cars on one day are not treated as independent, and uncertainty is resampled by whole weeks |
| A known drift bound ρ on whole-day configurations (condition (9)) | A day is a varying number of cars, each with a deadline and an energy; the 2020 closure is a jump no earlier data can measure | Not given; a norm on that space would itself be a choice |
| Enough days (Theorem 4) | 483 weekdays in development (computed), so 312 is met and 1,121 missed by 638; a calendar year has 261 weekdays (computed), about 250 once holidays are removed; 182 in 2021 (computed) | Met at the first setting on development data only |
| A fixed finite decision set (Theorem 4) | The offer is fitted continuously from the day's cars | Not met; a grid of 101 scales is the closest reading |
| Exchangeable days (split conformal) | The closure and the 2021 reopening break exchangeability | Its 90% is an empirical target across the change |
| Each outcome known before the next decision (adaptive conformal) | A day's outcome is defined within the horizon, so it is known by 20:00 | Met by construction |

So the day-level rate in this project rests on a calibration rule over past days or on a
union bound across cars; the requirements of the 2022 theorem are reported beside the days
available, and no certificate from it is claimed.
