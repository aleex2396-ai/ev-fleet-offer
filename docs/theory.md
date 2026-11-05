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

## How programme (24) is solved

The conditions of Theorem 2 are Lemma 1 of the 2024 paper applied to each car, and Lemma 1 is
necessary and sufficient for γᵢ + ΓᵢU₀ ⊆ Uᵢ whatever half-space description is used for U₀
and Uᵢ. `offer.solve` therefore writes each set with irredundant rows (its power box and one
row on total energy, 2K + 1 rows instead of the 4K of H = (L, −L, I, −I)) and, for each row,
Lemma 1's multipliers as the dual of maximising that row over U₀. The feasible values of p,
α, γᵢ and Γᵢ, and so the optimal α, are those of (24) as printed; `tests/test_offer.py` solves
(24) as printed on small fleets and compares. Rows and columns that a car's set or U₀ fixes at
zero are dropped. The solver is HiGHS's interior-point method with crossover (through SciPy);
the allocation (31) is the optimal γᵢ and Γᵢ it returns. On made-up fleets of 40 cars over 40
steps the programme took about 10 seconds (computed).

**The car's vector in the base set.** Equation (11) averages the cars' vectors hᵢ, so it
depends on how each set is written, not only on the set. Here each car's power limit is the
tightest its set implies, min(c, e/δ) up to its deadline: no single step can take more than
the whole ceiling. With the cap alone, a car with no energy left that stays late would give U₀
charging capacity at hours no car can use, and the next statement would force α to 0.

## When the offer is zero

**Statement.** If at some step k the base set can charge (the k-th entry of its power vector
is positive) and every present car's set forces u_k = 0 (its deadline is before k, or its
ceiling is zero), the optimal α of (24) is 0.

*Proof.* For such a car, γᵢ + Γᵢu₀ must have k-th entry 0 for every u₀ in U₀. At u₀ = 0 this
gives (γᵢ)_k = 0; U₀ contains small multiples of the k-th unit vector, so (Γᵢ)_kk = 0. Then
the (k, k) entry of Σ Γᵢ = αI gives α = 0. ∎

The day-average base set of equation (11) cannot meet this case, since its power vector is
zero wherever no car can charge. A base set fixed from the training period can: on any day on
which no present car's learned set reaches a late step at which training cars could charge,
the offer is the single profile p. How often this happens is measured for every arm on
development data before the tag, beside the paper's own day-average base set.

## How a day's offer is checked

The nine profiles drawn from p + αU₀ (analysis plan, Section 3) are p + αu₀ for these points
u₀ of U₀ (`offer.checked_profiles`). For each checkpoint t (12:00, 14:00, 16:00, 18:00), many
points reach the most and the least cumulative energy by t; the pair checked moves as much
energy across t as it can: one charges as much as it can before t, earliest first, and nothing
after; the other charges nothing before t and as much as it can after, earliest first. The
ninth is the least-cost profile under a stylised time-of-use tariff (relative prices: 1 from
11:00 to 15:00, 2 from 10:00 to 11:00 and 15:00 to 16:00, 4 from 16:00 to 20:00) among the
points that deliver U₀'s largest total energy, cheapest steps first and earliest first among
equal prices; without that condition the least-cost point is zero, whose profile is p itself.

**Primary check.** Each car is given its allocation (31) for each profile and follows it as
far as its true set allows: no more than its true power limit at each step and its true
ceiling in total. The shortfall is the energy left over, summed over cars; the day counts as
delivered if, for all nine profiles, it is at most 10⁻⁴ kWh (`offer.DELIVERY_TOL_KWH`, a
rounding tolerance). **Secondary check.** The largest energy the cars actually present can
take, each within its true set, with their total power at no step above the profile's; the
shortfall is the profile's energy minus that. It is zero exactly when the profile can be split
among the cars, and it is never above the primary shortfall.

**When a day's outcome is known.** The outcome is defined within the horizon, but the energy a
car took between 10:00 and 20:00 is estimated from its recorded total (`docs/data.md`, check
D1), which is known only at unplug. So a day is passed to the day-level rules once its last
car has unplugged, at 20:00 at the earliest; a car that stays overnight delays it past the
next decision. The scramble test checks this ordering.

**The one-dimensional fallback** (used only if check D9 calls for it) is the segment b + a·v,
|a| ≤ A, with the shape v moving charging out of 10:00–11:00 (−1 kW per kW of amplitude) and
into 11:00–13:00 (+0.5). The reference b sums each car's flat half profile, half of
min(c, e/(δd)) up to its deadline, which lies inside its set with room either way. A linear
programme finds the largest A for which per-car shifts wᵢ, adding up to A·v, keep both
bᵢ + wᵢ and bᵢ − wᵢ in every car's set; by convexity every a in between is covered, and car i
is allocated bᵢ + (a/A)wᵢ. Its size is A, in kWh moved out of or into 10:00–11:00. It is
checked on its two ends and its reference, by the same two checks.

## The per-car models and the settings searched

Each model gives, for every car present at 10:00, lower quantiles of its remaining stay
(hours, at most the 10-hour horizon) and of the energy it will take before 20:00 (kWh), at the
16 levels 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
and 0.95 (`quantiles.LEVELS`). A level between two grid levels is read off by linear
interpolation; below 0.002 the bound is the trivial one (deadline at 10:00, energy zero).

**Families.** Gradient-boosted trees with pinball loss, one model per level and target
(scikit-learn's `HistGradientBoostingRegressor`, station as a categorical input; primary). A
network with one hidden layer (tanh) per target and one output per level, each output above
the one below by a softplus increment so that quantiles never cross, trained with Adam on the
mean pinball loss, the weights kept from the pass with the lowest loss on the last 15% of
training car-days in time order (second). Linear quantile regression, solved exactly by
linear programming with no penalty (baseline). Tree and linear quantiles that cross are
sorted, which cannot increase the pinball loss (Chernozhukov, Fernández-Val and Galichon,
2010). Inputs are listed in `features.py`; the baselines in `baselines.py`.

**Inputs.** The trees take the calendar month as a number and the station as a category. The
network and the linear model take the month as one indicator for each month present in their
training car-days, and the station the same way; a month or station absent from training gets
weight 1/k on each of the k seen, so the linear model predicts the average of their effects
(all zeros would leave that prediction undetermined, since the indicators of every training row
add up to the constant term). Their other inputs are standardised on the training car-days,
with missing values set to zero. A first run on the internal split gave these two models the
month as a number. With a 90-day window their training months are October to December, so the
month number rises steadily through training and then falls to 1 to 4 in the scored months, and
both models extended the trend into months they had never seen. With indicators the network's
mean pinball loss on the split (64 hidden units) fell from 0.523 to 0.474 hours for stay and
from 1.014 to 0.997 kWh for energy; the trees, which cannot extrapolate beyond the months they
split on, were unchanged. The selection was run again with indicators; no month after April
2019 had been scored when this was changed. That run gave unseen months and stations all zeros,
and under monthly refits, where the scored month is never among the training months of a 90-day
window, the linear model's predictions then depended on how each fit split the constant term
from the month effects: its mean pinball loss for stay was 0.81 hours on January to April 2019
against 0.53 in the single fit at 1 January. The run had scored May to August 2019. The
encoding above replaced it, and the search and refits were run again; the trees take neither
indicator and were unaffected.

**Refits.** At the first local midnight of each month every model is refitted on the car-days
whose outcome had been revealed by then, leaving out cars plugged in within 24 hours of the
refit (check D11) and car-days older than the history window; the rules that learn from whole
sessions see the sessions ended by then with the same exclusions. A model with fewer than 200
training car-days gives no bound.

**Scores.** A stay bound counts as missed if, rounded down to the 15-minute grid as the
engine uses it, it is later than the true deadline; an energy bound if it exceeds the energy
taken. Pinball loss is computed in hours against the deadline on the grid, and in kWh.

**Selection.** Settings are chosen on the internal split of the training period: each model
is fitted once at 1 January 2019 on what was known then and scored on the car-days of
January to April 2019. The score of a setting is its mean pinball loss for stay divided by
that of the population rule with all history on the same car-days, plus the same ratio for
energy; lower is better, and ties go to the earlier grid entry. The history window is chosen
first, on the population rule, whose only setting it is, and is then shared by every model
(analysis plan, Section 5: arms 4, 5 and 6 share it); the tree and network settings are then
chosen with that window. The mixture's number of components is chosen by the log-likelihood of
the sessions plugged in from January to April 2019, as Lee, Li and Low chose theirs by
held-out fit. The chosen settings are then frozen; they are refitted monthly through August
2019 and scored on the model-selection months, May to August 2019. The analysis plan allows
those months to choose settings as well (Section 4); for the per-car models they only score,
and they are left for the day-level settings fixed before the tag. No later month is scored
before the tag.

| Setting | Values searched | Fixed |
|---|---|---|
| History window (days) | all, 90, 180, 365 | |
| Trees: learning rate | 0.05, 0.1 | 200 iterations, no early stopping, seed 0 |
| Trees: leaves per tree | 7, 15, 31 | |
| Trees: smallest leaf (car-days) | 20, 100 | |
| Network: hidden units | 16, 64 | step 0.003, at most 300 passes, stop after 20 without gain, batches of 256, seed 0 |
| Network: weight decay | 0.0001, 0.01 | |
| Mixture: components | 2 to 8 | full covariance, 3 starts, seed 0; per-driver weights from 5 sessions |
| Stratified table | | cells of at least 30 car-days |
| Driver's own past | | at least 3 earlier sessions that lasted as long |
| Linear quantile regression | | no penalty |

**Chosen** (`fleetoffer percar` writes the whole search to `settings_search.csv`): a 90-day
history window (population score 1.970, against 2.000 with all history and 2.015 with 180 days;
all history and 365 days coincide here, since the data begin in April 2018); trees with
learning rate 0.1, 7 leaves and smallest leaf 20 (score 1.265; the twelve tree settings ranged
from 1.265 to 1.282); a network with 16 hidden units and weight decay 0.0001 (1.475; range
1.475 to 1.583); a mixture of 7 components. On the internal split the network's kept weights
came from within its first 12 passes for each target, and training all 300 passes keeps the
same weights: its loss on the held-back weeks never improves later. At this data size (1,931
training car-days) it is the weaker learned family.

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
| Each outcome known before the next decision (adaptive conformal) | A day's outcome uses each car's recorded energy total, known only at unplug (check D1's fallback), so a car that stays overnight delays it past the next decision | Not met on every day; the share of days whose outcome is known before the next decision is reported, and the rule is updated as outcomes arrive |

So the day-level rate in this project rests on a calibration rule over past days or on a
union bound across cars; the requirements of the 2022 theorem are reported beside the days
available, and no certificate from it is claimed.
