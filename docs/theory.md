# Theory: the statements the project reads against

This note states the published results behind the design exactly, with their numbering, the
one elementary proposition the project adds, and what each result requires set against what
the data provide. Numbers marked *computed* come from `src/fleetoffer/calibrate.py`, are
printed by `fleetoffer requirements`, and are checked in `tests/test_calibrate.py`.

Each paper's symbols are kept within its section. In particular α is the scaling factor of
programme (24) in the first two sections, the share of discarded samples in the 2022
theorem, and the target miss rate in Gibbs and Candès; γᵢ is a per-car translation in the
2024 paper, γ the step size of adaptive conformal inference, and, in Theorem 5 of the 2022
paper, the margin by which its constraint is tightened.

## The per-car set

The decision is made at t₀ = 10:00 for the horizon 10:00 to 20:00, in K = 40 steps of
δ = 0.25 hours. For a car with deadline d (a number of steps, 0 ≤ d ≤ K), energy ceiling e
(kWh after 10:00) and power cap c (kW), its set of charging profiles u = (u₁, …, u_K) is

U(d, e) = { u : 0 ≤ u_k ≤ c for k ≤ d;  u_k = 0 for k > d;  δ (u₁ + … + u_K) ≤ e }.

Because u ≥ 0, the ceiling on the total also bounds the cumulative energy at every step. This
is a charging-only special case of the generalised battery model of Al Taha, Vincent and
Bitar (2024, equation (2)), built from their Example 1 with three changes: no discharge; the
energy ceiling as the upper limit on cumulative energy, in place of the battery's free capacity
(their equation (6)); and no lower limit, in place of their equation (7), which from the
deadline on requires the driver's desired final state of charge. The zero profile lies in every U(d, e), so "the fleet can
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
- **Example 1** (equations (4)–(7)) builds a car's set from a power range (which may include
  discharge), an upper limit on cumulative energy, the battery's free capacity (equation (6)),
  and a lower limit (equation (7)) that from the deadline on is the driver's desired final
  state of charge less the initial one. The sets above replace the upper limit with the energy
  ceiling and drop the lower one.
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
approximation is measured. The exact aggregate set for this class of sets (charging only, one
connection interval, limits on total energy), not only its band, is a generalised polymatroid
(Mukhi, Qu, You and Abate, 2025, Corollary 3.9; Mukhi, Loho and Abate, IEEE Transactions on
Smart Grid 17(3), 2131–2142, 2026).

## How programme (24) is solved

Besides the sum condition (20), the conditions of Theorem 2 are Lemma 1 of the 2024 paper
applied to each car, and Lemma 1 is necessary and sufficient for γᵢ + ΓᵢU₀ ⊆ Uᵢ whatever
half-space description is used for U₀ and Uᵢ, provided U₀ is nonempty (the lemma's
assumption; here U₀ contains the zero profile). `offer.solve` therefore writes each set with irredundant rows (its power box and one
row on total energy, 2K + 1 rows instead of the 4K of H = (L, −L, I, −I)) and, for each row,
Lemma 1's multipliers as the dual of maximising that row over U₀. The feasible values of p,
α, γᵢ and Γᵢ, and so the optimal α, are those of (24) with its constraint α > 0 relaxed to
α ≥ 0, so that a day on which nothing can be offered still has a solution, the single profile
p; `tests/test_offer.py` solves (24) as printed, with the same relaxation, on small fleets and
compares. Rows and columns that a car's set or U₀ fixes at
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
ceiling is zero), no α > 0 meets the conditions of (24): the programme as printed has no
feasible point, and with α ≥ 0 its optimum is 0.

*Proof.* For such a car, γᵢ + Γᵢu₀ must have k-th entry 0 for every u₀ in U₀. At u₀ = 0 this
gives (γᵢ)_k = 0; U₀ contains small multiples of the k-th unit vector, so (Γᵢ)_kk = 0. Then
the (k, k) entry of Σ Γᵢ = αI gives α = 0. ∎

The day-average base set of equation (11), with each car's vector written as above, cannot
meet this case, since its power vector is then zero wherever no car can charge. A base set fixed from the training period can: on any day on
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
  solution set is not inside the ε-feasible set, the decisions whose constraint fails with
  probability at most ε under the distribution of the next sample, is at most |X| · Ψ(αN; p₁, …, p_N), with
  pᵢ = (ε − ρ(N + 1 − i)/rᵢ)₊ and Ψ the distribution function of a sum of independent
  Bernoulli variables (Definition 2). With the radii of Remark 4, every pᵢ = ε − θ and Ψ is a
  binomial distribution function.
- **Corollary 1.** For δ in (0, 1), α in (0, ε) and θ in (0, ε − α), with the radii of
  Remark 4 (equation (14)), N ≥ ln(|X|/δ) / (2(ε − α − θ)²) gives confidence at least 1 − δ.
  It follows from Theorem 4 through Hoeffding's inequality, so it is sufficient, not
  necessary.
- **Theorem 5** (Section III-D) allows a decision set that need not be finite: if the
  constraint is Lipschitz in the decision with constant L (Assumption 1) and the decision set
  has diameter at most D (Assumption 2), the same bound holds for the constraint tightened by
  a margin γ, with |X| replaced by (LD/γ + 1)ⁿ for n decision variables.

**With days as samples** (computed; ε = 0.10, a failure probability of at most 10% on the next
day; δ = 0.05; X a grid of 101 scaling levels of the offer):

| α | θ | Each ball is the drift budget times | Theorem 4 bound first at most δ | Theorem 4 bound at most δ for every N from | Corollary 1 |
|---|---|---|---|---|---|
| 0.02 | 0.040 | 25 | 285 days | 312 days | 2,379 days (7.6 times) |
| 0.05 | 0.025 | 40 | 1,057 days | 1,121 days | 6,089 days (5.4 times) |

The bound is not monotone in N, because αN is rounded down; "for every N from" is the
reading that does not depend on a lucky N. Section III suggests no value for θ, α or δ.

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
The rule as run (next section) uses the one side of this result that a delivery target needs,
which allows a smaller step.

## The day-level rules as run

Arms 4, 5 and 6 give every car present at 10:00 its two bounds at one shared quantile level q,
chosen each morning from earlier days (`daylevel.py`).

**The day score.** For a car and a target, its score is the largest level on a grid (0, then
0.002 to 0.95 in steps of 0.001) at which its bound, as the engine uses it, held: the stay
bound rounded down to the 15-minute grid and capped at the horizon, against the true
deadline; the energy bound against the energy taken. The level is the standardisation the
plan calls for: it measures each shortfall on the probability scale of the model's own
predicted distribution, fitted on training data only, so a deadline error and an energy error
are on one scale. The day's score s is the smallest over its cars and both targets. Since
each bound rises with the level, every bound of the day held at level q exactly when q ≤ s.
The score is computed from the predictions made that morning, under the model then in force,
and stored once the day's outcome is known; it is never recomputed. A day is scored if at
least five present cars took more than 0.5 kWh after 10:00; only scored days enter the rules.

**The rule.** With n scored days in the last 90 days, a rule at miss level a uses the k-th
smallest of their scores, k = ⌊a(n + 1)⌋; the trivial bound (deadline at 10:00, energy 0) if
k < 1; the top of the grid if k > n. Under exchangeable days this is the split conformal
quantile, and the next day's bounds all hold with probability at least 1 − a.

- **Primary, adaptive conformal inference** with days as steps: a starts at a₁ = 0.10 and,
  after each scored day, aₜ₊₁ = aₜ + γ(0.10 − errₜ), with errₜ = 1 if some bound failed at
  the level used that day. A working level at or below 0 gives the trivial bound.
- **Secondary, split conformal** at a = 0.10, only on days whose window holds at least 40
  scored days (check D8); on other days the secondary arms use their own copy of the primary
  rule, and the record says which ran.

**What is kept of Proposition 4.1, and the step.** Summing the update over T scored days that
begin at working level a_S gives a_{S+T} = a_S + γ Σ (0.10 − errₜ), so the share of failed
days among them is 0.10 + (a_S − a_{S+T})/(γT). The trivial bound cannot fail, so a level at or
below 0 is never followed by a drop; with each outcome known before the next decision the
working level stays above −0.9γ, and the share of failed days is at most
0.10 + (a_S + 0.9γ)/(γT), whatever the days do. The rule starts at a_S = a₁; for a later
stretch, such as an evaluation period, a_S is the level logged on its first day, known before
the stretch begins. This is the side of Proposition 4.1 a delivery target needs. The other side (failures
not far below 10%) needs a level at or above 1 to fail for certain; here such a level gives the
top of the grid, which may hold, so that side is not claimed, and a rule that delivers more
often than 90% is reported as such. The step γ = 0.03 is the smallest, to two decimals, for
which the bound from a₁ is within 5 points over the scored weekdays of the frozen development
evaluation: 116 of its 130 weekdays have at least five cars with energy (check D3), and
γ ≥ a₁/(0.05 · 116 − 0.9) = 0.0204 (computed; `daylevel.aci_step`). With γ = 0.03 the bound is
13.6% at 116 days (computed). The paper's two-sided form, (max{a₁, 1 − a₁} + γ)/(γT), would
need γ ≥ 0.188 for the same 5 points, and each failed day would then cut the working level by
0.17.

**Late outcomes.** A day's outcome is known only once its last car has unplugged, so a car
that stays overnight delays it past the next decision. The rules update as outcomes arrive, in
that order. A day decided while the level was positive can then lower it after it has already
reached 0, by a further 0.9γ for each such day, so the bound above holds with 0.9γ replaced by
0.9γ times one more than the number of days pending at once; that number and the share of days
known before the next decision are reported.

**Arm 3, the union bound.** Each car's two bounds at level 0.10/(2n) for n present cars. A
level is used only where check D10 supports it (a level between two grid levels needs both);
elsewhere that bound is trivial, and the share of such cars is reported. D10 governs this arm
only: the calibrated arms choose their level from past days, not from the model's nominal
coverage.

## Robust aggregation: Mukhi, Qu, You and Abate (2025)

Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility",
HSCC 2025; arXiv 2405.08232. The robust aggregate flexibility set at confidence 1 − β is the
set of aggregate profiles that the realised fleet can satisfy with probability at least
1 − β (Definition 2.3), under charging requirements drawn independently from one known
distribution. Lemma 4.3 characterises that set as the intersection of the aggregate sets of
all populations within a Wasserstein ball that contains the realised population with
probability at least 1 − β (Theorem 4.2 and equation (11)), and Theorem 4.4 states it as a
generalised polymatroid. The direction a delivery guarantee needs follows from the
construction: a profile in the intersection can be met by every population in the ball, so
with probability at least 1 − β. The paper names one shared distribution for every vehicle as
a limitation of its model (Section 2.3). This
project's target differs: the realised day, with the distribution unknown, changing, and
sessions not independent, scored on nine profiles. It reports empirical delivery, and names
this result as the one that would apply if its assumptions held.

## What each day-level rule gives

| Rule | What it gives | Under which assumption |
|---|---|---|
| Union bound across the 2n per-car bounds, each at level 0.10/(2n) (0.00125 with 40 cars, computed) | Every bound holds, so the offer is delivered, with probability at least 90% on each day | Every per-car bound valid at its level, given what is known at 10:00 |
| Adaptive conformal inference, days as steps | As published, a share of failed days over T days within (max{α₁, 1 − α₁} + γ)/(γT) of 10%; as run here, at most 10% + (a_S + 0.9γ)/(γT) over T scored days from working level a_S (13.6% at 116 days from a₁) | None on the data; each day's outcome known before the next decision (late outcomes weaken the bound as stated above) |
| Split conformal rule on a window of past day scores | At least 90% for the next day | Exchangeable days |
| Theorem 4 of Yan, Parise and Bitar, days as samples | At most δ chance that some kept decision fails with probability above 10% on the next day | Independent days, a known drift bound ρ on whole-day configurations, a fixed finite decision set, and the day counts above |

## The requirements against the data

| Requirement | What the data provide | Status |
|---|---|---|
| Independent days (2022 theorem; also the reading of a long-run share) | Cars on one day share the day's events, prices and calendar; drivers return | Not given; cars on one day are not treated as independent, and uncertainty is resampled by whole weeks |
| A known drift bound ρ on whole-day configurations (condition (9)) | A day is a varying number of cars, each with a deadline and an energy; the 2020 closure is a jump no earlier data can measure | Not given; a norm on that space would itself be a choice |
| Enough days (Theorem 4) | 483 weekdays in development (computed), so 312 is met and 1,121 missed by 638; a calendar year has 261 weekdays (computed), about 250 once holidays are removed; 182 in 2021 (computed) | Met at the first setting on development data only |
| A fixed finite decision set (Theorem 4), or a constraint Lipschitz in the decision on a bounded set (Theorem 5) | The offer is fitted continuously from the day's cars | Not met for Theorem 4; a grid of 101 scales is the closest reading. Theorem 5 would need a Lipschitz constant for the day's shortfall in the offer's scale, which is not derived here |
| Exchangeable days (split conformal) | The closure and the 2021 reopening break exchangeability | Its 90% is an empirical target across the change |
| Each outcome known before the next decision (adaptive conformal) | A day's outcome uses each car's recorded energy total, known only at unplug (check D1's fallback), so a car that stays overnight delays it past the next decision | Not met on every day; the share of days whose outcome is known before the next decision is reported, and the rule is updated as outcomes arrive |

So the day-level rate in this project rests on a calibration rule over past days or on a
union bound across cars; the requirements of the 2022 theorem are reported beside the days
available, and no certificate from it is claimed.
