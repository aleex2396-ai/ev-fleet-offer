# What the guarantees require

This note states the published guarantees behind the three headline rules exactly, with
their theorem numbers, and sets each requirement against what the Caltech data provide.
Numbers marked "computed" come from `src/staypromise/theory.py` and are printed by
`staypromise requirements`.

## The promise as a chance constraint

A promise is a number of hours *x* for one plugged-in car, with the stay ξ (hours from plug-in
to unplug) as the uncertain quantity. The promise is kept when x ≤ ξ, and should be kept with
probability at least 1 − ε. Throughout, ε = 0.10: a 90% promise.

## The drift-aware rule: Yan, Parise and Bitar (2022)

S. Yan, F. Parise and E. Bitar, "Data-Driven Approximations of Chance Constrained Programs in
Nonstationary Environments", arXiv:2205.03748v1 (IEEE Control Systems Letters, 2022),
Section III. The method is a robust sample approximation of a chance constraint, not
distributionally robust optimisation; the paper's footnote 1 notes that applying the
distributionally robust approach here "is likely overly conservative".

**Assumptions.**

- The samples ξ₁, ξ₂, … are independent but not necessarily identically distributed.
- Condition (9): the 1-Wasserstein distance between the distributions of samples *i* and
  *i + k* is at most ρ(k), for **every** i ≥ 1 and k ≥ 0, where ρ is "a known function" with
  ρ(0) = 0. Ages are counted in samples, not in calendar time.
- For Theorem 4, the set of candidate decisions is finite (here: a grid of promise values).

**The method**, (10) to (12). Each past sample ξᵢ is replaced by a ball of radius rᵢ around
it. A promise is admissible if it is kept for every point of the ball, for at least
⌈(1 − α)N⌉ of the N samples, so at most ⌊αN⌋ samples may be discarded. For a promise with
stays on [0, ∞), the ball constraint reads x ≤ max(ξᵢ − rᵢ, 0). The largest admissible
promise is therefore the (⌊αN⌋ + 1)-th smallest of the widened-down stays max(ξᵢ − rᵢ, 0),
rounded down to the grid. (This last step follows from (12); the paper does not write it for
this constraint.)

**Lemma 3.** For one sample, the chance that the ball constraint holds is at most the chance
that the target constraint holds, plus ρ(N + 1 − i)/rᵢ. The proof goes through Markov's
inequality.

**Theorem 4.** The probability that the admissible set contains a promise that breaks the
90% target for the next sample is at most |X| Ψ(αN; p₁, …, p_N), with
pᵢ = (ε − ρ(N + 1 − i)/rᵢ)₊ and Ψ the distribution function of a sum of independent
Bernoulli variables (Definition 2).

**Remark 4 and Corollary 1.** With radii (14), rᵢ = ρ(N + 1 − i)/θ, every pᵢ equals ε − θ and
Ψ becomes a binomial distribution function. Corollary 1 bounds it with Hoeffding's
inequality: for δ in (0, 1), α in (0, ε) and θ in (0, ε − α),
N ≥ ln(|X|/δ) / (2(ε − α − θ)²) gives confidence at least 1 − δ.

**What this means for a 90% promise** (computed; 95% confidence; promises on a 0 to 24 hour
grid in steps of 1 hour, 25 candidates, or 15 minutes, 97 candidates):

| Candidates | α | θ | Each ball is the drift budget times | Samples needed, Theorem 4 | Samples needed, Corollary 1 |
|---|---|---|---|---|---|
| 25 | 0.02 | 0.040 | 25 | 253 | 1,943 |
| 25 | 0.05 | 0.025 | 40 | 863 | 4,972 |
| 25 | 0.05 | 0.040 | 25 | 4,542 | 31,074 |
| 25 | 0.08 | 0.010 | 100 | 6,676 | 31,074 |
| 97 | 0.02 | 0.040 | 25 | 311 | 2,366 |
| 97 | 0.05 | 0.025 | 40 | 1,104 | 6,057 |
| 97 | 0.05 | 0.040 | 25 | 5,862 | 37,853 |
| 97 | 0.08 | 0.010 | 100 | 8,639 | 37,853 |

"Samples needed, Theorem 4" is the smallest N from which Theorem 4's bound stays at or below
δ (the bound is not monotone in N because αN is rounded down). Corollary 1 is a sufficient
condition and asks for four to eight times more. The paper suggests no value for θ, α or δ.

Two consequences follow from the statements:

- **The balls are wide.** Because θ < ε − α ≤ ε, every ball is more than 1/ε = 10 times the
  drift budget at that age, and 25 to 100 times in the settings above.
- **A jump sets the budget at every age.** Condition (9) must hold for every starting sample.
  If a change of size J can happen at any time, then ρ(k) ≥ J for every k ≥ 1, so every past
  sample's ball is at least J/θ.

## The drift-free rule: adaptive conformal inference (Gibbs and Candès, 2021)

I. Gibbs and E. Candès, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021
(arXiv:2106.00170).

- **Update**, equation (2): the working level moves as αₜ₊₁ = αₜ + γ(α − errₜ), where errₜ is 1
  when the outcome falls outside the prediction set. For a promise, the set is the hours at or
  above the promise; a working level at or below 0 gives a promise of 0 hours, which is always
  kept.
- **Guarantee**, Proposition 4.1, equation (5): with probability one, for every T, the
  average miss rate over T steps is within (max{α₁, 1 − α₁} + γ)/(γT) of the target. The paper
  states that it "puts no constraints on the data generating distribution".
- **What it assumes**: a fixed step size γ (the experiments use 0.005), and each outcome
  revealed before the next prediction. The paper's discussion lists delayed outcomes as an
  open problem.

## Choosing the drift budget from data: Keehan, Anderson and Wiesemann (2025)

D. Keehan, E. Anderson and W. Wiesemann, arXiv:2510.18566 (v2, December 2025). Their drift is
measured in the ∞-Wasserstein distance, a stronger requirement than condition (9). Section 2
assumes the drift bound ρ is known, and states that in applications "one selects the value
of ρ that best fits the data via cross-validation". Their guarantees treat ρ and the weights
as fixed; none covers a value chosen from the data.

## Requirements against the data

| The guarantee's requirement | What Caltech provides | Status |
|---|---|---|
| Independent samples (both papers) | Cars on one day share the day's events and the app; the same drivers return | To be tested by the fleet-level check (each rule's promises summed per day) |
| A known drift budget for every pair of samples (condition 9) | A budget must be estimated; the 2020 closure is a jump of several hours in typical stay | Not given by the data; at most within an era |
| Enough samples (Theorem 4) | 15 to 45 sessions a week from November 2020, about 100 to 140 by summer 2021 (`fingerprints/caltech.csv`) | About 250 to 1,100 samples in the moderate settings above: weeks to months, not years |
| Balls of 25 to 100 times the drift budget (Lemma 3, Remark 4) | Typical stays of one to two hours after the reopening | A sample whose budget exceeds 0.02 to 0.07 hours is widened past a typical stay; to be measured on data from before March 2020 |
| Complete samples | At decision time, recent stays are still in progress | To be handled in the analysis plan |
| Each outcome known before the next promise (adaptive conformal) | A stay is known only when the car leaves | To be handled in the analysis plan |
| A drift budget that is known, not chosen from data (both drift papers) | It must be chosen from data | Not covered by either guarantee |
