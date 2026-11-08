# ev-fleet-offer

**Does a per-car model pay at fleet level? A charging-flexibility offer built from learned per-car bounds, calibrated to be delivered on 90% of days and scored by replay, before and across a real change in who uses a site and how.**

An aggregator can sell the charging flexibility of the cars at a car park only if it can say in advance how much charging it can shift, and then deliver it. That depends on when each car will leave and how much energy each will still take, both uncertain and both changing over time. At 10:00 on each weekday this project gives every car plugged in a lower bound on both, from a **population** view (the site's recent sessions, given how long the car has been plugged in) or a **learned** view (a model of that car and its driver). It turns the bounds into a fleet offer with the inner-approximation method of Al Taha, Vincent and Bitar (2024, Theorem 2), and scores each offer by replaying the cars that were actually present, on the public ACN-Data sessions at Caltech, with JPL as a second site. The hypotheses, thresholds and analyses were fixed in the git tag `analysis-plan` before the test months were opened; "registered" below means fixed there.

**The full report**, with the problem in notation, the proofs, the figures and every table: [`report/`](report/).

## Results

**Per-car learning tightens each car's bounds, and the gain does not reach the fleet offer.** From September 2019 to February 2020, at matched coverage (90.7% for both), the learned bound keeps 2.93 h of each car's remaining stay against 2.01 h for the population bound. Calibrated to be delivered on 90% of days, the fleet offer made from the learned bounds is 0.18 kWh against 1.76 kWh from the population bounds, delivered on 91% and 90% of days: 0.4% and 3.9% of the 45.6 kWh the same method offers with perfect knowledge of every car.

| Hypothesis | Sep 2019 to Feb 2020 (116 scored weekdays) | Sealed months (26) |
|---|---|---|
| H1: the learned offer is at least 10% larger than the population offer (registered rule) | Fail, with both offers in effect zero | Not testable: both offers are zero |
| H1 with both calibrated on delivery (secondary, chosen after the May to August 2019 look) | 0.18 against 1.76 kWh, delivered on 91% and 90% of days; difference −1.57 kWh (−1.78 to −1.36) | 0 against 0.84 kWh, delivered on 100% and 92%; difference −0.84 kWh (−1.53 to −0.16) |
| H2: the union-bound offer is delivered on at least 98% of days and is at most half the learned offer | Pass, vacuously: both offers are zero | Pass, vacuously |
| H3: after the 2021 reopening, an offer never updated fails, and the adaptive rule recovers within 40 scored weekdays | – | Not testable: the record ends 15 scored weekdays after the reopening; each clause needs 20 |
| H4: the learned stay bound keeps at least 0.5 h more than the population bound at matched 90% coverage | Pass: 2.93 against 2.01 h | Pass: 2.11 against 1.52 h |
| JPL, run once with the Caltech pipeline frozen | (pending) | |

Intervals are 95%, resampling whole weeks; offers are compared only at matched delivery rates. The sealed months are the scored weekdays from 1 March 2020: 9 before the campus closure and 17 from August 2021. The predictions written before any fleet result was seen are in [`docs/predictions.md`](docs/predictions.md).

**Outcome.** Of the outcomes the analysis plan names, the registered design gives a degenerate offer: both offers are zero, and H1 compares two zeros. The secondary design gives the second outcome: once both are calibrated by day, the learned offer is no larger than the population offer. The third, an offer calibrated car by car that fails after the change at a measured cost of recovery, could not be tested (H3).

**Where the offer is lost.** With perfect knowledge the offer reaches 42% of the exact band width (39.4 of 93.7 kWh). Under this formulation (one common level, a template, fixed allocation) the offer made from forecasts keeps about 4% of the perfect-knowledge offer (1.55 of 39.4 kWh). Knowing which cars take nothing more does not restore it. Splitting each profile among the cars with hindsight keeps between a fifth and three quarters, and there the learned model keeps the most, where under the fixed allocation the population model does. This is consistent with the allocation fixed at 10:00 being where the offer is lost. The registered offers are zero for a simpler reason: at the small levels the registered rules need, a car's energy bound is zero whenever its chance of taking nothing more is at least that level, and 16.4% of car-days take nothing more.

**Limits.** The energy a car takes after 10:00 is estimated, not measured, and the 6.7 kW power cap is assumed for every station. The test periods were looked at while the project was scoped ([`docs/prior-looks.md`](docs/prior-looks.md)), so they are a frozen retrospective check rather than a blind test, and the sealed months hold 26 scored weekdays.

## Design

- **Decision.** At 10:00 on each weekday, for the cars plugged in, a set of fleet charging profiles for 10:00 to 20:00 in 15-minute steps. Nothing after 10:00 is used.
- **Per-car sets.** A charging-only case of the battery model of the 2024 paper, with a deadline no later and an energy ceiling no higher than the car's predicted lower bounds; whenever every bound holds, the offer lies inside the fleet's true set.
- **Models.** Gradient-boosted quantile regression on inputs known at 10:00, refitted monthly, against the Kaplan–Meier population rule, a stratified table, each driver's own history, the mixture predictor of Lee, Li and Low (2019) and the typed plan.
- **Offer and delivery.** Programme (24) of the 2024 paper and its allocation of any offered profile among the cars, fixed at 10:00; each offer is checked on nine profiles against what the cars actually did.
- **Calibration.** One quantile level per day for every car, set from past days by adaptive conformal inference (Gibbs and Candès, 2021) to target delivery on 90% of days, beside a union bound across the cars. What each rule guarantees, and what the finite-sample theorem of Yan, Parise and Bitar (2022) would need with days as samples, is in [`docs/theory.md`](docs/theory.md).
- **Safeguards.** Only the replay engine reads unplug times, and a scramble test runs the whole pipeline to show no later field changes an earlier offer. Sealed data open once, after the tag; every earlier look is listed; each run records its code commit and input fingerprint.

## Where things are

- [`report/`](report/): the full report.
- [`docs/analysis-plan.md`](docs/analysis-plan.md) and [`docs/registration.md`](docs/registration.md): what the tag fixed, and every change after it.
- [`docs/theory.md`](docs/theory.md): the theorem statements used, what each requires, and the proofs.
- [`docs/data.md`](docs/data.md): what is downloaded, the cleaning rules and the data checks.
- [`docs/predictions.md`](docs/predictions.md), [`docs/prior-looks.md`](docs/prior-looks.md), [`docs/discovery-log.md`](docs/discovery-log.md), [`docs/related-work.md`](docs/related-work.md).
- [`src/fleetoffer/`](src/fleetoffer/): the code, one module per job, listed in its README.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run fleetoffer demo
```

The demo runs the whole pipeline on made-up fleets with known flexibility sets; it is not a result. Session data come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), available for educational and research use after registration, and are not redistributed here; if you use them, cite Lee, Li and Low (2019). With an access token in `ACN_API_TOKEN`:

```bash
uv run fleetoffer fetch caltech        # and jpl; stored under data/, never committed
uv run fleetoffer verify caltech       # same data as fingerprints/caltech.csv?
uv run fleetoffer percar               # per-car models and baselines, months through August 2019
uv run fleetoffer checks               # the checks before the tag, May to August 2019
uv run fleetoffer levels               # offer size against delivery at every level
uv run fleetoffer options              # calibration on delivery
uv run fleetoffer accounting           # where the perfect-knowledge offer goes
uv run fleetoffer evaluate evaluation  # September 2019 to February 2020; then sealed, jpl
uv run fleetoffer report               # figures and the results table
uv run fleetoffer note                 # the full report in report/, filled from the tables
```

Each command reads what the ones above it wrote. `evaluate` runs only in a clone that holds the `analysis-plan` tag, and `report` needs all three of its runs; `uv run fleetoffer --help` lists every command.

## Related work

Al Taha, Vincent and Bitar (2024) compute inner approximations of the aggregate flexibility of a fleet whose individual sets are known, and Mukhi, Loho and Abate (2026) give the exact aggregate set for this class of sets. Mukhi, Qu, You and Abate (2025) certify an aggregate set for cars drawn independently from one known distribution. Lee, Li and Low (2019) predict individual departures on this dataset, and Winschermann, Bañol Arias, Hoogsteen and Hurink (2023) measure, at an office car park, what individual averages of energy and dwell time add to a charging strategy, the nearest published answer to the question asked here. Further work, with what each paper does, is in [`docs/related-work.md`](docs/related-work.md).

## References

- Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4), 3142–3154, 2025 (online April 2024). doi 10.1109/TSG.2024.3384871.
- Mukhi, Loho and Abate, "Exact characterization of aggregate flexibility via generalized polymatroids", IEEE Transactions on Smart Grid 17(3), 2131–2142, 2026. doi 10.1109/TSG.2026.3659513.
- Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility", HSCC 2025. arXiv 2405.08232.
- Lee, Li and Low, "ACN-Data: analysis and applications of an open EV charging dataset", ACM e-Energy 2019.
- Winschermann, Bañol Arias, Hoogsteen and Hurink, "Assessing the value of information for electric vehicle charging strategies at office buildings", Renewable and Sustainable Energy Reviews 185, 113600, 2023. doi 10.1016/j.rser.2023.113600.
- Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments", IEEE Control Systems Letters 6, 2022. arXiv 2205.03748.
- Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS 2021. arXiv 2106.00170.

## Licence

Code: MIT. Data: see "How to run".
