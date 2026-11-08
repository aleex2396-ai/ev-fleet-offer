# ev-fleet-offer

**Does a per-car model pay at fleet level? A charging-flexibility offer built from learned per-car bounds, calibrated to be delivered on 90% of days and scored by replay, on months before and after the 2020 campus closure.**

- **Finding.** Per-car learning tightens each car's bounds; with each car's share of the offer fixed in the morning, the gain does not reach the fleet offer ([Results](#results)).
- **Full report**, with the problem in notation, the proofs, the figures and every table: [`report/`](report/).

## Design

- **Problem.** An aggregator can sell the charging flexibility of the cars at a car park only if it can say in advance how much charging it can shift, and then deliver it. That depends on when each car will leave and how much energy it will still take (here, energy shaped by the site's own charging scheduler), both uncertain and both changing over time.
- **Decision.** At 10:00 on each weekday, for the cars plugged in, a set of fleet charging profiles for 10:00 to 20:00 in 15-minute steps. Nothing after 10:00 is used.
- **Two views of each car.** A lower bound on its remaining stay and on its remaining energy, from a **population** view (the site's recent sessions, given how long the car has been plugged in) or a **learned** view (a model of that car and its driver).
- **Models.** Learned: gradient-boosted quantile regression on inputs known at 10:00, refitted monthly. Against it: the Kaplan–Meier population rule, a stratified table, each driver's own history, the mixture predictor of Lee, Li and Low (2019) and the typed plan.
- **Per-car sets.** A charging-only case of the battery model of Al Taha, Vincent and Bitar (2024), with a deadline no later and an energy ceiling no higher than the car's lower bounds. Whenever every bound holds, the offer lies inside the fleet's true set.
- **Offer.** Programme (24) of the same paper (Theorem 2): a shifted and scaled copy of a template set (the base set), with each car's share of any offered profile fixed at 10:00 by its affine allocation.
- **Offer form.** The registered arms use the registered one-dimensional fallback, which moves charging out of 10:00–11:00 and into 11:00–13:00: with the base set fixed in advance, the perfect-knowledge offer's scale was below 0.2 on 37.6% of model-selection weekdays (threshold 25%). The secondary analysis uses programme (24) with the day-average base set.
- **Calibration.** One quantile level per day for every car, set from past days by adaptive conformal inference (Gibbs and Candès, 2021) to target delivery on 90% of days, beside a union bound across the cars. What each rule guarantees, and what the finite-sample theorem of Yan, Parise and Bitar (2022) would need with days as samples, is in [`docs/theory.md`](docs/theory.md).
- **Scoring.** Each offer is replayed against the cars actually present and checked on nine profiles, on the public ACN-Data sessions at Caltech, with JPL as a second site.
- **Data roles.** Caltech through August 2019: development (May to August 2019 for model selection). September 2019 to February 2020: frozen evaluation. From 1 March 2020: sealed test, opened once. JPL: replication, run once.
- **Registration.** The hypotheses, thresholds and analyses were fixed in the git tag `analysis-plan` before the test months were opened; "registered" below means fixed there. The predictions written before any fleet result was seen are in [`docs/predictions.md`](docs/predictions.md).
- **Safeguards.** Only the replay engine reads unplug times, and a scramble test runs the whole pipeline to show no later field changes an earlier offer. Every earlier look at the test data is listed; each run records its code commit and input fingerprint.

## Results

**Per-car learning tightens each car's bounds; with each car's share of the offer fixed in the morning, the gain does not reach the fleet offer.** September 2019 to February 2020:

- **Per car.** At matched coverage (90.7% for both), the learned bound keeps 2.93 h of each car's remaining stay against 2.01 h for the population bound.
- **Fleet.** Calibrated to be delivered on 90% of days, the offer made from the learned bounds is 0.18 kWh against 1.76 kWh from the population bounds, delivered on 91% and 90% of days: smaller by 1.57 kWh (95% interval 1.36 to 1.78).
- **Against perfect knowledge.** Those offers are 0.4% and 3.9% of the 45.6 kWh the same method offers with perfect knowledge of every car.

| Hypothesis | Sep 2019 to Feb 2020 (116 scored weekdays) | Sealed months (26) |
|---|---|---|
| H1: the learned offer is at least 10% larger than the population offer (registered rule, one-dimensional fallback) | Fail, with both offers in effect zero | Not testable: both offers are zero |
| H1 with both calibrated on delivery (secondary, chosen after the May to August 2019 look) | 0.18 against 1.76 kWh, delivered on 91% and 90% of days; difference −1.57 kWh (−1.78 to −1.36) | 0 against 0.84 kWh, delivered on 100% and 92%; difference −0.84 kWh (−1.53 to −0.16) |
| H2: the union-bound offer is delivered on at least 98% of days and is at most half the learned offer | Pass, vacuously: both offers are zero | Pass, vacuously |
| H3: after the 2021 reopening, an offer never updated fails, and the adaptive rule recovers within 40 scored weekdays | – | Not testable: the record ends 15 scored weekdays after the reopening; each clause needs 20 |
| H4: the learned stay bound keeps at least 0.5 h more than the population bound at matched 90% coverage | Pass: 2.93 against 2.01 h | Pass: 2.11 against 1.52 h |

- Intervals are 95%, resampling whole weeks.
- Each offer is given with the share of days it was delivered on; in the sealed months the two are not at matched delivery (100% and 92%).
- The sealed months are the scored weekdays from 1 March 2020: 9 before the campus closure and 17 from July 2021.
- **JPL**, a replication run once and read descriptively:
  - With the Caltech pipeline frozen, the calibrated learned offer is 0.28 kWh against 0.54 kWh for the population, delivered on 99% and 96% of days.
  - Updating on JPL days, 1.88 against 1.87 kWh, both delivered on 90% of days: a difference of 0.01 kWh (95% interval −0.52 to 0.55).
  - One of its 37 runs (the stratified model calibrated on delivery, updating on JPL days) did not finish and is left out; no result here uses it.

**Outcome**, of those the analysis plan names:

- **Registered design:** a degenerate offer. Both offers are zero, and H1 compares two zeros.
- **Secondary design:** the second outcome. Once both are calibrated by day, the learned offer is no larger than the population offer.
- **Not tested:** the third, an offer calibrated car by car that fails after the change at a measured cost of recovery (H3).

**Where the offer is lost.** Diagnostics added after the first look at May to August 2019 and fixed before the tag; the shares are from those months, and the frozen evaluation gives the same first two.

- With perfect knowledge the offer reaches 42% of the exact band width (39.4 of 93.7 kWh).
- Under this formulation (one common level, a template, fixed allocation) the offer made from forecasts keeps about 4% of the perfect-knowledge offer (1.55 of 39.4 kWh).
- Knowing which cars take nothing more does not restore it.
- At level 0.15, every bound of the learned model held on 0% of scored weekdays and the allocation fixed at 10:00 delivered on 1% of them, while a split chosen in hindsight delivered on 100%.
- Across levels the hindsight split, an upper bound rather than a method, keeps a fifth to a half of the perfect-knowledge offer at 90% delivery (a quarter to three quarters in the frozen evaluation), and there the learned model keeps the most.
- This is consistent with the fixed allocation, which is guaranteed only when every bound holds, being where the offer is lost.
- The registered offers are zero for a simpler reason: at the small levels the registered rules need, a car's energy bound is zero whenever its chance of taking nothing more is at least that level, and 16.4% of car-days take nothing more.

**Per-car models**, May to August 2019:

- The boosted trees have the lowest stay loss of every per-car model: mean pinball loss 0.444 h, against 0.474 for each driver's own past and 0.594 for the population rule.
- For energy they only tie each driver's own past (0.683 against 0.684 kWh).
- Neither the trees nor the population rule is calibrated at its nominal level: at level 0.10 their stay bounds missed on 14.2% and 2.8% of car-days, so H4 compares them at matched coverage.

**Limits.**

- The energy a car takes after 10:00 is estimated, not measured, and is shaped by the site's own scheduler: cars took a median 0.64 of the energy their drivers asked for, so every energy bound here bounds scheduled energy, not need.
- The 6.7 kW power cap is assumed for every station.
- The test periods were looked at while the project was scoped ([`docs/prior-looks.md`](docs/prior-looks.md)), so they are a frozen retrospective check rather than a blind test.
- The sealed months hold 26 scored weekdays.

## Where things are

- [`report/`](report/): the full report.
- [`docs/analysis-plan.md`](docs/analysis-plan.md) and [`docs/registration.md`](docs/registration.md): what the tag fixed, and every change after it.
- [`docs/theory.md`](docs/theory.md): the theorem statements used, what each requires, and the proofs.
- [`docs/data.md`](docs/data.md): what is downloaded, the cleaning rules and the data checks.
- [`docs/predictions.md`](docs/predictions.md), [`docs/prior-looks.md`](docs/prior-looks.md), [`docs/discovery-log.md`](docs/discovery-log.md), [`docs/related-work.md`](docs/related-work.md).
- [`src/fleetoffer/`](src/fleetoffer/): the code, one module per job, listed in its README.

## How to run

- Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run fleetoffer demo
```

- The demo runs the whole pipeline on made-up fleets with known flexibility sets; it is not a result.
- Session data come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), available for educational and research use after registration, and are not redistributed here; if you use them, cite Lee, Li and Low (2019).
- With an access token in `ACN_API_TOKEN`:

```bash
uv run fleetoffer fetch caltech        # and jpl; stored under data/, never committed
uv run fleetoffer verify caltech       # same data as fingerprints/caltech.csv?
uv run fleetoffer percar               # per-car models and baselines, months through August 2019
uv run fleetoffer checks               # the checks before the tag, May to August 2019
uv run fleetoffer levels               # offer size against delivery at every level
uv run fleetoffer options              # calibration on delivery
uv run fleetoffer accounting           # where the perfect-knowledge offer goes
uv run fleetoffer evaluate evaluation  # September 2019 to February 2020; then sealed, jpl
uv run fleetoffer report               # figures, results table and the full report in report/
```

- Each command reads what the ones above it wrote.
- `evaluate` runs only in a clone that holds the `analysis-plan` tag, and `report` needs all three of its runs.
- `uv run fleetoffer --help` lists every command.

## Related work

- Al Taha, Vincent and Bitar (2024): inner approximations of the aggregate flexibility of a fleet whose individual sets are known.
- Mukhi, Loho and Abate (2026): the exact aggregate set for this class of sets.
- Mukhi, Qu, You and Abate (2025): an aggregate set certified for cars drawn independently from one known distribution.
- Lee, Li and Low (2019): forecasts of individual departures on this dataset.
- Winschermann, Bañol Arias, Hoogsteen and Hurink (2023): what individual averages of energy and dwell time add to a charging strategy at an office car park, the nearest published answer to the question asked here.
- Further work, with what each paper does: [`docs/related-work.md`](docs/related-work.md).

## References

- Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4), 3142–3154, 2025 (online April 2024). doi 10.1109/TSG.2024.3384871.
- Mukhi, Loho and Abate, "Exact characterization of aggregate flexibility via generalized polymatroids", IEEE Transactions on Smart Grid 17(3), 2131–2142, 2026. doi 10.1109/TSG.2026.3659513.
- Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility", HSCC 2025. arXiv 2405.08232.
- Lee, Li and Low, "ACN-Data: analysis and applications of an open EV charging dataset", ACM e-Energy 2019.
- Winschermann, Bañol Arias, Hoogsteen and Hurink, "Assessing the value of information for electric vehicle charging strategies at office buildings", Renewable and Sustainable Energy Reviews 185, 113600, 2023. doi 10.1016/j.rser.2023.113600.
- Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments", IEEE Control Systems Letters 6, 2022. arXiv 2205.03748.
- Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS 2021. arXiv 2106.00170.

## Licence

- Code: MIT.
- Data: see "How to run".
