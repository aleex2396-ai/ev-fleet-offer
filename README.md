# ev-fleet-offer

**Does a per-car model pay at fleet level? A charging-flexibility offer built from learned per-car bounds, calibrated to be delivered on 90% of days and scored by replay, before and across a real change in who uses a site and how.**

An aggregator that schedules electric-vehicle charging can sell flexibility to the grid only if it can say in advance how much charging it can shift and then deliver it. At a car park the offer depends on the cars plugged in: when each will leave, and how much energy each will still take. Both are uncertain, and both change over time.

At 10:00 on each weekday this project gives every car plugged in a lower bound on when it will leave and on how much energy it will still take, either from a **population** view (the site's recent sessions, given how long the car has been plugged in) or from a **learned** view (a model of that car and its driver). It turns the bounds into a fleet offer with the inner-approximation method of Al Taha, Vincent and Bitar (2024, Theorem 2), sets one quantile level per day from past days to target delivery on 90% of days, and scores each offer by replaying the cars that were actually present. The data are the public charging sessions of ACN-Data at Caltech, with the Jet Propulsion Laboratory (JPL) as a second site. The hypotheses, thresholds and analyses were fixed in the git tag `analysis-plan` before the test months were opened (`docs/analysis-plan.md`, `docs/registration.md`); "registered" below means fixed there.

## Results

**Per-car learning tightens each car's bounds, and the gain does not reach the fleet offer.** From September 2019 to February 2020, at matched coverage (90.7% for both), the learned bound keeps 2.93 h of each car's remaining stay against 2.01 h for the population bound. Calibrated to be delivered on 90% of days, the fleet offer made from the learned bounds is 0.18 kWh against 1.76 kWh from the population bounds, delivered on 91% and 90% of days: 0.4% and 3.9% of the 45.6 kWh the same method offers with perfect knowledge of every car.

The runs are those of the tagged commit; every change made after the tag is listed in `docs/registration.md`. A scored weekday is one on which at least five cars present at 10:00 take energy afterwards. Means are over scored weekdays, 95% intervals resample whole weeks, and offers are compared only at matched delivery rates.

### Registered hypotheses

The sealed months are the scored weekdays from 1 March 2020: 9 in March 2020, before the campus closure, and 17 from August 2021, after the reopening; few cars came in between.

| Hypothesis | September 2019 to February 2020 (116 scored weekdays) | Sealed months (26) |
|---|---|---|
| H1: the learned offer is at least 10% larger than the population offer, under the registered day-level rule | Fail by the registered rule, with both offers in effect zero: the learned offer is zero on every day, the population offer on all but one (0.0034 kWh on 23 September 2019, not delivered) | Not testable: both offers are zero |
| H2: the union-bound offer is delivered on at least 98% of days and is at most half the learned offer | Pass, vacuously: delivered on 98% of days, and both offers are zero | Pass, vacuously: delivered on every day, and both offers are zero |
| H3: after the 2021 reopening, an offer never updated fails, and the adaptive rule restores 90% of days within 40 scored weekdays | – | Not testable: the registered rule dates the reopening to 2 August 2021, and the record ends 15 scored weekdays later; each clause needs 20 |
| H4: the learned stay bound keeps at least 0.5 h more than the population bound at matched 90% coverage | Pass: 2.93 h against 2.01 h (coverage 90.7% and 90.7%, 2,055 car-days) | Pass: 2.11 h against 1.52 h (89.4% and 89.2%, 426 car-days) |
| JPL, run once with the Caltech pipeline frozen | (pending) | |

The predictions written before any fleet result was seen are in `docs/predictions.md`.

### Calibrated on delivery (secondary, chosen after the May to August 2019 look)

The registered rule counts a day as failed when any car's bound fails, and the levels it then needs give every offer zero (below). A secondary rule, fixed before the tag and labelled as chosen after the first development results, counts a day as failed only when the offer was not delivered, and uses the day-average base set of the 2024 paper. Under it the learned offer is 0.18 kWh against 1.76 kWh for the population, delivered on 91% and 90% of days (difference −1.57 kWh, interval −1.78 to −1.36). In the sealed months it is 0 against 0.84 kWh, delivered on 100% and 92% of days (an offer of zero is always delivered; difference −0.84 kWh, interval −1.53 to −0.16).

### Where the offer goes

The offer made with perfect knowledge of every car's set, beside the exact band and beside offers under each relaxation (day-average base set; shares of the perfect-knowledge offer, ranging over the learned, population and stratified models). The rows are read separately, not added. Every setting was chosen on May to August 2019, so that column describes the method; it is not a test.

| | May to Aug 2019 | Sep 2019 to Feb 2020 | Sealed months | 2021 only | JPL |
|---|---|---|---|---|---|
| Scored weekdays (fewest behind a row) | 85 (84) | 116 (115) | 26 (16) | 17 (7) | (pending) |
| Exact band of the cars' true sets (kWh) | 93.7 | 108.5 | 63.9 | 59.8 | |
| Offer with perfect knowledge (kWh, share of the band) | 39.4 (42%) | 45.6 (42%) | 25.0 (39%) | 24.9 (42%) | |
| From forecasts, delivered on 90% of days | 0% to 3.6% | 0.5% to 3.6% | 0.3% to 2.7% | 0% to 0.8% | |
| The same, with the cars that take nothing more known | 0% to 1.9% | 0% to 4.3% | 5.0% to 8.1% | 0.8% to 3.3% | |
| The same forecasts, each profile split among the cars with hindsight, delivered on 90% of days | 21% to 56% | 26% to 74% | 39% to 68% | 30% to 60% | |
| Population rule calibrated on delivery, as run (days delivered) | 3.9% (88%) | 3.9% (90%) | 3.4% (92%) | 1.0% (94%) | |

In the three rows at 90% of days, each model's level is the largest swept level delivered on 90% of scored weekdays (under the fixed allocation, or under the hindsight split in its row), a level chosen with hindsight. Each share is taken over the days its row covers: a model makes no offer on a day without a fit, so a row can cover fewer days than the perfect-knowledge offer. The step with the empty cars known also refits the models.

With perfect knowledge the offer reaches 42% of the exact band width (39.4 of 93.7 kWh on May to August 2019). That gap belongs to programme (24) as used here (its structure-preserving form, this base set and an allocation fixed at 10:00), not to the conditions of the 2024 paper's Theorem 2 alone; the programme maximises the base set's scale, not the band width scored here. Under this formulation (one common level, a template, fixed allocation) the offer made from forecasts keeps about 4% of the perfect-knowledge offer (1.55 of 39.4 kWh, population model calibrated on delivery, delivered on 88% of days). Knowing which cars take nothing more does not restore it; neither diagnostic is a method an operator could run. A split chosen with hindsight keeps between a fifth and three quarters, and the order of the models reverses: under the fixed allocation the population model keeps the most in every column, under the hindsight split the learned model does (56% against 21% for the population model on May to August 2019, 74% against 32% from September 2019 to February 2020, 68% against 56% in the sealed months, 60% against 39% in 2021 alone). This is consistent with the allocation fixed at 10:00, more than the cars that take nothing, being where the offer is lost: at level 0.15 on May to August 2019, every bound of the learned model held on none of the scored weekdays, the fixed allocation delivered on 1% of them, and the hindsight split on all of them. The hindsight split chooses both the level and the split after the fact, so it bounds what a re-split could recover; it does not show what an online re-split would.

### Why the registered offers are zero

A 90%-of-days target with about 18 cars present, read as every bound holding, needs each car's bounds at a small level: the registered rules settled at a median of 0.3% to 2.8% on May to August 2019. A lower bound on a car's remaining energy at level q is zero exactly when the car's chance of taking nothing more is at least q (`docs/theory.md`, "Zero ceilings at small levels"), and 16.4% of car-days at 10:00 take nothing more. At the level each registered rule reached, the share of cars that took energy but were given a ceiling of zero was 100%, 100% and 73% for the learned, stratified and population models on May to August 2019, and 88%, 99% and 69% from September 2019 to February 2020. In the sealed months the median level of every registered rule was 0, where every bound is zero by construction, so those months do not test the proposition. Three readings were registered before the counts were read: a large share at the rules' levels, a share that falls as the level rises, and a much smaller share once the empty cars are known. All three hold; the second and third hold by construction for the empirical rules (their quantiles never cross, and a rule refitted on car-days that took energy has no atom at zero), so the first is the informative one.

### Outcome

Of the outcomes the analysis plan names (Section 1), the registered design gives a degenerate offer: both offers are zero, and H1 compares two zeros. The secondary design gives the second outcome: once both are calibrated by day, the learned offer is no larger than the population offer. The third, an offer calibrated car by car that fails after the change at a measured cost of recovery, could not be tested (H3).

### Limits

The energy a car takes after 10:00 is estimated, not measured, and the site's own scheduler shapes it. Whether a car is still charging at 10:00, which an operator would see, is not in the data; the diagnostic with the empty cars known stands in for it. The power cap of 6.7 kW is assumed for every station. Only nine profiles of each offer are checked. The test periods were looked at while the project was scoped: stay statistics, a reading of raw sessions and, for September 2019 to February 2020, the energy of 50 sampled sessions and the daily counts of cars that set the adaptive rule's step (`docs/prior-looks.md`). They are a frozen retrospective check rather than a blind test. The sealed months hold 26 scored weekdays, so their comparisons are wide.

The note in `note/` sets out the problem, the two propositions with proofs, and these results in five pages (`uv run fleetoffer note`).

## Design

- **The decision.** At 10:00 on each weekday, for the cars plugged in at that moment, a set of fleet charging profiles for 10:00 to 20:00 in 15-minute steps. Nothing after 10:00 is used; arrivals are not forecast.
- **Per-car sets.** A charging-only case of the generalised battery model of Al Taha, Vincent and Bitar (2024, equation (2)): power between 0 and a cap while plugged in, nothing after a deadline, and cumulative energy after 10:00 at most a ceiling. A learned set uses a deadline no later than the car is predicted to leave and a ceiling no higher than it is predicted to take; whenever every car's two bounds hold, the offered set lies inside the fleet's true set (`docs/theory.md`). Charging nothing is always allowed, so "delivered" means the fleet can follow the profile within caps, deadlines and ceilings, not that each driver is charged.
- **Per-car models.** Gradient-boosted quantile regression of remaining stay and remaining energy from inputs known at 10:00 (time, calendar, station, cars present, the driver's earlier sessions, a typed plan if recorded in time), refitted monthly on the past 90 days. It is benchmarked against the Kaplan–Meier quantile given the time already plugged in (the population rule), a stratified table by plug-in hour, each driver's own history, the mixture predictor of Lee, Li and Low (2019) and the typed plan at face value.
- **The offer.** Programme (24) of the 2024 paper fits a shifted, scaled base set inside the sum of the per-car sets, and its Section IV splits any offered profile among the cars by an affine map fixed at 10:00. The conditions are sufficient, not necessary; for these sets the exact band of the fleet's cumulative energy has a closed form, and the gap to it is reported.
- **Delivery and calibration.** Each day's offer is checked on nine profiles against what the cars actually did: under the allocation fixed at 10:00 (the primary check, the one an aggregator could execute) and under a split chosen with hindsight (an upper bound). One quantile level per day is set from past days by adaptive conformal inference (Gibbs and Candès, 2021) to target delivery on 90% of days, beside a union bound across the cars. What each rule guarantees, and what the finite-sample theorem of Yan, Parise and Bitar (2022) would need with days as samples, is stated in `docs/theory.md`.
- **No look-ahead.** Only the replay engine reads unplug times. A scramble test alters every field that becomes known after a cutoff, runs the whole pipeline, linear programme included, and requires every earlier offer to be unchanged.
- **Sealed test data.** Caltech sessions from 1 March 2020 open only once the `analysis-plan` tag exists; JPL is analysed once, with the settings frozen. Every earlier look at these periods is listed in `docs/prior-looks.md`.
- **Provenance.** Each run writes a manifest with the code commit, parameters, package versions and a fingerprint of its input.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run fleetoffer demo
```

The demo runs the whole pipeline on made-up fleets with known flexibility sets, so the linear programme is checked where the answer is known. It is not a result.

Session data come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), available for educational and research use after registration; they are not redistributed here. If you use the data, cite Lee, Li and Low (2019). With an access token in `ACN_API_TOKEN`:

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
uv run fleetoffer note                 # the note, filled from the tables, printed to PDF
```

Each command reads what the ones above it wrote. `evaluate` runs only in a clone that holds the `analysis-plan` tag, and `report` needs all three of its runs. `series` and `d1` (the time-series check), `snapshot-check` and `requirements` are described in `docs/data.md` and `docs/theory.md`; `uv run fleetoffer --help` lists every command.

The energy a car takes after 10:00 is not a field of the session record. The per-session time series that would give it failed a quality check fixed before it was fetched (300 sampled sessions: 85% usable against a 95% threshold, median energy error 5.1% against 5%), so it is estimated from each session's recorded energy and charging-done time, and the energy ceiling is approximate. `docs/data.md` describes what is downloaded, the cleaning rules and their counts, the time-series check and the estimate, and the sealed periods.

```
src/fleetoffer/
  acn.py          ACN-Data download, sessions and per-session time series, with fingerprints
  data.py         sessions table, cleaning rules, seals
  interface.py    what a model may see at the decision time
  engine.py       replay per car and per day; the only reader of unplug times
  academic.py     Caltech's academic calendar and Institute holidays, as day flags
  features.py     inputs known at 10:00, from sessions that had ended
  quantiles.py    per-car quantile models (boosted trees, a small network, linear) and their
                  monthly refits inside the engine
  baselines.py    population rule, stratified table, each driver's own past, the mixture
                  predictor of Lee, Li and Low, the typed plan
  models.py       Kaplan-Meier quantiles and the earlier per-car stay rules
  percar.py       the per-car layer on development months: settings search, refits, scores
  offer.py        per-car sets, the inner-approximation programme, the affine allocation,
                  the nine-profile delivery checks and the one-dimensional fallback
  calibrate.py    what each day-level route to 90% of days requires (the theorems' arithmetic)
  daylevel.py     the day-level rules: one shared level from past days (adaptive and split
                  conformal, on every bound holding or on delivery), the union bound, and the
                  sweep of offers over every level
  devchecks.py    the checks on development data before the tag (analysis plan, Section 10a)
  options.py      calibration on delivery and the offer forms, on the model-selection months
  levels.py       the level sweep: offer size against delivery at every level, and its tables
  evaluation.py   the runs after the tag: frozen evaluation, sealed months, both JPL tracks,
                  sensitivities and the hypotheses' readings
  h0.json         the base set of every run after the tag, committed and loaded
  resample.py     uncertainty by resampling whole weeks
  checks.py       the look-ahead scramble test, per car and per day
  synthetic.py    made-up fleets with known flexibility sets
  figures.py      figures, each drawn from a table also written to disk
  note.py         the note on the results: every result read from the tables, printed to PDF
  pipeline.py     end-to-end runs with manifests
tests/            unit tests, including the look-ahead tests (no network needed)
note/             the note's page, with a named field for each result
fingerprints/     SHA-256 of every downloaded month
docs/             analysis plan, what the tag fixes (registration), data notes, theory with theorem
                  statements, earlier looks, written predictions, discovery log, related work
```

## Related work

Al Taha, Vincent and Bitar (2024) compute inner approximations of the aggregate flexibility of a fleet whose individual sets are known, and Mukhi, Loho and Abate (2026) give the exact aggregate set for this class of sets as a generalised polymatroid. Mukhi, Qu, You and Abate (2025) certify an aggregate set for cars drawn independently from one known distribution, and Su, Xiang, Liu and Nie (2026) aggregate per-vehicle sets under a modelled chance of early departure. On the prediction side, Lee, Li and Low (2019) predict individual departures on this dataset, Genov and others (2023) forecast session flexibility on it, and Winschermann, Bañol Arias, Hoogsteen and Hurink (2023) measure, at an office car park, what individual averages of energy and dwell time add to a charging strategy, the nearest published answer to the question asked here. Work that forecasts fleet-level flexibility directly, calibrates fleet forecasts, or bids a fleet under a chance constraint is listed in `docs/related-work.md`, with what each paper does. None of the papers listed there builds the aggregate set from learned per-car bounds and scores it by replayed delivery in time order.

## References

- Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4), 3142–3154, 2025 (online April 2024). doi 10.1109/TSG.2024.3384871. arXiv 2207.07067.
- Mukhi, Loho and Abate, "Exact characterization of aggregate flexibility via generalized polymatroids", IEEE Transactions on Smart Grid 17(3), 2131–2142, 2026. doi 10.1109/TSG.2026.3659513.
- Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility", HSCC 2025. arXiv 2405.08232.
- Su, Xiang, Liu and Nie, "An inner approximation aggregation of electric vehicles considering uncertain dynamic characteristics of users", Sustainable Energy, Grids and Networks 47, 2026. doi 10.1016/j.segan.2026.102350.
- Lee, Li and Low, "ACN-Data: analysis and applications of an open EV charging dataset", ACM e-Energy 2019.
- Genov, De Cauwer, Van Kriekinge, Coosemans and Messagie, "Forecasting flexibility of charging of electric vehicles: tree and cluster-based methods", Applied Energy 353, 121969, 2023. doi 10.1016/j.apenergy.2023.121969.
- Winschermann, Bañol Arias, Hoogsteen and Hurink, "Assessing the value of information for electric vehicle charging strategies at office buildings", Renewable and Sustainable Energy Reviews 185, 113600, 2023. doi 10.1016/j.rser.2023.113600.
- Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments", IEEE Control Systems Letters 6, 2022. arXiv 2205.03748.
- Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS 2021. arXiv 2106.00170.

## Licence

Code: MIT. Data: see "How to run".
