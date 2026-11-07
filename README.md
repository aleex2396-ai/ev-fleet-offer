# ev-fleet-offer

**Does a per-car model pay at fleet level? A charging-flexibility offer built from learned per-car bounds, calibrated to be delivered on 90% of days and scored by replay, before and across a real change in who uses a site and how.**

An aggregator that schedules electric-vehicle charging can sell flexibility to the grid only if it can say in advance how much charging it can shift and then deliver it. At a car park the offer depends on the cars plugged in: when each will leave, and how much energy each will still take. Both are uncertain, and both change over time.

The offer can be built from a **population** view (every present car gets bounds from the site's recent sessions, given only how long it has already been plugged in) or from a **learned** view (each car gets its own bounds, predicted from what is known about it and its driver at the moment the offer is made). This project builds both on the public Caltech ACN-Data charging sessions and turns each into a fleet offer with the inner-approximation method of Al Taha, Vincent and Bitar (2024, Theorem 2). Each offer is calibrated on past days to a target of delivery on 90% of days, and both are scored by replaying them against the cars that were actually present. Three things are measured: how large each offer is beside the delivery rate it achieves, how much the learned model adds over the population view under the same calibration, and how both change across the 2020 campus closure, after which the median stay at the site fell from about six hours to under two.

The result is one of three sentences, each worth reporting: per-car learning offers more flexibility at the same delivery rate and keeps its margin through the change; or, once both are calibrated by day, it offers no more than the population view, so per-car accuracy does not survive aggregation; or offers calibrated car by car fail at fleet level right after the change, on days when many drivers' errors share a sign, and restoring 90% of days costs a measured share of the offer. Shared errors within a day are one explanation to be tested, not a conclusion written in advance.

## Status

In development. The checks, settings and per-car scores so far use Caltech data before September 2019 only (`docs/data.md`, `docs/theory.md`, `docs/registration.md`); the runs fixed by the tag have started, and no evaluation result is published here yet. The analysis plan (`docs/analysis-plan.md`: decision, offers, models, hypotheses with pass and fail criteria, the checks to run on development data, and the list of earlier looks at the data) is frozen by the git tag `analysis-plan` before any confirmatory run and before any sealed data are opened. "Pre-registered" below means fixed in that document before the tag.

## Development results (May to August 2019)

Every setting was chosen on these months, so their numbers describe the method; they are not a test. Means over the 85 scored weekdays, offers on the day-average base set (`uv run fleetoffer levels`, then `uv run fleetoffer accounting`; the analyses after the first fleet run are listed in `docs/registration.md`).

- With perfect knowledge the offer reaches 42% of the exact band width (39.4 of 93.7 kWh).
- Under this formulation (one common level, a template, fixed allocation) the offer made from forecasts keeps about 4% of the perfect-knowledge offer (1.55 of 39.4 kWh, population model calibrated on delivery, delivered on 88% of days). The registered rule gives the learned offer zero on every scored weekday.
- At the shared level each registered day-level rule reaches (median 0.3% for the learned model, 0.9% for the stratified table, 2.8% for the population model), most cars get an energy ceiling of zero: every car that took energy under the learned and stratified models, about three in four under the population model. A lower bound at level q is zero exactly when the car's chance of taking nothing more is at least q (`docs/theory.md`), and 16.4% of car-days at 10:00 take nothing more.
- Knowing which cars take nothing does not restore the offer: delivered on 90% of days under the fixed allocation, it keeps 0% to 1.9% of the perfect-knowledge offer. The same offers delivered by a split among the cars chosen with hindsight keep 21% to 56% at 90% delivery, an upper bound rather than a method. This is consistent with the allocation fixed at 10:00, more than the cars that take nothing, being where the offer is lost.

## Design

- **The decision.** At 10:00 on each weekday, for the cars plugged in at that moment, a set of fleet charging profiles for 10:00–20:00 in 15-minute steps. Nothing after 10:00 is used; no arrivals are forecast.
- **Per-car sets.** Each car's flexibility set is a charging-only special case of the generalised battery model of Al Taha, Vincent and Bitar (2024, equation (2)), built as in their Example 1 but without discharge, with an energy ceiling in place of the battery's capacity limit, and without the energy requirement: power between 0 and a cap while plugged in, zero after a deadline, and cumulative energy after 10:00 at most what the car will still take. The set asks nothing of the car: charging nothing is always allowed, so "delivered" means the fleet profile can be split within caps, deadlines and ceilings, not that each driver is charged. The energy given to the cars under each delivered profile is reported beside the delivery rate, and a version with a required-energy floor per car was planned and is not run (`docs/registration.md`). The learned set uses one-sided bounds, a deadline no later than the car is predicted to leave and an energy ceiling no higher than it is predicted to take; a short proposition in `docs/theory.md` shows that the offered set then lies inside the fleet's true set whenever every car's two bounds hold (a sufficient condition, not a necessary one).
- **The learned model.** Gradient-boosted quantile regression of remaining stay and remaining energy from inputs known at 10:00 (time, calendar, station, cars present, price regime, the driver's earlier sessions, a typed plan if recorded in time). For a car plugged in at 07:40 with five earlier sessions on record, the model returns at 10:00 lower quantiles of the hours it will still stay and the kWh it will still take; the offer uses those lower quantiles. It is to be trained in time order on sessions before March 2020, validated by pinball loss and coverage, and benchmarked against the Kaplan–Meier quantile given the time already plugged in, a stratified table by plug-in hour, each driver's own history, the mixture predictor of Lee, Li and Low (2019) and the typed plan at face value.
- **The offer.** The structure-preserving linear programme of the 2024 paper (its programme (24)) fits a shifted, scaled base set inside the sum of the per-car sets; the allocation of its Section IV splits any offered profile among the cars by an affine map fixed at 10:00. The conditions are sufficient and conservative, as the paper states. For the charging-only sets used here the exact width of the fleet's cumulative-energy band has a closed form (and the exact aggregate set is a generalised polymatroid, Mukhi, Loho and Abate, 2026), so the gap between the offer and the exact band is reported on development days as the method's slack.
- **Reliability.** Delivery is checked by replay in two ways. The allocation fixed at 10:00 is checked against what each car actually accepted; this is the primary check, and the one an aggregator could execute. A split chosen with hindsight among the cars actually present, by linear programming, is reported as an upper bound. Nine fixed profiles from the offer are checked, not every profile, and the results say so. The 90%-of-days target is set by a day-level calibration rule (adaptive conformal inference with days as steps, or a split conformal rule on the day's largest error), compared with a union bound across the cars' deadline and energy bounds. What each rule guarantees, and under which assumptions, is stated in `docs/theory.md`. So is the finite-sample theorem of Yan, Parise and Bitar (2022) for chance constraints under drift, with what it would need with days as samples: independent days, a known bound on how far whole-day configurations can drift, and 312 or 1,121 past days at the two settings used there. The record holds 483 weekdays before the change and 182 after it, so the count is met at the first setting and not at the second; the 90% target here rests on the calibration rules, not on that theorem.
- **Look-ahead, checked by a scramble test.** Current-session features use only what was observable by 10:00; features from other sessions use only sessions that had ended and matured by then; refits see only the past; day-level calibration uses only past days, with each day's score stored on that day under the model then in force. The replay engine is to be the only code that reads unplug times. A scramble test replays the whole pipeline, linear programme included, with every field that becomes known after a cutoff altered, including the later fields of sessions already plugged in at the cutoff, and requires every earlier offer to be identical.
- **Sealed test data.** Caltech sessions from 1 March 2020 open only once the `analysis-plan` git tag exists; the Jet Propulsion Laboratory site is analysed once, after the settings are frozen. Both periods were looked at while the project was scoped, and the looks are listed in `docs/prior-looks.md`: stay statistics for every Caltech day from March 2020 to September 2021, how often fixed and rolling per-car stay bounds held, a count-based fleet promise fitted in 2019 that held on no 2021 weekday, the day-to-day dispersion of early departures, and stay statistics at JPL. No learned per-car bound, no offer from the inner approximation, no day-level calibration rule and no learned-against-population comparison has been computed on them. The test is a frozen retrospective comparison on eras that were partly explored, and is described as such.
- **Every output records how it was made.** Each run writes a manifest with the code commit, parameters, package versions and a fingerprint of its input.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run fleetoffer demo
```

The demo is to run the whole pipeline on made-up sessions with known flexibility sets, so the linear programme is checked where the answer is known. It is not a result.

## Data

Session data come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), available for educational and research use after registration; they are not redistributed here. With an access token in `ACN_API_TOKEN`:

```bash
uv run fleetoffer fetch caltech    # stored under data/, never committed
uv run fleetoffer verify caltech   # same data as this repository's fingerprints/caltech.csv?
uv run fleetoffer percar           # per-car models and baselines, months through August 2019
uv run fleetoffer checks           # checks D2 to D12 and the offers of every arm, May-August 2019
```

The energy delivered after 10:00 is not a field of the session record. The per-session time series that would give it failed a quality check fixed before it was fetched (300 sampled sessions: 85% usable against a 95% threshold, median energy error 5.1% against 5%), so it is estimated from each session's recorded energy and charging-done time, and the energy ceiling is approximate. `docs/data.md` describes what is downloaded, the cleaning rules and their counts, the time-series check and the estimate, the sealed periods, and a check of the download against an independent public copy. If you use the data, cite Lee, Li and Low (2019).

## Closest work

Al Taha, Vincent and Bitar (2024) compute inner approximations of the aggregate flexibility of a fleet whose individual sets are given; Mukhi, Loho and Abate (2026) give the exact aggregate set for this class of sets as a generalised polymatroid. Mukhi, Qu, You and Abate (2025) certify an aggregate set for cars drawn independently from one known distribution and name per-vehicle distributions as a limit of that model; Yan, Parise and Bitar (2022) bound the sample size of a chance-constrained decision when the distribution drifts by a known amount; Gibbs and Candès (2021) keep long-run coverage under arbitrary shift without estimating it. On the prediction side, Lee, Li and Low (2019) predict individual departures on this dataset and Genov and others (2023) forecast session flexibility on it with gradient-boosted trees and mixtures, measuring the effect of driver registration; Huber, Dann and Weinhardt (2020) use per-session quantile forecasts of parking time and energy in charging coordination; Winschermann, Bañol Arias, Hoogsteen and Hurink (2023) measure, at an office car park, what individual averages of energy and dwell time add to a charging strategy, the nearest published answer to the question asked here. Su, Xiang, Liu and Nie (2026) aggregate per-vehicle sets under a modelled chance of early departure within the day, with no learning from each vehicle's history and no change across days.

Three lines of work take the other route or a neighbouring one. Thrän, Mareček, Shorten and Green (2025), Brinkel, Visser, van Sark and AlSkaif (2023), Essayeh and others (2025) and Chen and Hu (2026) forecast fleet-level flexibility boundaries or their parameters directly; Thrän and others write that aggregation avoids predicting individual behaviour. Wang, Wang, Wang and Dabbaghjamanesh (2026) calibrate zone-level charging forecasts with conformalised quantile regression and score a vehicle-to-grid envelope by delivery shortfall, as two separate modules; Wang, Abu-Znad, Du and Sao (2025) learn corrections to an inner-box aggregate from past schedule deviations without scoring delivery in time order; Taheri, Kekatos, Veeramachaneni and Zhang (2022) learn an aggregator's feasible set from labelled feasible and infeasible schedules and approximate it from inside by the shape a market prescribes. Lunde, Damm, Gade and Kazempour (2025) bid an EV fleet under the Danish rule that a reserve must be available with at least 90% probability, the operational form of a 90% target; Zhang, Hu, Munsing, Moura and Song (2019) offer regulation capacity from distributed resources, electric vehicles among them, under a data-driven distributionally robust chance constraint on the service delivered; Li and Li (2024) and Abedrabboh and Al-Fagih (2025) treat uncertain departures with a distributionally robust model and a worst-case model, the second beside re-solving the schedule as cars' availability is revealed. Two 2026 papers whose full text was not available when this was written (Luo, Chen and Zhu; Park and others) describe aggregate-level forecasting coupled to scheduling and robust virtual-battery aggregation; they are listed in the plan as unresolved. None of these builds the aggregate set from learned per-car bounds and scores it by replayed delivery in time order across a change in behaviour; this project does, and compares it with population offers calibrated in the same way.

## Layout

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
note/             the note's page, with a named field for each result (`fleetoffer note`)
fingerprints/     SHA-256 of every downloaded month
docs/             analysis plan, what the tag fixes (registration), data notes, theory with theorem
                  statements, earlier looks, written predictions, discovery log
```

## References

- Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4), 3142–3154, 2025 (online April 2024). doi 10.1109/TSG.2024.3384871. arXiv 2207.07067.
- Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments", IEEE Control Systems Letters 6, 2022. arXiv 2205.03748.
- Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility", HSCC 2025. arXiv 2405.08232.
- Mukhi, Loho and Abate, "Exact characterization of aggregate flexibility via generalized polymatroids", IEEE Transactions on Smart Grid 17(3), 2131–2142, 2026. doi 10.1109/TSG.2026.3659513.
- Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS 2021. arXiv 2106.00170.
- Thrän, Mareček, Shorten and Green, "Reserve provision from electric vehicles: aggregate boundaries and stochastic model predictive control", IEEE Transactions on Power Systems 40(5), 2025. doi 10.1109/TPWRS.2025.3539863.
- Zhang, Hu, Munsing, Moura and Song, "Data-driven chance-constrained regulation capacity offering for distributed energy resources", IEEE Transactions on Smart Grid 10(3), 2713–2725, 2019. doi 10.1109/TSG.2018.2809046 (abstract only).
- Lee, Li and Low, "ACN-Data: analysis and applications of an open EV charging dataset", ACM e-Energy 2019.
- Genov, De Cauwer, Van Kriekinge, Coosemans and Messagie, "Forecasting flexibility of charging of electric vehicles: tree and cluster-based methods", Applied Energy 353, 121969, 2023. doi 10.1016/j.apenergy.2023.121969.
- Huber, Dann and Weinhardt, "Probabilistic forecasts of time and energy flexibility in battery electric vehicle charging", Applied Energy 262, 114525, 2020. doi 10.1016/j.apenergy.2020.114525.
- Winschermann, Bañol Arias, Hoogsteen and Hurink, "Assessing the value of information for electric vehicle charging strategies at office buildings", Renewable and Sustainable Energy Reviews 185, 113600, 2023. doi 10.1016/j.rser.2023.113600.
- Su, Xiang, Liu and Nie, "An inner approximation aggregation of electric vehicles considering uncertain dynamic characteristics of users", Sustainable Energy, Grids and Networks 47, 2026. doi 10.1016/j.segan.2026.102350.
- Brinkel, Visser, van Sark and AlSkaif, "A novel forecasting approach to schedule aggregated electric vehicle charging", Energy and AI 14, 100297, 2023. doi 10.1016/j.egyai.2023.100297.
- Essayeh, Vilan, Homaee and Vahidinasab, "EV fleet flexibility estimation and forecasting for V2X applications", arXiv 2502.06435, 2025.
- Wang, Wang, Wang and Dabbaghjamanesh, "Calibrated probabilistic forecasting and measured discharge physics for deliverable electric vehicle flexibility", World Electric Vehicle Journal 17(7), 367, 2026. doi 10.3390/wevj17070367.
- Wang, Abu-Znad, Du and Sao, "Learning-assisted aggregate flexibility characterization for EV charging stations considering scheduling deviation uncertainty", IEEE ITEC 2025. doi 10.1109/ITEC63604.2025.11098065.
- Taheri, Kekatos, Veeramachaneni and Zhang, "Data-driven modeling of aggregate flexibility under uncertain and non-convex device models", IEEE Transactions on Smart Grid 13(6), 4572–4582, 2022. doi 10.1109/TSG.2022.3185532 (abstract only).
- Li and Li, "Distributionally robust evaluation for real-time flexibility of electric vehicles considering uncertain departure behavior and state-of-charge", IEEE Transactions on Smart Grid 15(4), 4288–4291, 2024. doi 10.1109/TSG.2024.3397057 (abstract only).
- Chen and Hu, "Evaluating and forecasting the aggregate flexibility of large-scale electric vehicle charging and swapping facilities", IEEE Transactions on Transportation Electrification 12(2), 2815–2829, 2026. doi 10.1109/TTE.2026.3655071 (abstract only).
- Abedrabboh and Al-Fagih, "Cost-effective scheduling and flexibility provision of EV aggregators under departure uncertainty", International Universities Power Engineering Conference (UPEC), 2025. doi 10.1109/UPEC65436.2025.11279874 (abstract only).
- Lunde, Damm, Gade and Kazempour, "Aggregator of electric vehicles bidding in Nordic FCR-D markets: a chance-constrained program", IEEE Transactions on Power Systems, 2025. doi 10.1109/TPWRS.2025.3558868.
- Luo, Chen and Zhu, "Coupling flexibility forecasting and day-ahead scheduling for EV aggregators: a frequency-enhanced Transformer approach", SSRN, 2026. doi 10.2139/ssrn.7316943 (abstract only).
- Park and others, "Robust virtual battery optimization for EV fleet aggregation: operational feasibility in day-ahead and real-time markets", eTransportation, 2026. doi 10.1016/j.etran.2026.100578 (title only).

## Licence

Code: MIT. Data: see above.

