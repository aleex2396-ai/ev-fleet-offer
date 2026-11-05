# ev-fleet-offer

**Does a per-car model pay at fleet level? A charging-flexibility offer built from learned per-car bounds, calibrated to be delivered on 90% of days and scored by replay, before and across a real change in who uses a site and how.**

An aggregator that schedules electric-vehicle charging can sell flexibility to the grid only if it can say in advance how much charging it can shift and then deliver it. At a car park the offer depends on the cars plugged in: when each will leave, and how much energy each will still take. Both are uncertain, and both change over time.

The offer can be built from a **population** view (every present car gets bounds from the site's recent sessions, given only how long it has already been plugged in) or from a **learned** view (each car gets its own bounds, predicted from what is known about it and its driver at the moment the offer is made). This project builds both on the public Caltech ACN-Data charging sessions and turns each into a fleet offer with the inner-approximation method of Al Taha, Vincent and Bitar (2024, Theorem 2). Each offer is calibrated on past days to a target of delivery on 90% of days, and both are scored by replaying them against the cars that were actually present. Three things are measured: how large each offer is beside the delivery rate it achieves, how much the learned model adds over the population view under the same calibration, and how both change across the 2020 campus closure, after which the median stay at the site fell from about six hours to under two.

The result is one of three sentences, each worth reporting: per-car learning offers more flexibility at the same delivery rate and keeps its margin through the change; or, once both are calibrated by day, it offers no more than the population view, so per-car accuracy does not survive aggregation; or offers calibrated car by car fail at fleet level right after the change, on days when many drivers' errors share a sign, and restoring 90% of days costs a measured share of the offer. Shared errors within a day are one explanation to be tested, not a conclusion written in advance.

## Status

In development. No results from real data have been produced yet. The analysis plan (`docs/analysis-plan.md`: decision, offers, models, hypotheses with pass and fail criteria, the checks to run on development data, and the list of earlier looks at the data) is frozen by the git tag `analysis-plan` before any confirmatory run and before any sealed data are opened. "Pre-registered" below means fixed in that document before the tag.

## Design

- **The decision.** At 10:00 on each weekday, for the cars plugged in at that moment, a set of fleet charging profiles for 10:00–20:00 in 15-minute steps. Nothing after 10:00 is used; no arrivals are forecast.
- **Per-car sets.** Each car's flexibility set is a charging-only special case of the generalised battery model of Al Taha, Vincent and Bitar (2024, equation (2)), built as in their Example 1 but without discharge and with an energy ceiling in place of the energy requirement: power between 0 and a cap while plugged in, zero after a deadline, and cumulative energy after 10:00 at most what the car will still take. The set asks nothing of the car: charging nothing is always allowed, so "delivered" means the fleet profile can be split within caps, deadlines and ceilings, not that each driver is charged. The energy given to the cars under each delivered profile is reported beside the delivery rate, and a version with a required-energy floor per car is a stated sensitivity. The learned set uses one-sided bounds, a deadline no later than the car is predicted to leave and an energy ceiling no higher than it is predicted to take; a short proposition in `docs/theory.md` shows that the offered set then lies inside the fleet's true set whenever every car's two bounds hold (a sufficient condition, not a necessary one).
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
```

The energy delivered after 10:00 and the charger's pilot signal are not fields of the session record; they come from the per-session time series, which the API serves one session per request. `docs/data.md` describes what is downloaded, the cleaning rules and their counts, the coverage of the time series, the sealed periods, and a check of the download against an independent public copy. If you use the data, cite Lee, Li and Low (2019).

## Closest work

Al Taha, Vincent and Bitar (2024) compute inner approximations of the aggregate flexibility of a fleet whose individual sets are given; Mukhi, Loho and Abate (2026) give the exact aggregate set for this class of sets as a generalised polymatroid. Mukhi, Qu, You and Abate (2025) certify an aggregate set for cars drawn independently from one known distribution and name per-vehicle distributions as a limit of that model; Yan, Parise and Bitar (2022) bound the sample size of a chance-constrained decision when the distribution drifts by a known amount; Gibbs and Candès (2021) keep long-run coverage under arbitrary shift without estimating it. On the prediction side, Lee, Li and Low (2019) predict individual departures on this dataset and Genov and others (2023) forecast session flexibility on it with gradient-boosted trees and mixtures, measuring the effect of driver registration; Huber, Dann and Weinhardt (2020) use per-session quantile forecasts of parking time and energy in charging coordination; Winschermann, Bañol Arias, Hoogsteen and Hurink (2023) measure, at an office car park, what individual averages of energy and dwell time add to a charging strategy, the nearest published answer to the question asked here. Su, Xiang, Liu and Nie (2026) aggregate per-vehicle sets under a modelled chance of early departure within the day, with no learning from each vehicle's history and no change across days.

Three lines of work take the other route or a neighbouring one. Thrän, Mareček, Shorten and Green (2025), Brinkel, Visser, van Sark and AlSkaif (2023) and Essayeh and others (2025) forecast fleet-level flexibility boundaries directly; Thrän and others write that aggregation avoids predicting individual behaviour. Wang, Wang, Wang and Dabbaghjamanesh (2026) calibrate zone-level charging forecasts with conformalised quantile regression and score a vehicle-to-grid envelope by delivery shortfall, as two separate modules; Wang, Abu-Znad, Du and Sao (2025) learn corrections to an inner-box aggregate from past schedule deviations without scoring delivery in time order. Lunde, Damm, Gade and Kazempour (2025) bid an EV fleet under the Danish rule that a reserve must be available with at least 90% probability, the operational form of a 90% target. Two 2026 papers whose full text was not available when this was written (Luo, Chen and Zhu; Park and others) describe aggregate-level forecasting coupled to scheduling and robust virtual-battery aggregation; they are listed in the plan as unresolved. None of these builds the aggregate set from learned per-car bounds and scores it by replayed delivery in time order across a change in behaviour; this project does, and compares it with population offers calibrated in the same way.

## Layout

```
src/fleetoffer/
  acn.py          ACN-Data download, sessions and per-session time series, with fingerprints
  data.py         sessions table, cleaning rules, seals
  interface.py    what a model may see at the decision time
  engine.py       event-by-event replay; the only reader of unplug times
  features.py     inputs known at 10:00, from sessions that had ended
  models.py       quantile models and baselines
  sets.py         per-car flexibility sets from bounds
  aggregate.py    the inner-approximation programme and the affine allocation
  calibrate.py    union bound, day-level conformal, adaptive conformal
  deliver.py      the nine-profile delivery checks and shortfalls
  checks.py       the look-ahead scramble test, per car and per day
  synthetic.py    made-up fleets with known flexibility sets
  figures.py      figures, each drawn from a table also written to disk
  pipeline.py     end-to-end runs with manifests
tests/            unit tests, including the look-ahead tests (no network needed)
fingerprints/     SHA-256 of every downloaded month
docs/             analysis plan, data notes, theory with theorem statements, earlier looks, discovery log
```

## References

- Al Taha, Vincent and Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations", IEEE Transactions on Smart Grid 16(4), 3142–3154, 2025 (online April 2024). doi 10.1109/TSG.2024.3384871. arXiv 2207.07067.
- Yan, Parise and Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments", IEEE Control Systems Letters 6, 2022. arXiv 2205.03748.
- Mukhi, Qu, You and Abate, "Robust aggregation of electric vehicle flexibility", HSCC 2025. arXiv 2405.08232.
- Mukhi, Loho and Abate, "Exact characterization of aggregate flexibility via generalized polymatroids", IEEE Transactions on Smart Grid 17(3), 2131–2142, 2026. doi 10.1109/TSG.2026.3659513.
- Gibbs and Candès, "Adaptive conformal inference under distribution shift", NeurIPS 2021. arXiv 2106.00170.
- Thrän, Mareček, Shorten and Green, "Reserve provision from electric vehicles: aggregate boundaries and stochastic model predictive control", IEEE Transactions on Power Systems 40(5), 2025. doi 10.1109/TPWRS.2025.3539863.
- Lee, Li and Low, "ACN-Data: analysis and applications of an open EV charging dataset", ACM e-Energy 2019.
- Genov, De Cauwer, Van Kriekinge, Coosemans and Messagie, "Forecasting flexibility of charging of electric vehicles: tree and cluster-based methods", Applied Energy 353, 121969, 2023. doi 10.1016/j.apenergy.2023.121969.
- Huber, Dann and Weinhardt, "Probabilistic forecasts of time and energy flexibility in battery electric vehicle charging", Applied Energy 262, 114525, 2020. doi 10.1016/j.apenergy.2020.114525.
- Winschermann, Bañol Arias, Hoogsteen and Hurink, "Assessing the value of information for electric vehicle charging strategies at office buildings", Renewable and Sustainable Energy Reviews 185, 113600, 2023. doi 10.1016/j.rser.2023.113600.
- Su, Xiang, Liu and Nie, "An inner approximation aggregation of electric vehicles considering uncertain dynamic characteristics of users", Sustainable Energy, Grids and Networks 47, 2026. doi 10.1016/j.segan.2026.102350.
- Brinkel, Visser, van Sark and AlSkaif, "A novel forecasting approach to schedule aggregated electric vehicle charging", Energy and AI 14, 100297, 2023. doi 10.1016/j.egyai.2023.100297.
- Essayeh, Vilan, Homaee and Vahidinasab, "EV fleet flexibility estimation and forecasting for V2X applications", arXiv 2502.06435, 2025.
- Wang, Wang, Wang and Dabbaghjamanesh, "Calibrated probabilistic forecasting and measured discharge physics for deliverable electric vehicle flexibility", World Electric Vehicle Journal 17(7), 367, 2026. doi 10.3390/wevj17070367.
- Wang, Abu-Znad, Du and Sao, "Learning-assisted aggregate flexibility characterization for EV charging stations considering scheduling deviation uncertainty", IEEE ITEC 2025. doi 10.1109/ITEC63604.2025.11098065.
- Lunde, Damm, Gade and Kazempour, "Aggregator of electric vehicles bidding in Nordic FCR-D markets: a chance-constrained program", IEEE Transactions on Power Systems, 2025. doi 10.1109/TPWRS.2025.3558868.
- Luo, Chen and Zhu, "Coupling flexibility forecasting and day-ahead scheduling for EV aggregators: a frequency-enhanced Transformer approach", SSRN, 2026. doi 10.2139/ssrn.7316943 (abstract only).
- Park and others, "Robust virtual battery optimization for EV fleet aggregation: operational feasibility in day-ahead and real-time markets", eTransportation, 2026. doi 10.1016/j.etran.2026.100578 (title only).

## Licence

Code: MIT. Data: see above.

