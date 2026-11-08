# What the tag fixes

The analysis plan (Section 7) lists what is fixed before the sealed loader opens. This page
gives each item's value and where it lives in the code. The commit tagged `analysis-plan` is
the one that is run on the sealed periods; a change after the tag is reported as a deviation
beside the original result. Where the plan left a term undefined, the reading used is stated
here and in `docs/theory.md`.

## Decision and sets

| Item | Fixed as | Where |
|---|---|---|
| Decision time, horizon, step | 10:00 Pacific time on weekdays; 10:00 to 20:00; 40 steps of 15 minutes | `engine.MORNING`, `offer.STEPS`, `offer.STEP_H` |
| Present at 10:00 | Plugged in before 10:00 and not unplugged by then | `engine.fleet_replay` |
| Scored day | At least five present cars that took more than 0.5 kWh after 10:00 | `daylevel.SCORED_MIN_CARS`, `engine.ENERGY_REMAINING_KWH` |
| Per-car set | U(d, e): power in [0, c] up to the deadline, zero after, total energy after 10:00 at most e | `offer.py`; `docs/theory.md` |
| Power cap | 6.7 kW for every station, an assumption (check D2 cannot be measured, below) | `offer.CAP_KW` |
| Deadline | Unplug time (20:00 if still plugged in), rounded down to the 15-minute grid | `engine.fleet_replay` |
| Energy after 10:00 | Recorded total spread evenly from plug-in to charging-done (unplug if missing or odd), check D1's fallback; scores and trains only, never an input at 10:00 | `engine.energy_in_window`; `docs/data.md` |

## Arms, models and settings

| Item | Fixed as | Where |
|---|---|---|
| Arms | 1 oracle; 2 population at level 0.10; 3 learned, union bound; 4 learned, 5 population, 6 stratified, each with the primary day-level rule; 4s, 5s, 6s the same with the secondary rule | `devchecks.make_arm` |
| Population window and conditioning | Sessions ended in the last 90 days, quantiles conditional on the time already plugged in at 10:00 | `baselines.Population`, `percar.CHOSEN` |
| Stratified table | Plug-in hour bin, Friday or not, linked or not; a cell under 30 car-days drops the weekday type, then the hour, then the link | `baselines.Stratified` |
| Learned model | Gradient-boosted quantile trees: learning rate 0.1, 7 leaves, smallest leaf 20, 200 iterations, seed 0; 90-day history; refitted at the first local midnight of each month; no bound below 200 training car-days | `percar.CHOSEN`, `percar.FIXED` |
| Settings searched and the selection rule | The full grid and rule of `docs/theory.md`, "The per-car models and the settings searched" | `percar.GRID` |
| Quantile grid | 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 | `quantiles.LEVELS` |
| Trivial bound | Deadline at 10:00 and energy 0, used below level 0.002 | `quantiles.at_level`, `daylevel.at_levels` |
| Calendar | Caltech registrar and Institute holiday calendars; JPL gets the same flags | `academic.py`; `docs/data.md` |
| Censoring at refits (D11) | Sessions plugged in within 24 hours of a refit are left out of its training | `quantiles.EXCLUDE_H` |

## The offer and its checks

| Item | Fixed as | Where |
|---|---|---|
| Programme | (24) of Al Taha, Vincent and Bitar with α ≥ 0, irredundant rows, HiGHS interior point with crossover; the dual simplex method on a day the interior-point method returns no answer; a base set whose energy ceiling is at most 10⁻⁷ kWh is treated as empty (α = 0) | `offer.solve`, `offer.SOLVER`, `offer.TRIVIAL_BASE_KWH`; `docs/theory.md` |
| Base set U₀ | h₀ from the training cars through April 2019, the same for every arm, committed as a file and loaded by every run after the tag; the day-average base set of the paper's equation (11) as a sensitivity | `src/fleetoffer/h0.json`, `evaluation.base_set`; values under D12 below |
| Nine profiles | Most and least cumulative energy at 12:00, 14:00, 16:00 and 18:00, and the least-cost profile under the stylised tariff | `offer.checked_profiles`, `offer.TARIFF` |
| Delivered | All nine profiles within 10⁻⁴ kWh under the allocation fixed at 10:00 (primary); the hindsight split (secondary) | `offer.DELIVERY_TOL_KWH` |
| Size | Width of the offer's cumulative-energy band at 14:00, 16:00 and 18:00, averaged, in kWh, with α | `offer.SIZE_CHECKPOINTS` |
| Fallback | One-dimensional offer b + a·v. D9 called for it (below), so it is the offer of every registered arm in every run after the tag | `offer.solve_segment`, `evaluation.REGISTERED_FORM`; `docs/theory.md` |

## Day-level rules

| Item | Fixed as | Where |
|---|---|---|
| Day score | Smallest, over cars and both targets, of the largest grid level at which the bound held | `daylevel.day_score`; `docs/theory.md` |
| Primary rule | Adaptive conformal inference, days as steps: start 0.10, step γ = 0.03, target miss 0.10, 90-day window of scored days, updated as outcomes arrive; a level at or below 0 gives the trivial bound, one at or above 1 the top of the grid | `daylevel.Calibrated`, `daylevel.ACI_GAMMA` |
| Secondary rule | Split conformal at 0.10 on the same window, only where it holds at least 40 scored days (D8), otherwise its own copy of the primary rule | `daylevel.Calibrated(rule="split")` |
| Arm 3 levels | 0.10/(2n) for n present cars, used only where D10 supports the level | `daylevel.UnionBound`, `devchecks.D10_SUPPORTED` |
| Target | Delivered on 90% of scored days | `daylevel.TARGET_MISS` |

## Hypotheses and uncertainty

| Item | Fixed as | Where |
|---|---|---|
| Hypotheses | H1 to H4 with the thresholds of the plan, Section 9 | `docs/analysis-plan.md` |
| Primary contrast | H1: arm 4 against arm 5, relative difference of mean offer size on the scored weekdays of September 2019 to February 2020, with a paired week-block interval. H1 passes if the relative difference is at least 10% and the interval of the paired difference in kWh lies above zero; that interval is defined in every resample, and wherever the relative difference is defined (arm 5's total above zero) the two have the same sign. The relative interval is reported over the resamples in which it is defined, with their number | `resample.paired_interval`, `evaluation.paired` |
| H4, matched reliability | Each model's stay bound at the level split conformal gives on its own past car-days (90-day window, miss 10% of car-days), recomputed each day; both achieved coverages reported | `percar.matched` |
| Confirmatory and descriptive | H1 and H4 confirmatory; H2 and H3 checks; monthly and per-site contrasts descriptive | plan, Section 9 |
| Uncertainty | Whole local weeks (Monday to Sunday) drawn with replacement, both arms together, 10,000 draws, seed 0, 95% percentile interval; two-week blocks as the sensitivity | `resample.py` |

## Sealed periods

| Item | Fixed as | Where |
|---|---|---|
| Reopening | The first weekday of 2021 on which the average of cars present at 10:00 over the last ten weekdays, that day included, is at least five (each day's average uses that day and earlier weekdays only, so the rule runs in time order); H3's window starts there | `evaluation.reopening` |
| H3's frozen offer | Arms 4 and 5 (and 4d and 5d in the secondary analysis) whose learning stops on 1 March 2020: no refit, no new day score, the calibration level in force that day; delivered on fewer than 80% of the first 20 scored weekdays from the reopening | `evaluation.Frozen`, `evaluation.h3` |
| Recovery window | A 90% running average of delivered days over 20 scored weekdays, reached within 40 scored weekdays of the reopening, by arm 4 (arm 5 and the secondary arms reported beside it) | `evaluation.h3` |
| JPL | Every weekday of JPL's record, in two tracks, both starting from the pipeline as it stood at the end of the frozen development evaluation (every refit and day score through February 2020). Track 1 (primary): that pipeline frozen, shown no JPL driver history. Track 2: its Caltech car-days and day scores dropped, its fitted models and working miss level kept, learning from JPL outcomes as they mature and refitting once a JPL window holds 200 car-days | `evaluation.for_jpl`, `evaluation.run_jpl` |
| Uncertainty for H1 | One-week blocks; two-week blocks as the sensitivity | `evaluation.paired` |
| Shared tree fits | Within one run after the tag, arms whose per-car model is fitted with the same settings on the same training rows read one fit from the run's own store instead of each refitting (`quantiles.stored_fit`, keyed by a hash of the settings, the library version and every training row); no fit passes between runs. Each worker process runs the trees on one thread | `quantiles.FIT_STORE`, `quantiles.worker_pool` |
| Decision time and cap sensitivities | The decision at 09:00 (the horizon, checkpoints and tariff counted in steps from 09:00, so ending at 19:00) and power caps of 3.3 and 11 kW, each a separate replay from the first download month: the oracle and arms 4 and 5 under the registered design, and 1a, 4d and 5d under the secondary analysis, on the frozen evaluation and the sealed months, not at JPL; H1's contrast at each setting, descriptive | `evaluation.SENSITIVITIES` |
| Earlier looks | `docs/prior-looks.md` | |
| Predictions | Written before any fleet-layer development result was seen | `docs/predictions.md` |

## The checks before the tag

Run by `fleetoffer checks`; outputs in `results/checks/`. The numbers below are from a clean
rerun from committed code before the tag, whose results matched the first run's except for
solver run times. D2 to D7 and D12 are in `docs/data.md` with their numbers; in short:

| Check | Result | Consequence |
|---|---|---|
| D1, energy after 10:00 | Failed on its sample (`docs/data.md`) | The fallback estimate; the ceiling is approximate; the realised-cap sensitivity is dropped |
| D2, caps | Not measurable without the time series | 6.7 kW for every station, an assumption; 3.3 and 11 kW as sensitivities |
| D3, cars at 10:00 | Median 14, 10th percentile 7 over 218 weekdays | 10:00 kept; 17 weekdays not scored, 14 of them in the frozen evaluation (116 scored there) |
| D4, unclaimed with nothing left | 10.2% of car-days | Under 30%: no rule, no claimed-only sensitivity |
| D5, typed requests in time | 222 weekdays at 30% or more | Typed-request sensitivity kept |
| D6, driver history | 82% to 92% of present cars linked with three earlier sessions from December 2018 | Known and anonymous cars evaluated separately |
| D7, energy as a proxy | Median ratio 0.64, rank correlation 0.75 | Not an adequate proxy: the headline says the energy is scheduler-shaped |
| D8, calibration windows | The window held at least 40 scored days on all 87 fitted weekdays of May to August 2019 | The secondary rule ran as split conformal on each of them |
| D10, extreme quantiles | Supported levels: stay 0.002, 0.005; energy 0.002, 0.005, 0.01; the same as committed | Arm 3 uses these levels only |
| D11, censoring at refits | Applied in every refit | |
| D12, the base set | Non-empty, positive width at all 40 steps | h₀ below |

D9 (the programme's degeneracy and run time), the share of days on which α = 0, and the
week-to-week spread of offer sizes come from the offers themselves. They were computed after
the written predictions were stored (`docs/predictions.md`), on the 85 scored weekdays of May
to August 2019:

| Check | Result | Consequence |
|---|---|---|
| D9, the programme | The oracle's α was below 0.2 on 37.6% of scored weekdays (threshold 25%); none of the 704 solves of the eight learned and population arms took over 60 seconds (longest 1.1 s), and the oracle's, whose sets are larger, took at most 25 s | The one-dimensional fallback for every registered arm after the tag |
| α = 0 | On the fixed base set, α = 0 on every day for every arm but the oracle (37.6%); on the day-average base set, on 100% of days for arms 3 and 4, 95.3% for arm 6, 3.5% for arm 5 | Reported; the registered rule's learned offer is zero (`docs/data.md`) |
| Late outcomes | On the 83 scored weekdays of arms 4, 5 and 6, 78.3% of outcomes were known before the next decision; at most 4 were pending at once | Reported; with 4 pending at once the bound on the share of failed days in `docs/theory.md` becomes 10% + (a_S + 4.5γ)/(γT), 16.8% at 116 days from a₁ (computed) |
| Spread, registered rule | Every learned and population offer zero, so the spread is zero | No reading of H1's 10% is possible under the registered rule |
| Spread, secondary analysis | Day-average base set, calibrated on delivery: weekly difference sd 0.36 kWh, projected standard error over 26 weeks 0.071 kWh, 10% of the population offer 0.155 kWh | No threshold reset |

## Registered and not run

Three sensitivities of the analysis plan (Section 3) are not run. This was decided before the
tag, before any result from September 2019 onward existed, for the reasons given; each is a
limitation of the results.

| Sensitivity | Why it is not run |
|---|---|
| A required-energy floor per car at an upper quantile of its energy need, with the share of learned sets that are then empty | It needs a second per-car set form (a floor on the energy given by the deadline) with its own inner approximation, allocation and delivery checks; not built in the time available. A floor can only shrink the offer |
| Total offered power capped at a stated share of the car park's panel rating | The panel rating is not in the data and no source for it was found; the shared supply limit is not modelled |
| The cohort with a typed energy request before 10:00, scored on the remaining request | It needs a second energy target through the engine, the learned model and the population rule; not built in the time available. Check D7 found recorded energy an inadequate proxy for the driver's need (median ratio of recorded to requested energy 0.64), so every energy bound here bounds energy shaped by the site's scheduler |

At JPL the level sweep is run on the one-dimensional fallback offer only (above). If the runs
after the tag cannot all end in time, the secondary analysis in JPL's second track is the first
run left out, and is then reported as not run.

## Added after the first fleet run (May to August 2019)

The registered day-level rule gave a learned offer of zero on every scored weekday of May to
August 2019. The analyses below were chosen after that look; they are fixed here, before the
tag, for every later period, so their later results are prospective, but their origin is
exploratory. The registered design stays the primary result.

| Item | Fixed as | Where |
|---|---|---|
| Secondary analysis | Arms 4, 5 and 6 with the primary rule's adaptive conformal inference applied to a day score on delivery: the largest level of the 17 levels (0 and the quantile grid) at which the offer, made again from that morning's bounds once the outcome is known, was delivered under the fixed allocation at that level and every lower one; programme (24) on the day-average base set; the oracle on the same base set beside them (arms 4d, 5d, 6d, 1a) | `daylevel.DeliveryCalibrated`, `evaluation.SECONDARY` |
| H1 under both designs | Arm 4 against arm 5 and arm 4d against arm 5d; where both arms offer zero on every day, H1 reads "not testable: both arms offer zero" | `evaluation.paired` |
| Level sweep (descriptive) | For each scored weekday, model (4, 5, 6) and offer form (fixed base set, day-average base set, fallback), the offer at each of the 17 levels made from that morning's bounds: size, delivery under the fixed allocation and with the hindsight split, whether every bound held, the cars given a zero energy ceiling or a zero stay. At JPL the fallback form only (programme (24) at every level of every JPL day does not fit the time available) | `daylevel.LevelSweep`, `levels.py`, `evaluation.SWEEP_FORMS` |
| Hindsight ceiling | The sweep's curve with hindsight delivery: the largest offer of the form that a per-request split chosen with hindsight could deliver; an upper bound, not a method that could be deployed, and not the registered service | `levels.table` |
| Matched comparison | Per model and form, the largest mean offer among the swept levels delivered on at least 90% (and 95%) of scored weekdays, with no interpolation | `levels.matched` |
| Diagnostic, empty cars known | The sweep when the cars that truly take nothing after 10:00 (at most 10⁻⁹ kWh, check D4's zero) are known and dropped, and the per-car models are refitted on car-days that took energy; a perfect-knowledge yardstick for live charging status at 10:00, not a method that could be deployed | `LevelSweep(drop_empty=True)`, `fleet_replay(reveal_empty=True)` |
| Accounting of the perfect-knowledge offer | Per period and offer form, each at its own delivery rate shown beside it: the exact band of the true sets (a ceiling no single offer need reach), the oracle's offer, the largest swept offer of each model delivered on 90% of days under the fixed allocation (a level chosen with hindsight, not a method), the same with the empty cars known and with a split chosen with hindsight (an upper bound), and each day-level rule as run; at JPL track 1; for the sealed months also 2021 alone | `levels.accounting`, `fleetoffer accounting` |
| Zero ceilings at small levels | A proposition (`docs/theory.md`) stated after the first fleet run and before the sweep's counts were read: a lower bound at level q is zero exactly when the car's chance of taking nothing is at least q. Its readings, the share of cars with a zero ceiling at each level, with and without the empty cars known, beside the offer, are reported for May to August 2019 and every later period | `levels.table` (`zero_ceiling_share`, `zero_ceiling_share_with_energy`) |
| Rehearsal | Every run after the tag was first run end to end on Caltech development data (June to August 2019); its numbers are not results. It found four faults, fixed before the tag: the second site's sweep diagnostic did not reveal the empty cars; H1's zero exclusion was read on the relative interval, which is undefined in resampled weeks where the population offer is zero (now read on the kWh interval above); the report's late panel was fixed to 2021; and the reopening rule's ten-weekday window stopped at 1 January (it now reaches into December 2020, as the rule's wording allows) | `fleetoffer evaluate rehearsal-*`, `fleetoffer report --rehearsal` |

**h₀, the base set's power limit (kW) at the start of each 15-minute step**, from every car
present at 10:00 on a weekday through April 2019; energy limit 5.774 kWh:

| 10:00 | 10:15 | 10:30 | 10:45 | 11:00 | 11:15 | 11:30 | 11:45 |
|---|---|---|---|---|---|---|---|
| 5.057 | 5.027 | 4.993 | 4.951 | 4.906 | 4.847 | 4.761 | 4.685 |
| 12:00 | 12:15 | 12:30 | 12:45 | 13:00 | 13:15 | 13:30 | 13:45 |
| 4.600 | 4.514 | 4.433 | 4.372 | 4.312 | 4.247 | 4.168 | 4.120 |
| 14:00 | 14:15 | 14:30 | 14:45 | 15:00 | 15:15 | 15:30 | 15:45 |
| 4.075 | 4.033 | 3.967 | 3.885 | 3.820 | 3.755 | 3.666 | 3.561 |
| 16:00 | 16:15 | 16:30 | 16:45 | 17:00 | 17:15 | 17:30 | 17:45 |
| 3.409 | 3.237 | 2.988 | 2.742 | 2.423 | 2.074 | 1.716 | 1.423 |
| 18:00 | 18:15 | 18:30 | 18:45 | 19:00 | 19:15 | 19:30 | 19:45 |
| 1.136 | 0.907 | 0.727 | 0.605 | 0.477 | 0.391 | 0.319 | 0.268 |

## Changes after the tag

Each change below was made after the `analysis-plan` tag; the runs themselves are those of the
tagged commit, and every original reading stays beside the corrected one.

| Change | What and why | Where |
|---|---|---|
| H3 when the days are too few | The tagged code labels each clause of H3 "fail" when fewer than 20 scored weekdays follow the reopening, though a clause that needs 20 weekdays cannot be read on fewer. On the sealed months the reopening rule gives 2 August 2021 and the data end on 13 September 2021: 31 weekdays, 15 of them scored. Both clauses are reported as not testable, with these counts; the runs' stored "fail" labels are kept, and the report reads the clauses again from the stored numbers. No window is moved and no test is added | `evaluation.h3_readings`; `results_table` |
| Median cars in the zero-ceiling section | `docs/theory.md` gave the union bound's level as 0.29% at a median of 17.5 cars on the scored weekdays of May to August 2019; 17.5 is the median over all 88 weekdays from 1 May 2019. On the 85 scored weekdays the median is 18 cars and α/(2n) is 0.28%. Corrected in the text; no analysis uses the number | `docs/theory.md`, "Zero ceilings at small levels" |
| September 2019 to February 2020 on two processes | The run's last 17 jobs (the level sweep) were shared between the registered command and a second process on the same tagged checkout, which took the queue from its end; each job is saved as it ends, and the registered command, run again, read every saved job back and wrote the tables. No job, setting or reading changed | `options.run_saved` |
| Accounting shares on the days each row covers | The tagged accounting divided each row's mean offer by the perfect-knowledge mean over every scored weekday, but a swept model makes no offer on a day without a fit, so some rows cover fewer days. Each share is now taken over the days its row covers, and the count is shown. The effect is at most 0.4 points on May to August 2019 (84 of 85 days) and September 2019 to February 2020 (115 of 116). On the sealed months the rows cover 23 of 26 days, and 16 with the empty cars known, 9 of them in March 2020: the hindsight split was 61.2%, 51.0% and 35.1% for the learned, population and stratified models and is 67.7%, 56.4% and 38.8%; with the empty cars known 4.6%, 7.5% and 6.7%, now 5.0%, 8.1% and 7.4%; the forecasts 0.3%, 2.5% and 1.3%, now 0.3%, 2.7% and 1.4%. The 2021-only accounting, computed by the tagged report, is now shown beside the sealed months | `levels.same_days`; `fleetoffer accounting`, `fleetoffer report` |
| Figure 4 on scored weekdays | The tagged report drew Figure 4 and its summary from every weekday of the two runs, including unscored days and the months before each period; it now uses each period's scored weekdays | `fleetoffer report` |
| The adaptive rule's bound as run | The report quoted the bound on the share of failed days from a₁ = 0.10 (13.6% at 116 days); the rules entered the frozen evaluation at working levels of 0.10 to 0.31, which give 13.7% to 19.8%. No run or rule changed | `docs/theory.md`, the report |
| Late outcomes counted | `docs/theory.md` ("Late outcomes") says the share of days whose outcome is known before the next decision, and the most days pending at once, are reported; the tagged report did not compute them. Read from the frozen evaluation's logs: 85% of scored weekdays known before the next decision, at most 3 pending at once, which loosens the bound above to 16.1% to 22.1%. No run or rule changed | `note.late_outcomes` |
| Per-car scores | Section 4 of the plan reports pinball loss and coverage for every per-car model, known and anonymous drivers apart, and says so if the learned model does not beat each driver's own past or the stratified table; the tagged report did not show them. The report now gives them on May to August 2019 (from `fleetoffer percar`, run before the tag) and, for the learned model and the population rule, on the frozen evaluation's car-days, read from the run's predictions. On energy the learned model ties each driver's own past (0.683 against 0.684 kWh), and the report says so | `note.frozen_percar`, `note.percar_numbers` |
| Monthly lines across the closure | The monthly panels of Figures 2 and 3 joined March 2020 to July 2021 across the months with no scored weekday; the lines now break at a gap of more than 45 days | `figures._runs` |
| JPL with one job left out | The JPL run's last job (the stratified model calibrated on delivery, updating on JPL days) was still inside the solver on 8 November 2026 after about 35 hours of computing; the tagged code sets no time limit, and the longest single solve in any finished run took 182 s. JPL's tables were built from the 36 saved jobs by the tagged code, nothing run again. The job feeds one comparison only, the learned against the stratified model calibrated on delivery when updating, which no table or figure uses; it is registered and not run | `results/evaluate-jpl/NOTE.txt`; the run's manifest, `jobs_left_out` |
| An empty piece of the JPL perfect-knowledge run | The perfect-knowledge offer at JPL was run in eight half-year pieces; the first covers months with no JPL weekday, and its empty day table left the joined time column untyped, so the tagged code stopped at the monthly table. The piece holds no day, car or log row and is left out of the join; no number changes | `pipeline.offering_runs` |
