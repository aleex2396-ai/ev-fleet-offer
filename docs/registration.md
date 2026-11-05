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
| Programme | (24) of Al Taha, Vincent and Bitar with α ≥ 0, irredundant rows, HiGHS interior point with crossover | `offer.solve`, `offer.SOLVER`; `docs/theory.md` |
| Base set U₀ | h₀ from the training cars through April 2019, the same for every arm; the day-average base set of the paper's equation (11) as a sensitivity | `engine.training_base_set`; values under D12 below |
| Nine profiles | Most and least cumulative energy at 12:00, 14:00, 16:00 and 18:00, and the least-cost profile under the stylised tariff | `offer.checked_profiles`, `offer.TARIFF` |
| Delivered | All nine profiles within 10⁻⁴ kWh under the allocation fixed at 10:00 (primary); the hindsight split (secondary) | `offer.DELIVERY_TOL_KWH` |
| Size | Width of the offer's cumulative-energy band at 14:00, 16:00 and 18:00, averaged, in kWh, with α | `offer.SIZE_CHECKPOINTS` |
| Fallback | One-dimensional offer b + a·v, used for the whole sealed run if D9 calls for it | `offer.py`; `docs/theory.md` |

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
| Primary contrast | H1: arm 4 against arm 5, relative difference of mean offer size on the scored weekdays of September 2019 to February 2020, with a paired week-block interval | `resample.paired_interval(relative=True)` |
| H4, matched reliability | Each model's stay bound at the level split conformal gives on its own past car-days (90-day window, miss 10% of car-days), recomputed each day; both achieved coverages reported | `percar.matched` |
| Confirmatory and descriptive | H1 and H4 confirmatory; H2 and H3 checks; monthly and per-site contrasts descriptive | plan, Section 9 |
| Uncertainty | Whole local weeks (Monday to Sunday) drawn with replacement, both arms together, 10,000 draws, seed 0, 95% percentile interval; two-week blocks as the sensitivity | `resample.py` |

## Sealed periods

| Item | Fixed as | Where |
|---|---|---|
| Reopening | The first weekday of 2021 from which the running average over ten weekdays of cars present at 10:00 is at least five; H3's window starts there | plan, Section 7 |
| Recovery window | A 90% running average of delivered days over 20 weekdays, reached within 40 scored weekdays of the reopening | plan, Section 9 (H3) |
| JPL | Two tracks: the Caltech-trained pipeline with no JPL updates (primary), and the same pipeline updating from JPL outcomes as they mature | plan, Section 7 |
| Earlier looks | `docs/prior-looks.md` | |
| Predictions | Written before any fleet-layer development result was seen | `docs/predictions.md` |

## The checks before the tag

Run by `fleetoffer checks`; outputs in `results/checks/`. D2 to D7 and D12 are in
`docs/data.md` with their numbers; in short:

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
week-to-week spread of offer sizes come from the offers themselves; they are added with the
fleet-layer development results once the written predictions are stored.

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
