# Data

## Source and terms

All sessions come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), the public record
of the Adaptive Charging Network's car parks. Access needs a free token; registration states
that the data are for "educational and research purposes". The data are not redistributed in
this repository. Cite Lee, Li and Low (e-Energy '19) when using them.

Two sites are used:

| Site | Role in this project | Months in the API |
|---|---|---|
| Caltech campus car park | development (before March 2020) and main test (from 1 March 2020) | April 2018 to September 2021 |
| Jet Propulsion Laboratory (JPL) | replication, analysed once after the settings are frozen | September 2018 to September 2021 |

Caltech's development period, 25 April 2018 to 29 February 2020, is split in time order
(analysis plan, Section 7):

| Block | Dates | Used for |
|---|---|---|
| Training | 25 April 2018 to 30 April 2019 | fitting models; the base set of the offer |
| Model selection | May to August 2019 | choosing settings, which are then frozen; warming up the day-level calibration |
| Frozen development evaluation | September 2019 to February 2020 | hypotheses H1 and H4, with nothing chosen on it |

The period contains a real change that is not sealed: charging became paid for sessions
linked to a driver on 1 November 2018 (Lee, Li and Low, 2019).

The third site in the API (an office car park, `office001`) has too few drivers to carry a
claim and is not used.

## Download

`fleetoffer fetch <site>` asks the API for one calendar month at a time: every session whose
plug-in time falls in that month in UTC, start included, end excluded, 100 sessions a page. A
month is accepted only if the number of sessions received equals the total the server
reports and no session appears twice; otherwise it is asked for again. Each month is stored
as canonical JSON (sessions sorted by their database ID, keys sorted, no whitespace),
gzip-compressed, under `data/raw/<site>/`.

The SHA-256 of each month's canonical JSON is committed in `fingerprints/<site>.csv`, with
the session count. `fleetoffer verify <site>` recomputes them, so anyone with a token can
confirm they hold exactly the data analysed here.

| Site | Sessions | First and last plug-in (Pacific time) | Months with no sessions |
|---|---|---|---|
| Caltech | 31,424 in 42 months | 25 April 2018 to 13 September 2021 | September and October 2020 |
| JPL | 33,638 in 37 months | from September 2018 to September 2021 | September and October 2020 |

Both sites fall to a few dozen sessions a month from April 2020, during the campus closures.
The counts per month are in `fingerprints/`. JPL's first and last plug-in dates are not given
here because finding them means loading its sessions.

## Sealed data

Two parts of the download are sealed in code (`data.py`), so the public history shows they
were not read before the analysis was fixed:

- **Caltech from 1 March 2020 (Pacific time)**, the main test period. The loaders return only
  sessions plugged in before it. They open the rest only once the analysis plan's git tag,
  `analysis-plan`, exists in the repository.
- **JPL**, the replication site. The loaders refuse it unless explicitly unsealed, which only
  the replication run does, after the settings are frozen.

A test checks that no library code unseals either. Earlier looks at both, made before the
seals existed, are listed in `prior-looks.md`.

## Cleaning

Cleaning drops only records that cannot be a charging session, in this order, and counts each:

1. `no_times`: the session ID, plug-in time or unplug time is missing.
2. `not_positive`: the car unplugs at or before it plugs in.
3. `duplicate`: a session ID already seen; the first is kept.

Nothing is dropped for being unusual: very short and very long stays stay in, because an
offer made at 10:00 meets them in practice.

Caltech: of 31,424 records, none was dropped by any of the three rules (counted before the
test-period seal was added; before 1 March 2020, 27,584 records, none dropped). JPL's counts
will be reported with the replication run: its months are downloaded and fingerprinted, but
its sessions are not loaded into any table before then.

## What drivers typed

Drivers could enter in the charging app how long they would stay and how much energy they
wanted. Every entry is kept with the time it was recorded (`data.load_user_inputs`). Entries
are never used as the true departure. `data.session_flags` adds descriptive flags:

| Flag | Meaning |
|---|---|
| `claimed` | the session is linked to a registered driver |
| `typed` | the driver entered at least one plan |
| `input_lag_s` | seconds from plug-in to the first entry being recorded |
| `input_with_plugin` | that entry was recorded within one second of the plug-in |
| `typed_minutes` | the stay the first entry stated |
| `typed_60`, `typed_480` | the first entry stated exactly one hour, or exactly eight hours |
| `edits` | how many times the entry was changed afterwards |

They describe how the data were recorded; none is used to drop sessions. As inputs at
10:00, a typed entry is used only if it was recorded before 10:00, and a session counts as
linked to a driver only if the link was recorded before 10:00; the final `claimed` flag and
the number of `edits` are never used as inputs (analysis plan, Section 7). Typed stays of
exactly 60 and 480 minutes are flagged because they may be app defaults; no dated source
says so, and they are not called defaults here.

**When a session counts as linked, and which plan is used.** A session receives a driver
identifier when the driver claims it in the app, which records an entry. The link is dated
by the session's first entry (`data.driver_links`, column `linked_time`); a session with a
driver identifier but no dated entry would count as anonymous at every decision. At 10:00 the
engine shows a car's driver only if `linked_time` is before 10:00, and shows the app entries
recorded before 10:00 (`data.app_entries`); the plan used is the latest of those. Among the
22,738 Caltech sessions plugged in before 1 September 2019, every one of the 8,934 with a
driver identifier has a dated entry; the first entry came a median of 0.4 minutes after
plug-in, within 20 minutes for 99% of them, and at most 6.2 days later. Of the 4,011 car-days
at 10:00 in that period with an entry recorded before 10:00, 293 had more than one, and on
204 the latest differed from the first. Every session with a driver identifier has an app
entry, and no session without one has any, so at 10:00 a car counts as linked exactly when it
has a plan recorded before 10:00: on all 9,148 car-days at 10:00 up to August 2019 the two
inputs agree. Driver history and a typed plan therefore always come together, and the
comparison of known and anonymous cars (check D6) is also a comparison of cars with and
without a typed plan.

## Academic calendar and holidays

Inputs at 10:00 include three day flags (`academic.day_flags`): instruction in session (from
the beginning of instruction to the end of a term, study periods and final examinations
included), summer (between the end of the third term and the next beginning of instruction),
and Institute holiday. Days that are neither are the winter and spring recesses. Terms, and
the holidays they name (Thanksgiving and the day after, Martin Luther King Day, Presidents'
Day, Memorial Day), come from the Office of the Registrar's academic calendars for 2017-18 to
2021-22 (registrar.caltech.edu, documents 7450, 7449, 17372, 7212 and 7213). Independence Day,
Labor Day and the year-end closure come from Caltech's events calendar, which lists
Independence Day on 4-5 July 2019, 3 July 2020 and 5 July 2021, Labor Day on 2 September 2019
and 7 September 2020, and Institute holidays from 24 December to 1 January for 2018-19 and
2020-21. Where no entry was found (Independence Day 2018, Labor Day 2018 and 2021, the
closures of 2019-20 and 2021-22) the federal date and 24 December to 1 January are used. The
sources were read on 5 November 2026. The calendar covers January 2018 to June 2022 and refuses
other dates. The same flags are given at JPL, whose own schedule (including any alternate
Fridays off) is not modelled.

## Per-session time series

The energy a car takes after 10:00, and the pilot signal the charging network sent it, are
not fields of the session record. They would come from the API's time-series endpoint
(`sessions/<site>/ts`), which takes the same plug-in-time windows as the session endpoint and
returns, for each session, the charging current and the pilot signal at the network's sampling
times, about every four seconds. Its records are large (about 0.6 MB of JSON per session), so
pages hold 10 sessions. The pilot signal is the network's own current limit sent to the car,
not a capability of the vehicle, and is never used as an input at 10:00.

`fleetoffer series caltech` fetches the development period this way one UTC day at a time
(a month of records is too large to hold at once), from the first session's day to the start
of the sealed test period, so that no record from it is requested; each day is stored and
fingerprinted like the session months (`fingerprints/caltech-series.csv`, with each day's UTC
window). `fleetoffer series caltech --sample` fetches the sample of check D1, one request
per session by a one-second window on its plug-in time (`fingerprints/caltech-series-d1-sample.csv`).
`fleetoffer verify caltech --series` recomputes both.

**The rules of check D1**, fixed in `pipeline.py` and `data.py` before any series was fetched:

- *Sample:* 300 development sessions drawn without replacement from all 27,584, by
  `numpy.random.default_rng(20261105)` over the sorted session IDs.
- *Usable series:* a charging-current record with as many timestamps as values, at least two,
  every timestamp readable and later than the one before, and every value a finite number.
  Anything else is counted under its reason (`data.SERIES_REASONS`).
- *Energy:* each sample's current, negative values counted as zero, holds until the next
  sample; nothing is counted after the last one. Power is current times 208 V, the line
  voltage at which a 32 A charger is the plan's 6.66 kW.
- *Error:* |integrated − recorded| / recorded, for usable sessions whose recorded energy is
  above zero; sessions with zero recorded energy are counted separately.
- *Pass:* at least 95% of the sample usable, and a median error of at most 5% (analysis plan,
  Section 10a). `fleetoffer d1` writes the per-session table, the summary and a manifest.

**Result.** `fleetoffer d1`, on the stored sample, fails on both counts:

| | Rule | Sample |
|---|---|---|
| Usable series | at least 95% | 256 of 300 (85.3%) |
| Median error, usable sessions | at most 5% | 5.09% (256 sessions; 48% within 5%) |

Of the 44 sessions without a usable series, 8 have no series at all (recorded energy 0.9 to
34.4 kWh) and 36 have timestamps that are not strictly increasing. No sampled current is
negative.

**What the failure consists of.** The same command writes these readings
(`d1_readings.json`) only when the check fails; they were taken after the result and change
neither the rule nor the result.

- All 36 non-increasing series repeat a timestamp; none goes backwards. In the median such
  series, 0.02% of samples repeat a time.
- Most of the error is one common scale. The ratio of integrated to recorded energy has a
  median of 0.951 (5th to 95th percentile, 0.923 to 1.097). The median ratio would be 1 at
  about 219 V rather than the 208 V fixed in the rule; the session record does not report
  voltage, so this is a reading, not a finding. With that one scale removed, the median error
  is 1.5% and 89% of sessions are within 5%.
- No sampled current exceeds 34.7 A and no pilot signal exceeds 32 A.

The readings point at the rule's two fixed choices, 208 V and a strictly increasing time
order, more than at the series. Changing either now would be choosing the rule after seeing
its result, so the check stands as failed and the fallback it states applies.

**The fallback** (analysis plan, Section 10a). Energy between 10:00 and 20:00 is estimated
from the session record: the recorded energy is spread evenly from plug-in to the end of
charging, which is the charging-done time if it is recorded and later than plug-in and
otherwise unplug; a charging-done time after unplug counts as unplug
(`engine.energy_in_window`). Of the 27,584 development sessions, the charging-done time is
missing in 721, not after plug-in in 27, and after unplug in 1,131. The estimate uses the
recorded total and the charging-done time, which are known only later, so it scores days and
trains models on past sessions and is never an input at 10:00.

On the sample, for the 95 usable sessions plugged in before 10:00 on a weekday and still
plugged in at 10:00 that day, the estimate is compared with the series scaled to the recorded
total, so that only the timing differs. The series give a median of 2.23 kWh between 10:00
and 20:00. The median absolute gap is 0.23 kWh (90th percentile 1.67 kWh), and 8.9% of the
series value in the median among the 65 cars above 0.5 kWh. The estimate leans late: 513.5 kWh
in total against 469.5 kWh (9.4% more), and 70 cars above 0.5 kWh against 65, as expected if
charging runs faster early in a session than the flat spread assumes.

**What follows, as the plan states it** (Sections 10 and 10a). The energy ceiling is
approximate and is called so wherever it is used. The realised-cap sensitivity, which needed
the pilot signal of every session, is dropped. The power cap is an assumption throughout:
check D2 needs measured power at each station, which only the series give, so 6.7 kW stands
for all stations, with 3.3 kW and 11 kW as sensitivities. The series of all development
sessions are not fetched; `fleetoffer series caltech` remains for anyone who wants them. The
sample's series stay stored and fingerprinted as the record of the check.

## Checks before the tag

The checks of the analysis plan, Section 10a, are run by `fleetoffer checks` on Caltech data
through August 2019, except D3, which counts cars on the model-selection and frozen
evaluation months as its rule states. D1 is above. The results that do not involve an offer:

- **D2, caps.** Not measurable: it needs each station's measured power, which only the time
  series give, and they failed D1. 6.7 kW stands for every station as an assumption. For
  context only, the average power over each session (recorded energy over the time from
  plug-in to charging-done) is above 6.7 kW for 2.7% of the 22,237 sessions with both times,
  above 7.5 kW for 0.23%, and its 99th percentile is 6.83 kW. An average understates the peak,
  so this neither confirms nor rejects the cap.
- **D3, cars with energy at 10:00.** Over the 218 weekdays of May 2019 to February 2020, the
  median number of present cars that took more than 0.5 kWh after 10:00 is 14 and the 10th
  percentile is 7, so 10:00 is kept (the rule asks for at least 10 and 4). 17 weekdays have
  fewer than five such cars and are not scored: 3 of the 88 model-selection weekdays and 14 of
  the 130 frozen evaluation weekdays, which leaves 116 scored weekdays there. Taken alone, the
  frozen evaluation months have a 10th percentile of 3.9; the rule pools both periods.
  Counts per weekday: `results/checks/d3_counts.csv`.
- **D4, unclaimed sessions.** Of 9,128 car-days at 10:00, 56.2% are unclaimed, 16.4% have
  nothing left to take (an estimate of at most 10⁻⁹ kWh after 10:00), and 10.2% are both. That
  is under the 30% of the rule, so no ceiling is set to zero by rule and the claimed-only
  sensitivity is not added.
- **D5, typed requests in time.** On 222 of the 350 weekdays with cars that took energy, at
  least 30% of those cars had a typed request recorded before 10:00 (median share 83%). That
  is more than 200 weekdays, so the typed-request sensitivity is kept.
- **D6, driver history.** The share of present cars linked to a driver by 10:00 rises from
  under 3% before September 2018 to 82% in November 2018, when charging became paid for
  claimed sessions, and stays between 84% and 93% through August 2019; the share also with at
  least three earlier ended sessions follows it, between 82% and 92% from December 2018. Per
  month: `results/checks/d6_by_month.csv`. Known and anonymous cars are evaluated separately.
- **D7, energy as a proxy for need.** Among the 3,948 sessions with a typed request before
  10:00, each counted once, the recorded energy is a median 0.64 of the latest request (the
  rule asks for 0.8 to 1.2) and the rank correlation is 0.75 (at least 0.6). The ratio fails,
  so the results' headline says that the energy is scheduler-shaped: cars took about two
  thirds of what their drivers asked for, while larger requests still went with larger
  deliveries.
- **D12, the base set.** The base set's vector h₀, from every car present at 10:00 on a
  weekday through April 2019, has a power limit that falls from 5.06 kW at 10:00 to 0.27 kW at
  19:45 and is positive at all 40 steps, and an energy limit of 5.77 kWh; U₀ is non-empty with
  positive width at every step. The full vector is in `docs/registration.md`.

## Check against an independent public copy

SustainGym (Yeh et al., NeurIPS 2023 Datasets and Benchmarks Track) publishes four extracts
of the Caltech site under CC-BY-4.0, made from the same API by other people. The files used
here are pinned to commit `eac2a4d5ce4ccf44b13c290edac72532c68e30f5` of
[chrisyeh96/sustaingym](https://github.com/chrisyeh96/sustaingym) and checked by SHA-256
before use (`acn.py`). `fleetoffer snapshot-check <folder>` compares each extract with
the download over the same Pacific-time days:

| Extract | Sessions in extract | Sessions here | In both | Plug-in and unplug times equal (to 1 s) | Driver-link flag equal |
|---|---|---|---|---|---|
| May to August 2019 | 3,527 | 3,527 | 3,527 | 3,527 | 3,527 |
| September to December 2019 | 3,177 | 3,177 | 3,177 | 3,177 | 3,177 |
| February to May 2020 | 1,269 | 1,269 | 1,269 | 1,269 | 1,269 |
| May to August 2021 | 1,829 | 1,829 | 1,829 | 1,829 | 1,829 |

All 9,802 sessions agree exactly, including the time-zone conversion. This comparison was
run before the test-period seal was added and reports agreement counts only; since the seal,
`snapshot-check` compares the two extracts from before March 2020 and lists the other two as
sealed. The extracts have no driver IDs, and in their unclaimed rows the "estimated
departure" is a copy of the actual departure, so they are used here only for this check.