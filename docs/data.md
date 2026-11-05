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