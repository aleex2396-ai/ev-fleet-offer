# Data

## Source and terms

All sessions come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), the public record
of the Adaptive Charging Network's car parks. Access needs a free token; registration states
that the data are for "educational and research purposes". The data are not redistributed in
this repository. Cite Lee, Li and Low (e-Energy '19) when using them.

Two sites are used:

| Site | Role in this project | Months in the API |
|---|---|---|
| Caltech campus car park | development (before March 2020) and main test (after the 2020 closure) | April 2018 to September 2021 |
| Jet Propulsion Laboratory (JPL) | replication, analysed once after the settings are frozen | September 2018 to September 2021 |

The third site in the API (an office car park, `office001`) has too few drivers to carry a
claim and is not used.

## Download

`staypromise fetch <site>` asks the API for one calendar month at a time: every session whose
plug-in time falls in that month in UTC, start included, end excluded, 100 sessions a page. A
month is accepted only if the number of sessions received equals the total the server
reports and no session appears twice; otherwise it is asked for again. Each month is stored
as canonical JSON (sessions sorted by their database ID, keys sorted, no whitespace),
gzip-compressed, under `data/raw/<site>/`.

The SHA-256 of each month's canonical JSON is committed in `fingerprints/<site>.csv`, with
the session count. `staypromise verify <site>` recomputes them, so anyone with a token can
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

Nothing is dropped for being unusual: very short and very long stays stay in, because a
promise rule meets them in practice.

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

They describe how the data were recorded; none is used to drop sessions.

## Check against an independent public copy

SustainGym (Yeh et al., NeurIPS 2023 Datasets and Benchmarks Track) publishes four extracts
of the Caltech site under CC-BY-4.0, made from the same API by other people. The files used
here are pinned to commit `eac2a4d5ce4ccf44b13c290edac72532c68e30f5` of
[chrisyeh96/sustaingym](https://github.com/chrisyeh96/sustaingym) and checked by SHA-256
before use (`snapshots.py`). `staypromise snapshot-check <folder>` compares each extract with
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