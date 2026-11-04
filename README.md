# ev-stay-promise

**Does estimating drift pay? Three ways to keep a dwell-time commitment reliable across an
abrupt change in driver behaviour.**

An aggregator that schedules electric-vehicle charging can only offer the flexibility it
can count on. The simplest commitment about one plugged-in car is a one-sided lower bound
on its remaining dwell time: "this car will stay at least *X* more hours", required to hold
with a stated probability, for example 90%. A rule that sets *X* from past sessions has two
scores, always reported together: how often its commitments hold (coverage) and how large
they are (hours committed).

When behaviour changes, there are two principled ways to keep such a commitment reliable.
One estimates a budget for how far the distribution of stays can drift as data age, and
treats each past stay as uncertain by an amount that grows with that budget before setting
the commitment (a robust sample approximation of a chance constraint under drift). The
other estimates nothing about the drift: it corrects the commitment online from its own
record of kept and broken commitments (adaptive conformal inference). They offer different kinds of guarantee:
the first holds with high probability under assumptions about the drift; the second holds
on average over time with no such assumptions. The simplest alternative to both is a rolling
window of recent sessions.

This project compares the three, with a never-updated rule as a reference, on the public
Caltech ACN-Data charging sessions across the 2020 campus closure. For each rule it
measures how many hours it commits while meeting the reliability target, and how quickly
its reliability recovers after the change. Whether the drift-aware rule earns anything over
the drift-free one depends on whether drift can be estimated from data of this size, so the
project also measures how much of an estimated drift is sampling noise.

## Status

In development. No results from real data have been produced yet. The analysis plan
(hypotheses, measures and pass/fail criteria) will be committed to `docs/` before any
confirmatory run, together with a list of the exploratory looks at the data that preceded
it.

## Design

- **No look-ahead by construction.** The replay engine (`engine.py`) is the only code that
  reads true departure times. A rule sees earlier sessions only as they were known at the
  decision time (a car still plugged in appears with its stay so far, not its final stay),
  and learns the outcome of each of its promises only at the moment that outcome happens.
- **Tested for look-ahead.** `check_no_lookahead` replays a rule twice at several cutoff
  times: once on the real data, once with everything after the cutoff scrambled. Every
  promise made before the cutoff must be identical. The test suite runs it on every rule,
  and on two deliberately cheating rules that it must catch, one of which reads the future
  directly and one of which has a setting fitted on the whole data set.
- **Cars still plugged in are used, not dropped.** Stay-length quantiles use the
  Kaplan-Meier estimator, so recent sessions do not bias promises towards short stays.
- **Reliability and size together.** Every summary reports the share of promises kept, the
  mean hours promised, and the share of cars that received no promise.
- **Settings fixed before confirmation** (from the data stage on). Tuning will use only data
  from before the 2020 change, and every setting comes from a rule written down before any
  run. Caltech sessions from 1 March 2020, the main test period, are sealed in the loader:
  they open only once the analysis plan's git tag (`analysis-plan`) exists. The Jet
  Propulsion Laboratory site in the same data set is held back as a replication site: the
  loader refuses it unless explicitly unsealed, and it will be analysed once, after the
  settings are frozen. A test checks that no library code unseals either.
- **Every output records how it was made.** Each run writes a manifest with the code
  commit, parameters, package versions and a fingerprint of its input.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run staypromise demo
```

The demo runs the whole pipeline on made-up sessions and writes tables, a figure and a
manifest to `results/demo/`. It checks that everything runs; it is not a result.

## Data

Session data come from [Caltech ACN-Data](https://ev.caltech.edu/dataset), available for
educational and research use after registration. The data are not redistributed here. With
your own access token in `ACN_API_TOKEN` (or in `~/.config/acndata/api_key`):

```bash
uv run staypromise fetch caltech    # about 30 minutes; stored under data/, never committed
uv run staypromise verify caltech   # same data as this repository's fingerprints/caltech.csv?
```

`docs/data.md` describes what is downloaded, the cleaning rules and their counts, the sealed
test period, and a check of the download against an independent public copy.
`docs/theory.md` states the guarantees behind the rules with their theorem numbers; its
table of required sample sizes is printed by `uv run staypromise requirements`. If you use the data, cite:

> Z. J. Lee, T. Li and S. H. Low. ACN-Data: Analysis and Applications of an Open EV
> Charging Dataset. *Proceedings of the Tenth International Conference on Future Energy
> Systems (e-Energy '19)*, 2019.

## Layout

```
src/staypromise/
  acn.py         ACN-Data download, month by month, with fingerprints
  data.py        stored months to the session table; cleaning rules and flags
  snapshots.py   comparison with an independent public copy of the Caltech data
  sessions.py    the session table and its checks
  interface.py   what a rule may see: Request, History, Outcome
  engine.py      event-by-event replay; the only reader of true departure times
  survival.py    Kaplan-Meier stay-length quantiles
  rules.py       promise rules
  checks.py      the look-ahead check
  scoring.py     coverage and promise size
  synthetic.py   made-up sessions with known behaviour
  figures.py     figures, each drawn from a table that is also written to disk
  pipeline.py    end-to-end runs with manifests
tests/           unit tests, including the look-ahead tests (no network needed)
fingerprints/    SHA-256 of every downloaded month, so others can confirm the same data
docs/            data notes, theory requirements, analysis plan and discovery log
```

## Licence

Code: MIT (see `LICENSE`). Data: see above.
