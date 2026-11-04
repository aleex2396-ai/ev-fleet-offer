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
One estimates how far the distribution of stays has drifted and makes the commitment robust
to that much drift (distributionally robust optimisation under drift). The other estimates
nothing about the drift: it corrects the commitment online from its own record of kept and
broken commitments (adaptive conformal inference). They offer different kinds of guarantee:
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
  from before the 2020 change. The Jet Propulsion Laboratory site in the same data set will
  be held back as a replication site and analysed once, after the settings are frozen.
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
educational and research use after registration. The data are not redistributed here; the
download script will use the reader's own access token. If you use the data, cite:

> Z. J. Lee, T. Li and S. H. Low. ACN-Data: Analysis and Applications of an Open EV
> Charging Dataset. *Proceedings of the Tenth International Conference on Future Energy
> Systems (e-Energy '19)*, 2019.

## Layout

```
src/staypromise/
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
tests/           unit tests, including the look-ahead tests
docs/            analysis plan and discovery log
```

## Licence

Code: MIT (see `LICENSE`). Data: see above.
