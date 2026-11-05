# Earlier looks at the data

Exploratory analyses of ACN-Data were run while this project was being scoped, in October
and November 2026, before the analysis plan was written. They are listed here because an
outcome that has been seen can no longer be tested blind. Everything below was read by the
person running this analysis. None of it should be treated as unseen.

"Outcome statistics" means anything computed from departure times: stay lengths, how far
departures fell from what drivers typed, and how often a lower bound on the stay held.
"Counts" means session counts, field names and similar metadata.

## Caltech, before March 2020 (development data)

Outcome statistics were computed for whole months (October and November 2018; May, June and
July 2019; February 2020), for every day from May to August 2019, for a random sample of
days across the whole period, and repeatedly on the public SustainGym extracts of May to
December 2019. What was learned:

- median stays of about 5 to 6.5 hours; drivers left about 0.5 to 1 hour later than they
  had typed; typed stays of exactly 8 hours were almost absent (11 before March 2020);
- one-sided 90% lower bounds, fixed or rolling, held in about 86% to 92% of sessions;
- month-to-month changes in the stay distribution (1-Wasserstein distance) of 0.18 to 0.45
  hours, against a sampling-noise level of 0.21 to 0.35 hours;
- the day-to-day dispersion of early departures, close to what independent drivers would
  give.

## Caltech, March 2020 onward (main test)

Outcome statistics were computed for every day from March 2020 to September 2021, for a
random sample of days, and repeatedly on the SustainGym extracts of February to May 2020 and
May to August 2021. What was learned:

- the closure itself (24 sessions in April 2020, none in September and October 2020), median
  stays of about 1 to 2 hours afterwards, and departures 4 to 6 hours earlier than typed;
- the median stay in May to August 2021 against May to August 2019 (1.68 against 6.05 hours),
  and the number of sessions per day in 2021, about half the 2019 level;
- 26% to 36% of typed stays were exactly 8 hours in 2021, and typed entries were recorded at
  the plug-in rather than after it;
- a lower bound fitted before the closure and never updated held in 37% to 87% of sessions
  afterwards, depending on how it was fitted and the month; rolling-window bounds recovered
  to about 85% to 95%;
- distances between months' stay distributions, and the drift radius and sample size that
  the guarantee of Yan, Parise and Bitar (2022) for chance constraints under drift would need
  at these distances;
- the day-to-day dispersion of early departures in summer 2021;
- on the public extracts of May to August 2019 and 2021: the distance between the two
  summers' stay distributions; a lower bound fitted in summer 2019 (kept in 61% to 62% of
  2021 sessions); rolling windows of 100, 300 and 1,000 sessions; a fleet-level promise (cars
  plugged in at 2 pm on weekdays; one fitted in summer 2019 held on no 2021 weekday); and
  the promise sizes of the Yan, Parise and Bitar rule with fixed drift values (0.52 to 0.75
  hours at zero drift). These were run in November 2026, while this project was being scoped.

## Jet Propulsion Laboratory (replication site)

Outcome statistics were computed on a random sample of days from November 2018 to September
2021, for every day from May to August 2021, and (typed entries only) for June 2021. What was
learned:

- median stays of about 8.5 hours before the closure and 5.3 to 6 hours after; departures
  moved from about 0.5 hours later than typed to about 0.4 hours earlier;
- 25% to 38% of typed stays were exactly 8 hours from mid-2020;
- a lower bound fitted at the end of 2019 and never updated held in 74% to 96% of sessions
  in the months after March 2020; rolling-window bounds held in 71% to 100%;
- month-to-month distances between stay distributions in summer 2021, and the dispersion of
  early departures before the closure.

## The office car park (`office001`)

Counts only, and one month checked for driver IDs (none). It has no sessions before March
2019.

## Counts whose period was not recorded

An earlier summary gave the share of Caltech sessions linked to a registered driver (47%) and
the share with a typed plan recorded within ten minutes of the plug-in (46%). Its period and
exact definitions were not recorded. Both are recomputed in this repository with stated
definitions (analysis plan, Section 10a), and the earlier figures are not used.

## Not looked at

The looks above concern stays, typed plans and session counts. None is recorded as computing
a statistic of the energy cars took (the reading of 192 raw sessions below showed each one's
recorded kWh), and the per-session time series (charging current and pilot signal) had not
been downloaded for any period. At neither site has any of this
project's quantities been computed: no learned per-car bound, no per-car flexibility set, no
offer from the inner approximation, no delivery check, no day-level calibration rule and no
comparison of learned against population bounds. No rule with an online correction from its
own record of held and failed bounds (adaptive conformal inference) has been run at either
site, nor the drift-aware rule of Yan, Parise and Bitar with its drift allowance estimated
from the data as it arrives (it was run at Caltech only with fixed drift values, as listed
above).

## Looks made as part of this repository

- Downloads of both sites with session counts and fingerprints (`docs/data.md`).
- A comparison of the Caltech download with the SustainGym extracts (`docs/data.md`).
- A reading of 192 raw Caltech sessions picked at random (fixed seed), from before March
  2020 and from 2021, including three drivers followed over 20 visits in each period: for each
  session its plug-in and unplug times, stay, recorded kWh, driver link and typed plan, read
  without computing statistics.
- Check D1 (analysis plan, Section 10a): the time series of 300 development sessions drawn by
  a fixed seed, 50 of them from the frozen evaluation months (September 2019 to February
  2020), and the energy statistics of `docs/data.md`, including, after the check failed,
  energy between 10:00 and 20:00 for the 95 sampled sessions present at 10:00 on a weekday.
  No stay statistic and no bound was computed. Nothing from March 2020 onward was requested.
