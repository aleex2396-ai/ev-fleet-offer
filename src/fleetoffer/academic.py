"""Caltech's academic calendar and Institute holidays, 2017-18 to 2021-22, as day flags.

The flags are inputs at the decision time (analysis plan, Section 4): on a given local date,
whether instruction is in session (`in_term`, from the beginning of instruction to the end of
the term, study periods and final examinations included), whether it is the summer between
the end of the third term and the next beginning of instruction (`summer`), and whether it is
an Institute holiday (`holiday`). Days that are neither in term nor summer are the winter and
spring recesses.

Sources, read on 5 November 2026:

- Terms, and the holidays they name (Thanksgiving, Martin Luther King Day, Presidents' Day,
  Memorial Day): the Office of the Registrar's academic calendars,
  https://registrar.caltech.edu/documents/7450/AcademicCalendar2017-18.pdf,
  .../7449/academiccalendar2018-19.pdf,
  .../17372/AcademicCalendar2019-20_Updated_5-15-20_77pYNuK.pdf,
  .../7212/AcademicCalendar2020-21.pdf and .../7213/AcademicCalendar2021-22.pdf.
- Independence Day, Labor Day and the year-end closure, which the registrar's calendars do
  not list: Caltech's events calendar (https://www.caltech.edu/campus-life-events/calendar),
  which lists Independence Day on 4-5 July 2019, 3 July 2020 and 5 July 2021, Labor Day on
  2 September 2019 and 7 September 2020, and "Institute holidays" from 24 December to
  1 January for 2018-19 and 2020-21. For the years it was not found for (Independence Day
  2018, Labor Day 2018 and 2021, the closure of 2019-20 and 2021-22) the same rule is
  applied: the federal date, and 24 December to 1 January.
"""

from __future__ import annotations

import pandas as pd

# (beginning of instruction, end of term), local dates, inclusive.
TERMS = (
    ("2018-04-02", "2018-06-15"),
    ("2018-10-01", "2018-12-14"),
    ("2019-01-07", "2019-03-20"),
    ("2019-04-01", "2019-06-14"),
    ("2019-10-01", "2019-12-13"),
    ("2020-01-06", "2020-03-18"),
    ("2020-03-30", "2020-06-12"),
    ("2020-09-29", "2020-12-11"),
    ("2021-01-04", "2021-03-17"),
    ("2021-03-29", "2021-06-11"),
    ("2021-09-27", "2021-12-10"),
    ("2022-01-03", "2022-03-16"),
    ("2022-03-28", "2022-06-10"),
)
# Third-term ends to the next beginning of instruction, exclusive of both.
SUMMERS = (
    ("2018-06-15", "2018-10-01"),
    ("2019-06-14", "2019-10-01"),
    ("2020-06-12", "2020-09-29"),
    ("2021-06-11", "2021-09-27"),
)
HOLIDAYS = (
    "2018-05-28",
    "2018-07-04",
    "2018-09-03",
    "2018-11-22",
    "2018-11-23",
    "2019-01-21",
    "2019-02-18",
    "2019-05-27",
    "2019-07-04",
    "2019-07-05",
    "2019-09-02",
    "2019-11-28",
    "2019-11-29",
    "2020-01-20",
    "2020-02-17",
    "2020-05-25",
    "2020-07-03",
    "2020-09-07",
    "2020-11-26",
    "2020-11-27",
    "2021-01-18",
    "2021-02-15",
    "2021-05-31",
    "2021-07-05",
    "2021-09-06",
    "2021-11-25",
    "2021-11-26",
    "2022-01-17",
    "2022-02-21",
    "2022-05-30",
)
CLOSURE_YEARS = (2017, 2018, 2019, 2020, 2021)  # 24 December of the year to 1 January after
FIRST, LAST = pd.Timestamp("2018-01-01"), pd.Timestamp("2022-06-30")


def _days(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(first, last, freq="D")


def holidays() -> pd.DatetimeIndex:
    closures = [_days(f"{year}-12-24", f"{year + 1}-01-01") for year in CLOSURE_YEARS]
    named = pd.DatetimeIndex(pd.to_datetime(list(HOLIDAYS)))
    return named.append(closures).unique().sort_values()


def day_flags(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """``in_term``, ``summer`` and ``holiday`` for local dates (time of day is ignored).

    Dates outside the calendar's span (2018 to June 2022) are refused, so a site or period
    the calendar does not cover cannot pass silently as "no term, no holiday".
    """
    days = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    if len(days) and (days.min() < FIRST or days.max() > LAST):
        raise ValueError(f"the academic calendar covers {FIRST:%Y-%m-%d} to {LAST:%Y-%m-%d}")
    in_term = pd.Series(False, index=range(len(days)))
    summer = pd.Series(False, index=range(len(days)))
    for first, last in TERMS:
        in_term |= (days >= first) & (days <= last)
    for after, before in SUMMERS:
        summer |= (days > after) & (days < before)
    return pd.DataFrame(
        {
            "in_term": in_term.to_numpy(),
            "summer": summer.to_numpy(),
            "holiday": days.isin(holidays()),
        }
    )
