"""What is known about each present car at the decision time: the inputs of the per-car models.

`at_decision` turns a `DayView` (the engine's account of what is known at 10:00) into one row
per present car. It reads nothing else, so the inputs inherit the engine's guarantee: a car's
driver is known only if the link was recorded before 10:00, a typed plan only if it was
entered before 10:00, and a driver's history only from sessions that had ended before 10:00
(analysis plan, Sections 4 and 7). Energy so far is not an input: its estimate uses fields
recorded only at unplug.

The inputs (analysis plan, Section 4):

- the session: plug-in time (hours from the decision day's local midnight, negative for a
  car plugged in on an earlier day), time plugged in so far, weekday, month, the academic
  calendar's flags (`academic.day_flags`), station, the number of cars present, and the price
  regime: from 1 November 2018 charging was paid for sessions linked to a driver and free for
  the others (Lee, Li and Low, 2019);
- the driver, if linked before 10:00: the number of their earlier sessions that had ended by
  then, the 10th and 50th percentiles of those sessions' stays (hours) and energies (kWh),
  and days since their last plug-in;
- the typed plan, the latest app entry recorded before 10:00: hours from 10:00 to the typed
  departure, the typed stay and energy, and flags for typed stays of exactly 60 and exactly
  480 minutes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fleetoffer.academic import day_flags
from fleetoffer.interface import DayView

LOCAL_TZ = "America/Los_Angeles"
PAID_FROM = pd.Timestamp("2018-11-01", tz=LOCAL_TZ)
NUMERIC = (
    "plugin_hour",
    "elapsed_h",
    "weekday",
    "month",
    "in_term",
    "summer",
    "holiday",
    "cars_present",
    "paid_regime",
    "linked",
    "paid",
    "n_past",
    "stay_q10",
    "stay_q50",
    "kwh_q10",
    "kwh_q50",
    "days_since_last",
    "typed",
    "typed_remaining_h",
    "typed_minutes",
    "typed_kwh",
    "typed_60",
    "typed_480",
)
KEYS = ("decision_time", "session_id", "connect_time")
_HOUR = pd.Timedelta(hours=1)


def at_decision(
    day: DayView, tz: str = LOCAL_TZ, calendar: Callable[..., pd.DataFrame] = day_flags
) -> pd.DataFrame:
    """One row per car in ``day.present``, in the same order: the keys, the numeric inputs
    (`NUMERIC`, missing where they do not apply) and ``station``. `calendar` gives the day
    flags; tests on made-up dates pass their own."""
    cars = day.present
    t = day.decision_time
    local = t.tz_convert(tz)
    midnight = local.normalize()
    out = pd.DataFrame(
        {
            "decision_time": t,
            "session_id": cars["session_id"].to_numpy(),
            "connect_time": cars["connect_time"].to_numpy(),
        }
    )
    out["connect_time"] = pd.to_datetime(out["connect_time"], utc=True)
    out["plugin_hour"] = ((out["connect_time"].dt.tz_convert(tz) - midnight) / _HOUR).to_numpy()
    out["elapsed_h"] = cars["elapsed_h"].to_numpy(float)
    out["weekday"] = float(local.day_of_week)
    out["month"] = float(local.month)
    flags = calendar(pd.DatetimeIndex([local]))
    for name in ("in_term", "summer", "holiday"):
        out[name] = float(flags[name].iloc[0])
    out["cars_present"] = float(len(cars))
    out["paid_regime"] = float(local >= PAID_FROM)
    users = np.array([u if isinstance(u, str) else None for u in cars["user_id"]], dtype=object)
    linked = np.array([u is not None for u in users], dtype=bool)
    out["linked"] = linked.astype(float)
    out["paid"] = out["linked"] * out["paid_regime"]
    out = out.join(_driver_history(day, users, t))
    out = out.join(_typed_plan(day, t))
    station = (
        cars["station_id"] if "station_id" in cars.columns else pd.Series(None, index=cars.index)
    )
    out["station"] = station.to_numpy(object)
    return out


def _driver_history(day: DayView, users: np.ndarray, t: pd.Timestamp) -> pd.DataFrame:
    cols = ["n_past", "stay_q10", "stay_q50", "kwh_q10", "kwh_q50", "days_since_last"]
    out = pd.DataFrame(np.nan, index=range(len(users)), columns=cols)
    linked = [i for i, u in enumerate(users) if u is not None]
    out.loc[linked, "n_past"] = 0.0
    ended = day.ended
    if not linked or len(ended) == 0 or "user_id" not in ended.columns:
        return out
    wanted = {users[i] for i in linked}
    past = ended[ended["user_id"].isin(wanted)]
    if len(past) == 0:
        return out
    stay = (past["disconnect_time"] - past["connect_time"]) / _HOUR
    kwh = past["kwh_delivered"] if "kwh_delivered" in past.columns else stay * np.nan
    frame = pd.DataFrame(
        {"user": past["user_id"].to_numpy(), "stay": stay.to_numpy(), "kwh": kwh.to_numpy(float)}
    )
    frame["since"] = ((t - past["connect_time"]) / pd.Timedelta(days=1)).to_numpy()
    groups = frame.groupby("user")
    stats = pd.DataFrame(
        {
            "n_past": groups.size().astype(float),
            "stay_q10": groups["stay"].quantile(0.1),
            "stay_q50": groups["stay"].quantile(0.5),
            "kwh_q10": groups["kwh"].quantile(0.1),
            "kwh_q50": groups["kwh"].quantile(0.5),
            "days_since_last": groups["since"].min(),
        }
    )
    for i in linked:
        if users[i] in stats.index:
            out.loc[i, cols] = stats.loc[users[i], cols].to_numpy(float)
    return out


def _typed_plan(day: DayView, t: pd.Timestamp) -> pd.DataFrame:
    cars = day.present
    cols = ["typed", "typed_remaining_h", "typed_minutes", "typed_kwh", "typed_60", "typed_480"]
    out = pd.DataFrame(np.nan, index=range(len(cars)), columns=cols)
    out["typed"] = 0.0
    entries = day.entries
    if len(entries) == 0:
        return out
    mine = entries[entries["session_id"].isin(set(cars["session_id"]))]
    if len(mine) == 0:
        return out
    latest = (
        mine.sort_values(["modified_at", "entry"], kind="stable")
        .drop_duplicates("session_id", keep="last")
        .set_index("session_id")
    )
    plan = latest.reindex(cars["session_id"].to_numpy())
    has = plan["modified_at"].notna().to_numpy()
    minutes = plan["minutes_available"].to_numpy(float)
    out["typed"] = has.astype(float)
    out["typed_remaining_h"] = ((plan["requested_departure"] - t) / _HOUR).to_numpy(float)
    out["typed_minutes"] = minutes
    out["typed_kwh"] = plan["kwh_requested"].to_numpy(float)
    out["typed_60"] = np.where(has, minutes == 60, np.nan)
    out["typed_480"] = np.where(has, minutes == 480, np.nan)
    return out
