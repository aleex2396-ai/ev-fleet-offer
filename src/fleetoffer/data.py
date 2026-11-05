"""The session table: stored ACN-Data months turned into one row per charging session.

The table's rules live in `validate_sessions`, which every other module reads sessions
through:

- ``session_id``: unique text identifier.
- ``connect_time``, ``disconnect_time``: timezone-aware timestamps, with the car leaving
  strictly after it plugs in.
- ``user_id`` (optional): the driver's identifier, or missing.

Times are handled internally as float hours since 1970-01-01 UTC.

Cleaning drops only records that cannot be a charging session; everything else is kept and,
where useful, flagged, so no analysis choice is hidden in the cleaning:

- ``no_times``: the session ID, plug-in time or unplug time is missing.
- ``not_positive``: the car unplugs at or before it plugs in.
- ``duplicate``: a session ID already seen (the first is kept).

What drivers typed in the charging app is kept in a separate table, one row per entry, each
with the time it was entered. It is never used as the true departure. `session_flags` adds
descriptive flags about it (claimed by a known driver, typed entry recorded at the same second
as the plug-in, typed stay of exactly 60 or 480 minutes); they describe how the data were
recorded and are not used to drop anything.

Per-session time series give the charging current at about four-second intervals.
`parse_series` checks a record against the usability rule of check D1 (analysis plan,
Section 10a) and `energy_kwh` integrates the current: each sample's current, negative values
counted as zero, holds until the next sample, and nothing is counted after the last one.
Power is current times the line voltage, `VOLTS` (208 V at Caltech, where a 32 A charger is
the plan's 6.66 kW).

Two parts of the data are sealed, so the public history shows they were not read before the
analysis was fixed:

- The replication site (JPL): `load_sessions` refuses it unless ``unseal=True`` is passed,
  which only the replication run does, after the analysis settings are frozen.
- Caltech's main test period, sessions plugged in from 1 March 2020 (Pacific time): by
  default the loaders return only the sessions before it. ``unseal=True`` opens it, and only
  once the analysis plan's git tag (`PLAN_TAG`) exists in the repository.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer.acn import months_between, raw_path, read_month, read_series, series_path

SEALED_SITES = frozenset({"jpl"})
TEST_PERIODS = {"caltech": pd.Timestamp("2020-03-01", tz="America/Los_Angeles")}
PLAN_TAG = "analysis-plan"
REPO = Path(__file__).resolve().parents[2]
_TIME = "%a, %d %b %Y %H:%M:%S GMT"


REQUIRED_COLUMNS = ("session_id", "connect_time", "disconnect_time")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_HOUR = pd.Timedelta(hours=1)


def validate_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return a checked copy of the table, in UTC, sorted by plug-in time."""
    missing = [c for c in REQUIRED_COLUMNS if c not in sessions.columns]
    if missing:
        raise ValueError(f"session table is missing columns: {missing}")

    out = sessions.copy()
    for col in ("connect_time", "disconnect_time"):
        if not isinstance(out[col].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{col} must be timezone-aware")
        if out[col].isna().any():
            raise ValueError(f"{col} has missing values")
        out[col] = out[col].dt.tz_convert("UTC")

    if not (out["disconnect_time"] > out["connect_time"]).all():
        raise ValueError("every session must end strictly after it starts")

    out["session_id"] = out["session_id"].astype(str)
    if out["session_id"].duplicated().any():
        raise ValueError("session_id values must be unique")

    if "user_id" not in out.columns:
        out["user_id"] = None
    out["user_id"] = out["user_id"].astype(object).where(out["user_id"].notna(), None)

    out = out.sort_values(["connect_time", "session_id"], kind="stable")
    return out.reset_index(drop=True)


def to_hours(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Timezone-aware timestamps to float hours since 1970-01-01 UTC."""
    return np.asarray((times - _EPOCH) / _HOUR, dtype=float)


def timestamp_to_hours(ts: pd.Timestamp | str) -> float:
    """One timezone-aware timestamp to float hours since 1970-01-01 UTC."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {ts}")
    return float((ts - _EPOCH) / _HOUR)


def from_hours(hours: np.ndarray | list[float]) -> pd.DatetimeIndex:
    """Float hours since 1970-01-01 UTC back to UTC timestamps."""
    return pd.to_datetime(np.asarray(hours, dtype=float) * 3600.0, unit="s", utc=True)


class SealedSiteError(PermissionError):
    """The replication site was requested before it was unsealed."""


class SealedPeriodError(PermissionError):
    """A main test period was requested before the analysis plan was tagged."""


def plan_is_tagged() -> bool:
    """True if the analysis plan's tag exists in the git repository holding this code."""
    try:
        found = subprocess.run(
            ["git", "-C", str(REPO), "tag", "--list", PLAN_TAG],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return found.stdout.strip() == PLAN_TAG


def _plugged_in_before(item: dict, start: pd.Timestamp) -> bool:
    try:
        when = datetime.strptime(item.get("connectionTime") or "", _TIME).replace(tzinfo=UTC)
    except ValueError:
        return False
    return when < start


def _times(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=_TIME, utc=True, errors="coerce")


def _column(raw: pd.DataFrame, name: str) -> pd.Series:
    return raw[name] if name in raw.columns else pd.Series(None, index=raw.index, dtype=object)


def _text_or_none(values: pd.Series) -> pd.Series:
    """Text values with missing ones as None (an object column, so None stays None)."""
    text = [None if v is None or v != v else str(v) for v in values]
    return pd.Series(text, index=values.index, dtype=object)


def parse_sessions(items: list[dict]) -> pd.DataFrame:
    """API records to a table, before cleaning. Times are UTC."""
    raw = pd.DataFrame.from_records(items)
    return pd.DataFrame(
        {
            "session_id": _text_or_none(_column(raw, "sessionID")),
            "connect_time": _times(_column(raw, "connectionTime")),
            "disconnect_time": _times(_column(raw, "disconnectTime")),
            "done_charging_time": _times(_column(raw, "doneChargingTime")),
            "kwh_delivered": pd.to_numeric(_column(raw, "kWhDelivered"), errors="coerce"),
            "user_id": _text_or_none(_column(raw, "userID")),
            "station_id": _text_or_none(_column(raw, "stationID")),
            "space_id": _text_or_none(_column(raw, "spaceID")),
        }
    )


def parse_user_inputs(items: list[dict]) -> pd.DataFrame:
    """One row per entry a driver typed in the app, with the time it was entered."""
    rows = []
    for item in items:
        entries = item.get("userInputs") or []
        for order, entry in enumerate(entries):
            rows.append(
                {
                    "session_id": str(item.get("sessionID")),
                    "entry": order,
                    "entries": len(entries),
                    "modified_at": entry.get("modifiedAt"),
                    "minutes_available": entry.get("minutesAvailable"),
                    "requested_departure": entry.get("requestedDeparture"),
                    "kwh_requested": entry.get("kWhRequested"),
                    "miles_requested": entry.get("milesRequested"),
                }
            )
    columns = [
        "session_id",
        "entry",
        "entries",
        "modified_at",
        "minutes_available",
        "requested_departure",
        "kwh_requested",
        "miles_requested",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out["modified_at"] = _times(out["modified_at"])
    out["requested_departure"] = _times(out["requested_departure"])
    for col in ("minutes_available", "kwh_requested", "miles_requested"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_sessions(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the three dropping rules in order; return the kept rows and the count per rule."""
    counts = {"records": len(table)}
    no_times = (
        table["session_id"].isna() | table["connect_time"].isna() | table["disconnect_time"].isna()
    )
    counts["no_times"] = int(no_times.sum())
    kept = table[~no_times]
    not_positive = kept["disconnect_time"] <= kept["connect_time"]
    counts["not_positive"] = int(not_positive.sum())
    kept = kept[~not_positive]
    duplicate = kept["session_id"].duplicated(keep="first")
    counts["duplicate"] = int(duplicate.sum())
    kept = kept[~duplicate]
    counts["kept"] = len(kept)
    kept = kept.sort_values(["connect_time", "session_id"], kind="stable")
    return kept.reset_index(drop=True), counts


WITH_PLUGIN_S = 1.0  # an entry this close to the plug-in time was recorded with it


def session_flags(sessions: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    """Per-session descriptive flags, aligned with `sessions`.

    ``claimed``: the session has a driver ID. ``typed``: the driver entered at least one plan.
    ``input_lag_s``: seconds from plug-in to the first entry. ``input_with_plugin``: that entry
    was recorded within one second of the plug-in. ``typed_minutes``: the stay the first entry
    stated. ``typed_60`` / ``typed_480``: it was exactly one hour / eight hours. ``edits``:
    entries after the first.
    """
    first = (
        inputs.sort_values(["session_id", "entry"])
        .drop_duplicates("session_id", keep="first")
        .set_index("session_id")
    )
    joined = sessions[["session_id", "connect_time", "user_id"]].join(first, on="session_id")
    lag = (joined["modified_at"] - joined["connect_time"]) / pd.Timedelta(seconds=1)
    typed = joined["entry"].notna()
    minutes = joined["minutes_available"]
    return pd.DataFrame(
        {
            "session_id": sessions["session_id"],
            "claimed": sessions["user_id"].notna().to_numpy(),
            "typed": typed.to_numpy(),
            "input_lag_s": lag.to_numpy(),
            "input_with_plugin": (lag.abs() <= WITH_PLUGIN_S).to_numpy(),
            "typed_minutes": minutes.to_numpy(),
            "typed_60": (minutes == 60).to_numpy(),
            "typed_480": (minutes == 480).to_numpy(),
            "edits": (joined["entries"].fillna(1) - 1).astype(int).where(typed, 0).to_numpy(),
        },
        index=sessions.index,
    )


def stored_months(data_root: Path, site: str) -> list[str]:
    folder = Path(data_root) / "raw" / site
    return sorted(p.name.removesuffix(".json.gz") for p in folder.glob("*.json.gz"))


def load_items(data_root: Path, site: str, *, unseal: bool = False) -> list[dict]:
    if site in SEALED_SITES and not unseal:
        raise SealedSiteError(
            f"{site} is the sealed replication site; it is analysed once, after the settings "
            "are frozen"
        )
    start = TEST_PERIODS.get(site)
    if start is not None and unseal and not plan_is_tagged():
        raise SealedPeriodError(
            f"{site} sessions from {start:%d %B %Y} are the main test period; they open only "
            f"once the '{PLAN_TAG}' tag exists in git"
        )
    sealed = start is not None and not unseal
    items: list[dict] = []
    for month in stored_months(data_root, site):
        first = pd.Timestamp(f"{month}-01", tz="UTC")
        if sealed and first >= start:
            continue
        month_items = read_month(data_root, site, month)
        if sealed and first + pd.DateOffset(months=1) > start:
            month_items = [item for item in month_items if _plugged_in_before(item, start)]
        items.extend(month_items)
    return items


def load_sessions(
    data_root: Path, site: str, *, unseal: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """The cleaned session table of one site, and the cleaning counts."""
    return clean_sessions(parse_sessions(load_items(data_root, site, unseal=unseal)))


def load_user_inputs(data_root: Path, site: str, *, unseal: bool = False) -> pd.DataFrame:
    return parse_user_inputs(load_items(data_root, site, unseal=unseal))


def stay_hours(table: pd.DataFrame) -> np.ndarray:
    return ((table["disconnect_time"] - table["connect_time"]) / pd.Timedelta(hours=1)).to_numpy()


def missing_months(data_root: Path, site: str, first: str, last: str) -> list[str]:
    return [m for m in months_between(first, last) if not raw_path(data_root, site, m).exists()]


# Per-session time series.

VOLTS = 208.0
SERIES_REASONS = (
    "usable",
    "no_series",
    "too_short",
    "length_mismatch",
    "bad_time",
    "not_increasing",
    "bad_current",
)


@dataclass(frozen=True)
class Series:
    """One session's charging current: sample times (hours since 1970 UTC) and amperes."""

    session_id: str
    times_h: np.ndarray
    amps: np.ndarray


def parse_series(record: dict) -> tuple[Series | None, str]:
    """The session's current series and ``"usable"``, or None and the reason it is not.

    Usable means: a ``chargingCurrent`` with as many timestamps as values, at least two of
    each, every timestamp readable and strictly later than the one before, and every value a
    finite number. The rule is the one fixed for check D1 before any series was examined.
    """
    session_id = str(record.get("sessionID"))
    current = record.get("chargingCurrent")
    if not isinstance(current, dict):
        return None, "no_series"
    stamps, values = current.get("timestamps") or [], current.get("current") or []
    if len(stamps) != len(values):
        return None, "length_mismatch"
    if len(stamps) < 2:
        return None, "too_short"
    times = pd.to_datetime(pd.Series(stamps, dtype=object), format=_TIME, utc=True, errors="coerce")
    if times.isna().any():
        return None, "bad_time"
    times_h = to_hours(times)
    if not (np.diff(times_h) > 0).all():
        return None, "not_increasing"
    try:
        amps = np.array([float(v) for v in values], dtype=float)
    except (TypeError, ValueError):
        return None, "bad_current"
    if not np.isfinite(amps).all():
        return None, "bad_current"
    return Series(session_id, times_h, amps), "usable"


def energy_kwh(
    series: Series, start_h: float = -math.inf, end_h: float = math.inf, volts: float = VOLTS
) -> float:
    """Energy delivered between `start_h` and `end_h`, holding each sample until the next."""
    t = series.times_h
    lo = np.clip(t[:-1], start_h, end_h)
    hi = np.clip(t[1:], start_h, end_h)
    amps = np.maximum(series.amps[:-1], 0.0)
    return float(np.sum(amps * (hi - lo)) * volts / 1000.0)


def load_series(
    data_root: Path, site: str, name: str | None = None, *, unseal: bool = False
) -> list[dict]:
    """Stored time-series records: every stored month, or the one named file.

    The seals of `load_items` apply: the replication site is refused, and Caltech records
    from its main test period are dropped unless it is unsealed after the plan's tag.
    """
    if site in SEALED_SITES and not unseal:
        raise SealedSiteError(f"{site} is the sealed replication site")
    start = TEST_PERIODS.get(site)
    if start is not None and unseal and not plan_is_tagged():
        raise SealedPeriodError(f"{site} series from {start:%d %B %Y} open only after the tag")
    if name is not None:
        items = read_series(data_root, site, name)
    else:
        folder = series_path(data_root, site, "x").parent
        months = sorted(p.name.removesuffix(".json.gz") for p in folder.glob("????-??.json.gz"))
        items = [item for month in months for item in read_series(data_root, site, month)]
    if start is not None and not unseal:
        items = [item for item in items if _plugged_in_before(item, start)]
    return items
