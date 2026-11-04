"""Turn stored ACN-Data months into the session table, with every cleaning rule counted.

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

The replication site is sealed: `load_sessions` refuses it unless ``unseal=True`` is passed,
which only the replication run does, after the analysis settings are frozen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from staypromise.acn import months_between, raw_path, read_month

SEALED_SITES = frozenset({"jpl"})
_TIME = "%a, %d %b %Y %H:%M:%S GMT"


class SealedSiteError(PermissionError):
    """The replication site was requested before it was unsealed."""


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
    items: list[dict] = []
    for month in stored_months(data_root, site):
        items.extend(read_month(data_root, site, month))
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
