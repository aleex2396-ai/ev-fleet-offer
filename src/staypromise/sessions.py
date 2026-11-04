"""The session table: one row per charging session, and the checks it must pass.

Every other module reads sessions through `validate_sessions`, so the table's rules live in
one place:

- ``session_id``: unique text identifier.
- ``connect_time``, ``disconnect_time``: timezone-aware timestamps, with the car leaving
  strictly after it plugs in.
- ``user_id`` (optional): the driver's identifier, or missing.

Times are handled internally as float hours since 1970-01-01 UTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
