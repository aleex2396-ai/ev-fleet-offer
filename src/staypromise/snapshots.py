"""Compare the downloaded Caltech sessions with an independent public copy.

SustainGym (Yeh et al., NeurIPS 2023 Datasets and Benchmarks) ships four extracts of the
Caltech site as CSV files under CC-BY-4.0. They were made from the same API by other people,
years earlier, so agreement with them checks the download and the time handling. The files are
pinned to one commit and checked by SHA-256 before use; get them from

    https://github.com/chrisyeh96/sustaingym/tree/<COMMIT>/sustaingym/data/evcharging/acn_data/caltech

Each file covers whole days in Pacific time, both ends included, as its name says.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

COMMIT = "eac2a4d5ce4ccf44b13c290edac72532c68e30f5"
SITE_TZ = "America/Los_Angeles"
FILES = {
    "2019-05-01 2019-08-31.csv.gz": (
        "d23a07e3b5175dca0da107610557f604fa177f71bfc696687c70a0470964047b"
    ),
    "2019-09-01 2019-12-31.csv.gz": (
        "c9181c918e2a834df4639e6c3e29328ed776d61419c9c89108e3c9e2b649919d"
    ),
    "2020-02-01 2020-05-31.csv.gz": (
        "794ad5e6bb8cf10d31ecaa7296a66aa171ee4ed1c9ab81933c6b4eeb3c38649e"
    ),
    "2021-05-01 2021-08-31.csv.gz": (
        "3d35712b2e390104425639bf00113189aac3ad1851c71c9a4a6f35e792f92414"
    ),
}


class SnapshotError(ValueError):
    """A snapshot file is missing or is not the pinned one."""


def read_snapshot(path: Path) -> pd.DataFrame:
    """One pinned snapshot as session_id, UTC plug-in and unplug times, and `claimed`."""
    path = Path(path)
    expected = FILES.get(path.name)
    if expected is None:
        raise SnapshotError(f"{path.name} is not one of the pinned snapshot files")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SnapshotError(f"{path.name} does not match its pinned SHA-256")
    raw = pd.read_csv(path, dtype={"session_id": str})
    return pd.DataFrame(
        {
            "session_id": raw["session_id"],
            "connect_time": pd.to_datetime(raw["arrival"], utc=True),
            "disconnect_time": pd.to_datetime(raw["departure"], utc=True),
            "claimed": raw["claimed"].astype(bool),
        }
    )


def window_of(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC start (included) and end (excluded) of the Pacific-time days a file covers."""
    first, last = name.removesuffix(".csv.gz").split(" ")
    start = pd.Timestamp(first, tz=SITE_TZ)
    end = pd.Timestamp(last, tz=SITE_TZ) + pd.Timedelta(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def compare(snapshot: pd.DataFrame, sessions: pd.DataFrame, name: str) -> dict:
    """Counts of agreement between one snapshot and the downloaded sessions on the same days."""
    start, end = window_of(name)
    ours = sessions[(sessions["connect_time"] >= start) & (sessions["connect_time"] < end)]
    both = snapshot.merge(ours, on="session_id", suffixes=("_snap", "_ours"))
    connect_gap = (both["connect_time_snap"] - both["connect_time_ours"]).abs()
    disconnect_gap = (both["disconnect_time_snap"] - both["disconnect_time_ours"]).abs()
    one_second = pd.Timedelta(seconds=1)
    return {
        "file": name,
        "snapshot": len(snapshot),
        "ours": len(ours),
        "both": len(both),
        "only_snapshot": int((~snapshot["session_id"].isin(ours["session_id"])).sum()),
        "only_ours": int((~ours["session_id"].isin(snapshot["session_id"])).sum()),
        "times_within_1s": int(
            ((connect_gap <= one_second) & (disconnect_gap <= one_second)).sum()
        ),
        "max_gap_s": float(max(connect_gap.max(), disconnect_gap.max()) / one_second)
        if len(both)
        else 0.0,
        "claimed_agree": int((both["claimed"] == both["user_id"].notna()).sum()),
    }


def check_all(
    folder: Path, sessions: pd.DataFrame, sealed_from: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Compare every pinned snapshot in `folder` with `sessions` (the cleaned Caltech table).

    A snapshot reaching into a sealed period (from `sealed_from` on) is listed but not compared.
    """
    rows = []
    for name in FILES:
        if sealed_from is not None and window_of(name)[1] > sealed_from:
            rows.append({"file": name, "status": "sealed test period"})
            continue
        path = Path(folder) / name
        if not path.exists():
            raise SnapshotError(f"missing {name}; see the module docstring for where to get it")
        rows.append({**compare(read_snapshot(path), sessions, name), "status": "compared"})
    return pd.DataFrame(rows)
