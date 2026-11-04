"""Comparison with the public SustainGym extracts, on made-up files."""

import gzip
import hashlib

import pandas as pd
import pytest

from staypromise import snapshots

NAME = "2019-05-01 2019-08-31.csv.gz"
HEADER = (
    "arrival,departure,requested_energy (kWh),delivered_energy (kWh),"
    "station_id,session_id,estimated_departure,claimed"
)
ROWS = [
    "2019-05-01 01:18:45-07:00,2019-05-01 15:52:36-07:00,60.0,44.0,CA-305,a,"
    "2019-05-01 09:20:45-07:00,True",
    "2019-08-31 23:30:00-07:00,2019-09-01 01:00:00-07:00,10.0,5.0,CA-306,b,"
    "2019-09-01 00:30:00-07:00,False",
]
CSV = "\n".join([HEADER, *ROWS]) + "\n"


def write(path, text):
    path.write_bytes(gzip.compress(text.encode(), mtime=0))
    return path


def ours(rows):
    return pd.DataFrame(
        [
            {
                "session_id": sid,
                "connect_time": pd.Timestamp(c, tz="UTC"),
                "disconnect_time": pd.Timestamp(d, tz="UTC"),
                "user_id": user,
            }
            for sid, c, d, user in rows
        ]
    )


def test_window_covers_whole_pacific_days_in_utc():
    start, end = snapshots.window_of(NAME)
    assert start == pd.Timestamp("2019-05-01 07:00", tz="UTC")
    assert end == pd.Timestamp("2019-09-01 07:00", tz="UTC")


def test_read_snapshot_refuses_files_that_are_not_pinned(tmp_path):
    with pytest.raises(snapshots.SnapshotError, match="not one of"):
        snapshots.read_snapshot(write(tmp_path / "other.csv.gz", CSV))
    with pytest.raises(snapshots.SnapshotError, match="SHA-256"):
        snapshots.read_snapshot(write(tmp_path / NAME, CSV))


def test_read_and_compare(tmp_path, monkeypatch):
    path = write(tmp_path / NAME, CSV)
    monkeypatch.setitem(snapshots.FILES, NAME, hashlib.sha256(path.read_bytes()).hexdigest())
    snap = snapshots.read_snapshot(path)
    assert snap.loc[0, "connect_time"] == pd.Timestamp("2019-05-01 08:18:45", tz="UTC")
    assert snap.loc[1, "disconnect_time"] == pd.Timestamp("2019-09-01 08:00", tz="UTC")

    table = ours(
        [
            ("a", "2019-05-01 08:18:45", "2019-05-01 22:52:36", "42"),
            ("b", "2019-09-01 06:30:00", "2019-09-01 08:00:02", None),
            ("c", "2019-06-01 12:00:00", "2019-06-01 13:00:00", None),
            ("d", "2019-09-01 07:00:00", "2019-09-01 08:00:00", None),  # after the window
        ]
    )
    result = snapshots.compare(snap, table, NAME)
    assert result == {
        "file": NAME,
        "snapshot": 2,
        "ours": 3,
        "both": 2,
        "only_snapshot": 0,
        "only_ours": 1,
        "times_within_1s": 1,
        "max_gap_s": 2.0,
        "claimed_agree": 2,
    }
