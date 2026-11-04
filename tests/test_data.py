"""Parsing, cleaning, flags and the replication seal, on made-up records."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from staypromise import acn, data

SRC = Path(data.__file__).parent


def api_record(i, connect, disconnect, *, user=None, inputs=(), session_id=None):
    return {
        "_id": f"id{i:04d}",
        "sessionID": session_id or f"s{i}",
        "connectionTime": connect,
        "disconnectTime": disconnect,
        "doneChargingTime": None,
        "kWhDelivered": 5.5,
        "userID": user,
        "stationID": "CA-1",
        "spaceID": "S-1",
        "userInputs": [
            {"modifiedAt": at, "minutesAvailable": minutes, "kWhRequested": 10.0}
            for at, minutes in inputs
        ]
        or None,
    }


RECORDS = [
    api_record(
        1,
        "Mon, 03 Jun 2019 15:00:00 GMT",
        "Mon, 03 Jun 2019 23:00:00 GMT",
        user=42,
        inputs=[("Mon, 03 Jun 2019 15:00:19 GMT", 480), ("Mon, 03 Jun 2019 16:00:00 GMT", 300)],
    ),
    api_record(
        2,
        "Mon, 03 Jun 2019 16:00:00 GMT",
        "Mon, 03 Jun 2019 17:00:00 GMT",
        user="7",
        inputs=[("Mon, 03 Jun 2019 15:59:59 GMT", 60)],
    ),
    api_record(3, "Mon, 03 Jun 2019 17:00:00 GMT", "Mon, 03 Jun 2019 18:30:00 GMT"),
    api_record(4, "Mon, 03 Jun 2019 18:00:00 GMT", None),
    api_record(5, "Mon, 03 Jun 2019 19:00:00 GMT", "Mon, 03 Jun 2019 19:00:00 GMT"),
    api_record(
        6, "Mon, 03 Jun 2019 20:00:00 GMT", "Mon, 03 Jun 2019 21:00:00 GMT", session_id="s3"
    ),
]


def test_parse_reads_utc_times_and_text_ids():
    table = data.parse_sessions(RECORDS)
    assert str(table["connect_time"].dt.tz) == "UTC"
    assert table.loc[0, "connect_time"] == pd.Timestamp("2019-06-03 15:00", tz="UTC")
    assert list(table["user_id"][:3]) == ["42", "7", None]
    assert pd.isna(table.loc[3, "disconnect_time"])


def test_cleaning_counts_every_dropped_record():
    kept, counts = data.clean_sessions(data.parse_sessions(RECORDS))
    assert counts == {"records": 6, "no_times": 1, "not_positive": 1, "duplicate": 1, "kept": 3}
    assert list(kept["session_id"]) == ["s1", "s2", "s3"]
    assert kept["connect_time"].is_monotonic_increasing
    assert data.stay_hours(kept).tolist() == [8.0, 1.0, 1.5]


def test_user_inputs_keep_every_entry_with_its_time():
    inputs = data.parse_user_inputs(RECORDS)
    assert list(inputs["session_id"]) == ["s1", "s1", "s2"]
    assert list(inputs["entries"]) == [2, 2, 1]
    assert inputs.loc[1, "modified_at"] == pd.Timestamp("2019-06-03 16:00", tz="UTC")


def test_flags_describe_how_inputs_were_recorded():
    kept, _ = data.clean_sessions(data.parse_sessions(RECORDS))
    flags = data.session_flags(kept, data.parse_user_inputs(RECORDS))
    assert list(flags["claimed"]) == [True, True, False]
    assert list(flags["typed"]) == [True, True, False]
    assert flags["input_lag_s"].tolist()[:2] == [19.0, -1.0]
    assert list(flags["input_with_plugin"]) == [False, True, False]
    assert list(flags["typed_480"]) == [True, False, False]
    assert list(flags["typed_60"]) == [False, True, False]
    assert list(flags["edits"]) == [1, 0, 0]


def store(tmp_path, site):
    acn.save_month(tmp_path, site, "2019-06", RECORDS[:3])
    acn.save_month(tmp_path, site, "2019-07", RECORDS[3:])


def test_load_sessions_reads_all_stored_months(tmp_path):
    store(tmp_path, "caltech")
    kept, counts = data.load_sessions(tmp_path, "caltech")
    assert counts["kept"] == 3
    assert data.missing_months(tmp_path, "caltech", "2019-05", "2019-08") == ["2019-05", "2019-08"]


def test_replication_site_is_sealed(tmp_path):
    store(tmp_path, "jpl")
    with pytest.raises(data.SealedSiteError):
        data.load_sessions(tmp_path, "jpl")
    with pytest.raises(data.SealedSiteError):
        data.load_user_inputs(tmp_path, "jpl")
    kept, _ = data.load_sessions(tmp_path, "jpl", unseal=True)
    assert len(kept) == 3


def test_no_library_code_unseals_the_replication_site():
    """Only an explicit replication run may pass unseal=True; library code never does."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "unseal":
                value = node.value
                passes_through = isinstance(value, ast.Name) and value.id == "unseal"
                if not passes_through:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
