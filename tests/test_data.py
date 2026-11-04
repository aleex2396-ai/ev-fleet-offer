"""Parsing, cleaning, flags and the two seals, on made-up records."""

import ast
import subprocess
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


# Caltech's test period starts at midnight Pacific on 1 March 2020, which is 08:00 UTC.
BOUNDARY = [
    api_record(10, "Sat, 29 Feb 2020 20:00:00 GMT", "Sun, 01 Mar 2020 01:00:00 GMT"),
    api_record(
        11,
        "Sun, 01 Mar 2020 07:30:00 GMT",
        "Sun, 01 Mar 2020 09:00:00 GMT",
        user=5,
        inputs=[("Sun, 01 Mar 2020 07:30:05 GMT", 90)],
    ),
    api_record(
        12,
        "Sun, 01 Mar 2020 08:30:00 GMT",
        "Sun, 01 Mar 2020 10:00:00 GMT",
        user=5,
        inputs=[("Sun, 01 Mar 2020 08:30:05 GMT", 60)],
    ),
    api_record(13, "Tue, 01 Jun 2021 15:00:00 GMT", "Tue, 01 Jun 2021 17:00:00 GMT"),
]


def store_boundary(tmp_path):
    acn.save_month(tmp_path, "caltech", "2020-02", BOUNDARY[:1])
    acn.save_month(tmp_path, "caltech", "2020-03", BOUNDARY[1:3])
    acn.save_month(tmp_path, "caltech", "2021-06", BOUNDARY[3:])


def test_caltech_test_period_is_sealed_by_default(tmp_path):
    store_boundary(tmp_path)
    kept, _ = data.load_sessions(tmp_path, "caltech")
    assert sorted(kept["session_id"]) == ["s10", "s11"]
    assert set(data.load_user_inputs(tmp_path, "caltech")["session_id"]) == {"s11"}


def test_caltech_test_period_opens_only_once_the_plan_is_tagged(tmp_path, monkeypatch):
    store_boundary(tmp_path)
    monkeypatch.setattr(data, "plan_is_tagged", lambda: False)
    with pytest.raises(data.SealedPeriodError):
        data.load_sessions(tmp_path, "caltech", unseal=True)
    monkeypatch.setattr(data, "plan_is_tagged", lambda: True)
    kept, _ = data.load_sessions(tmp_path, "caltech", unseal=True)
    assert sorted(kept["session_id"]) == ["s10", "s11", "s12", "s13"]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_plan_tag_is_read_from_git(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "start",
    )
    monkeypatch.setattr(data, "REPO", tmp_path)
    assert not data.plan_is_tagged()
    git(tmp_path, "tag", data.PLAN_TAG)
    assert data.plan_is_tagged()
    monkeypatch.setattr(data, "REPO", tmp_path / "not-a-repository")
    assert not data.plan_is_tagged()
