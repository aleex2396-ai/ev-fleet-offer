"""The session table, parsing, cleaning, flags and the two seals, on made-up records."""

import ast
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import table

from fleetoffer import acn, data

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


def test_valid_table_is_sorted_and_in_utc():
    t = table([("b", 2, 3), ("a", 0, 1)])
    t["connect_time"] = t["connect_time"].dt.tz_convert("America/Los_Angeles")
    out = data.validate_sessions(t)
    assert list(out["session_id"]) == ["a", "b"]
    assert str(out["connect_time"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    "breaks",
    [
        lambda t: t.drop(columns="disconnect_time"),
        lambda t: t.assign(connect_time=t["connect_time"].dt.tz_localize(None)),
        lambda t: t.assign(disconnect_time=t["connect_time"]),
        lambda t: t.assign(session_id="same"),
    ],
    ids=["missing column", "no time zone", "zero-length stay", "duplicate ids"],
)
def test_invalid_tables_are_refused(breaks):
    with pytest.raises(ValueError):
        data.validate_sessions(breaks(table([("a", 0, 1), ("b", 2, 3)])))


def test_hours_round_trip():
    times = pd.Series(pd.to_datetime(["2019-03-10 09:15:30", "2021-09-14 23:59:59"], utc=True))
    back = data.from_hours(data.to_hours(times))
    assert np.abs((back - pd.DatetimeIndex(times)).total_seconds()).max() < 1e-3


# Per-session time series.


def series_record(
    session_id="s1", stamps=None, current=None, connect="Mon, 03 Jun 2019 15:00:00 GMT"
):
    stamps = stamps or [
        "Mon, 03 Jun 2019 15:00:00 GMT",
        "Mon, 03 Jun 2019 15:30:00 GMT",
        "Mon, 03 Jun 2019 16:00:00 GMT",
    ]
    return {
        "_id": f"ts-{session_id}",
        "sessionID": session_id,
        "connectionTime": connect,
        "chargingCurrent": {"timestamps": stamps, "current": current or [10.0, 20.0, 5.0]},
    }


def test_energy_holds_each_sample_until_the_next():
    series, reason = data.parse_series(series_record())
    assert reason == "usable"
    # 10 A for half an hour, then 20 A for half an hour, at 208 V; nothing after the last.
    assert data.energy_kwh(series) == pytest.approx((10 * 0.5 + 20 * 0.5) * 208 / 1000)
    start = series.times_h[0] + 0.25
    assert data.energy_kwh(series, start_h=start) == pytest.approx((10 * 0.25 + 20 * 0.5) * 0.208)
    assert data.energy_kwh(series, end_h=start) == pytest.approx(10 * 0.25 * 0.208)
    negative, _ = data.parse_series(series_record(current=[-1.0, 20.0, 5.0]))
    assert data.energy_kwh(negative) == pytest.approx(20 * 0.5 * 0.208)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda r: r.pop("chargingCurrent"), "no_series"),
        (lambda r: r["chargingCurrent"].update(current=[1.0, 2.0]), "length_mismatch"),
        (
            lambda r: r["chargingCurrent"].update(
                timestamps=["Mon, 03 Jun 2019 15:00:00 GMT"], current=[1.0]
            ),
            "too_short",
        ),
        (lambda r: r["chargingCurrent"]["timestamps"].__setitem__(1, "not a time"), "bad_time"),
        (
            lambda r: r["chargingCurrent"]["timestamps"].__setitem__(
                2, "Mon, 03 Jun 2019 15:30:00 GMT"
            ),
            "not_increasing",
        ),
        (lambda r: r["chargingCurrent"]["current"].__setitem__(1, None), "bad_current"),
        (lambda r: r["chargingCurrent"]["current"].__setitem__(1, float("nan")), "bad_current"),
    ],
)
def test_unusable_series_are_named(change, reason):
    record = series_record()
    change(record)
    series, why = data.parse_series(record)
    assert series is None and why == reason
    assert why in data.SERIES_REASONS


def test_stored_series_keep_the_seals(tmp_path):
    before = series_record("s1", connect="Sat, 29 Feb 2020 20:00:00 GMT")
    after = series_record("s2", connect="Sun, 01 Mar 2020 09:00:00 GMT")
    acn.save_series(tmp_path, "caltech", "2020-02-29", [before])
    acn.save_series(tmp_path, "caltech", "2020-03-01", [after])
    acn.save_series(tmp_path, "caltech", "named", [before, after])
    assert [r["sessionID"] for r in data.load_series(tmp_path, "caltech")] == ["s1"]
    assert [r["sessionID"] for r in data.load_series(tmp_path, "caltech", "named")] == ["s1"]
    with pytest.raises(data.SealedSiteError):
        data.load_series(tmp_path, "jpl")
