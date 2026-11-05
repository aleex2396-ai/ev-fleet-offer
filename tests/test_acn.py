"""Downloading, storing, fingerprinting and cross-checking ACN-Data months, offline."""

import gzip
import hashlib
import re
import urllib.parse

import pandas as pd
import pytest

from fleetoffer import acn


def record(i: int, connect: str = "Sat, 01 Jun 2019 15:00:00 GMT") -> dict:
    return {
        "_id": f"id{i:04d}",
        "sessionID": f"s{i}",
        "connectionTime": connect,
        "disconnectTime": "Sat, 01 Jun 2019 23:00:00 GMT",
        "_links": {"self": {"href": f"sessions/{i}"}},
    }


class FakeServer:
    """Serves `n` records in pages, the way the API does; can drop or repeat one."""

    def __init__(self, n: int, *, glitch_once: str | None = None):
        self.records = [record(i) for i in range(n)]
        self.glitch_once = glitch_once
        self.calls: list[str] = []

    def __call__(self, url: str) -> dict:
        self.calls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        page, size = int(query["page"][0]), int(query["max_results"][0])
        rows = [dict(r) for r in self.records[(page - 1) * size : page * size]]
        if self.glitch_once == "drop" and page == 1:
            rows = rows[1:]
            self.glitch_once = None
        elif self.glitch_once == "repeat" and page == 2:
            rows = [dict(self.records[0]), *rows[1:]]
            self.glitch_once = None
        return {"_items": rows, "_meta": {"total": len(self.records), "page": page}}


def test_month_query_uses_a_half_open_utc_window():
    query = urllib.parse.parse_qs(
        urllib.parse.urlparse(acn.month_query("caltech", "2019-12", 3)).query
    )
    assert query["where"] == [
        'connectionTime>="Sun, 01 Dec 2019 00:00:00 GMT" and '
        'connectionTime<"Wed, 01 Jan 2020 00:00:00 GMT"'
    ]
    assert query["page"] == ["3"]
    assert query["max_results"] == [str(acn.PAGE_SIZE)]


@pytest.mark.parametrize("n", [0, 1, 99, 100, 101, 250])
def test_fetch_month_reads_every_page_and_drops_links(n):
    items = acn.fetch_month("caltech", "2019-06", FakeServer(n))
    assert [r["_id"] for r in items] == [f"id{i:04d}" for i in range(n)]
    assert all("_links" not in r for r in items)


@pytest.mark.parametrize(("glitch", "calls"), [("drop", 1 + 3), ("repeat", 3 + 3)])
def test_fetch_month_refetches_an_incomplete_month(glitch, calls):
    # A short first page ends the pass early; a repeated record is caught by the ID check.
    server = FakeServer(250, glitch_once=glitch)
    items = acn.fetch_month("caltech", "2019-06", server)
    assert len({r["_id"] for r in items}) == len(items) == 250
    assert len(server.calls) == calls


def test_fetch_month_gives_up_when_the_count_never_matches():
    class AlwaysShort(FakeServer):
        def __call__(self, url):
            data = super().__call__(url)
            data["_meta"]["total"] += 1
            return data

    with pytest.raises(acn.FetchError, match="server reports 11"):
        acn.fetch_month("caltech", "2019-06", AlwaysShort(10))


def test_unknown_site_is_refused():
    with pytest.raises(ValueError):
        acn.fetch_month("pasadena", "2019-06", FakeServer(1))


def test_fingerprint_ignores_record_order_and_key_order():
    a = [record(1), record(2)]
    b = [dict(reversed(list(record(2).items()))), record(1)]
    assert acn.fingerprint(a) == acn.fingerprint(b)
    changed = [record(1), record(2, connect="Sat, 01 Jun 2019 15:00:01 GMT")]
    assert acn.fingerprint(a) != acn.fingerprint(changed)


def test_stored_months_are_byte_stable_and_round_trip(tmp_path):
    items = [record(2), record(1)]
    first = acn.save_month(tmp_path, "caltech", "2019-06", items).read_bytes()
    second = acn.save_month(tmp_path, "caltech", "2019-06", list(reversed(items))).read_bytes()
    assert first == second
    assert gzip.decompress(first) == acn.canonical_bytes(items)
    assert acn.read_month(tmp_path, "caltech", "2019-06") == sorted(items, key=lambda r: r["_id"])


def test_fetch_site_writes_fingerprints_that_verify(tmp_path):
    table = acn.fetch_site(
        "caltech",
        "2019-11",
        "2020-01",
        data_root=tmp_path / "data",
        fingerprint_dir=tmp_path / "fp",
        get=FakeServer(120),
        progress=lambda line: None,
    )
    assert list(table["month"]) == ["2019-11", "2019-12", "2020-01"]
    report = acn.verify_fingerprints(tmp_path / "data", tmp_path / "fp", "caltech")
    assert report["matches"].all()

    acn.save_month(tmp_path / "data", "caltech", "2019-12", [record(1)])
    report = acn.verify_fingerprints(tmp_path / "data", tmp_path / "fp", "caltech")
    assert list(report.loc[~report["matches"], "month"]) == ["2019-12"]


def test_progress_lines_carry_counts_only(tmp_path):
    lines: list[str] = []
    acn.fetch_site(
        "jpl",
        "2019-06",
        "2019-06",
        data_root=tmp_path,
        fingerprint_dir=tmp_path,
        get=FakeServer(3),
        progress=lines.append,
    )
    assert lines == ["jpl 2019-06: 3 sessions"]


def test_token_comes_from_the_environment_or_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ACN_API_TOKEN", "  from-env \n")
    assert acn.read_token() == "from-env"
    monkeypatch.delenv("ACN_API_TOKEN")
    key = tmp_path / "key"
    key.write_text("from-file\n")
    monkeypatch.setenv("ACN_API_TOKEN_FILE", str(key))
    assert acn.read_token() == "from-file"
    monkeypatch.setenv("ACN_API_TOKEN_FILE", str(tmp_path / "missing"))
    with pytest.raises(acn.FetchError, match="no ACN-Data token"):
        acn.read_token()


def test_months_between_includes_both_ends():
    assert acn.months_between("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
    start, end = acn.month_window("2020-02")
    assert (start, end) == (
        pd.Timestamp("2020-02-01", tz="UTC"),
        pd.Timestamp("2020-03-01", tz="UTC"),
    )


# The independent public copy.

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
    start, end = acn.window_of(NAME)
    assert start == pd.Timestamp("2019-05-01 07:00", tz="UTC")
    assert end == pd.Timestamp("2019-09-01 07:00", tz="UTC")


def test_read_snapshot_refuses_files_that_are_not_pinned(tmp_path):
    with pytest.raises(acn.SnapshotError, match="not one of"):
        acn.read_snapshot(write(tmp_path / "other.csv.gz", CSV))
    with pytest.raises(acn.SnapshotError, match="SHA-256"):
        acn.read_snapshot(write(tmp_path / NAME, CSV))


def test_read_and_compare(tmp_path, monkeypatch):
    path = write(tmp_path / NAME, CSV)
    monkeypatch.setitem(acn.SNAPSHOT_FILES, NAME, hashlib.sha256(path.read_bytes()).hexdigest())
    snap = acn.read_snapshot(path)
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
    result = acn.compare_snapshot(snap, table, NAME)
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


def test_check_snapshots_lists_but_does_not_compare_sealed_ones(tmp_path, monkeypatch):
    path = write(tmp_path / NAME, CSV)
    later = "2021-05-01 2021-08-31.csv.gz"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(acn, "SNAPSHOT_FILES", {NAME: sha, later: "not needed"})
    table = ours([("a", "2019-05-01 08:18:45", "2019-05-01 22:52:36", "42")])
    sealed_from = pd.Timestamp("2020-03-01", tz="America/Los_Angeles")
    result = acn.check_snapshots(tmp_path, table, sealed_from)
    assert list(result["status"]) == ["compared", "sealed test period"]
    assert result.loc[0, "both"] == 1


# Per-session time series.

TS_FMT = "%a, %d %b %Y %H:%M:%S GMT"


def ts_record(i: int, connect: pd.Timestamp, samples: int = 3) -> dict:
    times = [(connect + pd.Timedelta(seconds=4 * k)).strftime(TS_FMT) for k in range(samples)]
    return {
        "_id": f"ts{i:04d}",
        "sessionID": f"s{i}",
        "connectionTime": connect.strftime(TS_FMT),
        "chargingCurrent": {"timestamps": times, "current": [10.0] * samples},
        "_links": {"self": {"href": f"sessions/{i}"}},
    }


class FakeSeriesServer:
    """Answers `sessions/<site>/ts` window queries in pages, as the API does."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.calls: list[str] = []

    def __call__(self, url: str) -> dict:
        self.calls.append(url)
        assert "/ts?" in url
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        lo, hi = re.findall(r'"([^"]+)"', query["where"][0])
        lo, hi = (pd.to_datetime(x, format=TS_FMT, utc=True) for x in (lo, hi))
        inside = [
            r
            for r in self.records
            if lo <= pd.to_datetime(r["connectionTime"], format=TS_FMT, utc=True) < hi
        ]
        page, size = int(query["page"][0]), int(query["max_results"][0])
        rows = [dict(r) for r in inside[(page - 1) * size : page * size]]
        return {"_items": rows, "_meta": {"total": len(inside), "page": page}}


def ts_records():
    start = pd.Timestamp("2020-02-27 16:00", tz="UTC")
    return [ts_record(i, start + pd.Timedelta(hours=7 * i)) for i in range(25)]


def test_series_windows_are_utc_days_up_to_the_sealed_period():
    until = pd.Timestamp("2020-03-01", tz="America/Los_Angeles")  # 08:00 UTC
    windows = acn.series_windows("2020-02-27", until)
    assert [w[0] for w in windows] == ["2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01"]
    assert windows[0][2] - windows[0][1] == pd.Timedelta(days=1)
    assert windows[-1][1:] == (
        pd.Timestamp("2020-03-01 00:00", tz="UTC"),
        pd.Timestamp("2020-03-01 08:00", tz="UTC"),
    )


def test_fetch_series_site_clips_and_resumes(tmp_path):
    server = FakeSeriesServer(ts_records())
    until = pd.Timestamp("2020-03-01", tz="America/Los_Angeles")
    kwargs = {"until": until, "data_root": tmp_path, "fingerprint_dir": tmp_path, "get": server}
    table = acn.fetch_series_site("caltech", "2020-02-27", **kwargs, progress=lambda _: 0)
    # Sessions every 7 hours from 27 Feb 16:00 UTC (2020 is a leap year): s0 and s1 on the
    # 27th, s2 to s4 on the 28th, s5 to s7 on the 29th, then s8 (1 March 00:00) and s9
    # (07:00) before 08:00 UTC; later ones are never requested.
    assert list(table["sessions"]) == [2, 3, 3, 2]
    assert all("ts?" in call for call in server.calls)
    stored = acn.read_series(tmp_path, "caltech", "2020-03-01")
    assert sorted(r["sessionID"] for r in stored) == ["s8", "s9"]
    assert all("_links" not in r for r in stored)

    calls = len(server.calls)
    again = acn.fetch_series_site("caltech", "2020-02-27", **kwargs, progress=lambda _: 0)
    assert len(server.calls) == calls  # nothing fetched twice
    pd.testing.assert_frame_equal(again, table)
    on_disk = pd.read_csv(tmp_path / "caltech-series.csv", dtype=str)
    assert list(on_disk["day"]) == ["2020-02-27", "2020-02-28", "2020-02-29", "2020-03-01"]
    report = acn.verify_series(tmp_path, tmp_path, "caltech-series", "caltech")
    assert report["matches"].all()


def test_series_pages_hold_ten_records():
    server = FakeSeriesServer(ts_records())
    start, end = pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-04-01", tz="UTC")
    items = acn.fetch_series_window("caltech", start, end, server)
    assert len(items) == 25 and len(server.calls) == 3  # pages of 10, 10 and 5


def test_fetch_series_sessions_takes_each_listed_session(tmp_path):
    records = ts_records()
    twin = ts_record(99, pd.to_datetime(records[3]["connectionTime"], format=TS_FMT, utc=True))
    server = FakeSeriesServer([*records, twin])  # s99 plugs in at the same second as s3
    listed = pd.DataFrame(
        {
            "session_id": ["s3", "s10"],
            "connect_time": [
                pd.to_datetime(records[k]["connectionTime"], format=TS_FMT, utc=True)
                for k in (3, 10)
            ],
        }
    )
    table = acn.fetch_series_sessions(
        "caltech", listed, "sample", data_root=tmp_path, fingerprint_dir=tmp_path, get=server
    )
    assert len(server.calls) == 2 and table.loc[0, "sessions"] == 2
    stored = acn.read_series(tmp_path, "caltech", "sample")
    assert sorted(r["sessionID"] for r in stored) == ["s10", "s3"]
    assert acn.verify_series(tmp_path, tmp_path, "caltech-series-sample", "caltech")[
        "matches"
    ].all()
