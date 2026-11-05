"""Downloading, storing, fingerprinting and cross-checking ACN-Data months, offline."""

import gzip
import hashlib
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
