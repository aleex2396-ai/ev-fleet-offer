"""Download ACN-Data sessions, one calendar month at a time, and store them with fingerprints.

ACN-Data (https://ev.caltech.edu/dataset) needs a free access token. It is read from the
environment variable ``ACN_API_TOKEN``, or from the file named by ``ACN_API_TOKEN_FILE``
(default ``~/.config/acndata/api_key``). It is sent only in the request header and never
written to disk or printed.

Each month is fetched with a ``connectionTime`` window in UTC (start included, end
excluded), 100 sessions a page. A month is accepted only if the number of sessions received
equals the total the server reports and no session appears twice; otherwise it is fetched
again. Large single requests time out on the server, which is why the unit is a month.

Each month is stored as canonical JSON (sessions sorted by ``_id``, keys sorted, no
whitespace), gzip-compressed, in ``data/raw/<site>/<YYYY-MM>.json.gz``. Its fingerprint is the
SHA-256 of the canonical JSON. Fingerprints are committed in ``fingerprints/<site>.csv`` so
anyone can confirm they analysed the same data without the data being redistributed.

The download is also checked against an independent public copy. SustainGym (Yeh et al.,
NeurIPS 2023 Datasets and Benchmarks) ships four extracts of the Caltech site as CSV files
under CC-BY-4.0. They were made from the same API by other people, years earlier, so
agreement with them checks the download and the time handling. The files are pinned to one
commit and checked by SHA-256 before use; get them from

    https://github.com/chrisyeh96/sustaingym/tree/<SNAPSHOT_COMMIT>/sustaingym/data/evcharging/acn_data/caltech

Each file covers whole days in Pacific time, both ends included, as its name says.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

API = "https://ev.caltech.edu/api/v1/sessions/"
SITES = ("caltech", "jpl", "office001")
PAGE_SIZE = 100
USER_AGENT = "ev-fleet-offer (python urllib)"
DEFAULT_TOKEN_FILE = "~/.config/acndata/api_key"

Getter = Callable[[str], dict]


class FetchError(RuntimeError):
    """A month could not be downloaded completely."""


def read_token() -> str:
    token = os.environ.get("ACN_API_TOKEN", "").strip()
    if token:
        return token
    path = Path(os.environ.get("ACN_API_TOKEN_FILE", DEFAULT_TOKEN_FILE)).expanduser()
    try:
        token = path.read_text().strip()
    except OSError:
        token = ""
    if not token:
        raise FetchError(
            "no ACN-Data token: set ACN_API_TOKEN, or put it in the file named by "
            f"ACN_API_TOKEN_FILE (default {DEFAULT_TOKEN_FILE})"
        )
    return token


def month_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The UTC start (included) and end (excluded) of a month written as YYYY-MM."""
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def month_query(site: str, month: str, page: int) -> str:
    start, end = month_window(month)
    fmt = "%a, %d %b %Y %H:%M:%S GMT"
    where = f'connectionTime>="{start.strftime(fmt)}" and connectionTime<"{end.strftime(fmt)}"'
    params = {"where": where, "page": page, "max_results": PAGE_SIZE}
    return f"{API}{site}?{urllib.parse.urlencode(params)}"


def http_getter(token: str, *, retries: int = 4, timeout: float = 90.0) -> Getter:
    """A function that GETs a URL with the token and returns the parsed JSON."""
    auth = "Basic " + base64.b64encode(f"{token}:".encode()).decode()

    def get(url: str) -> dict:
        delay = 2.0
        for attempt in range(retries):
            request = urllib.request.Request(
                url, headers={"Authorization": auth, "User-Agent": USER_AGENT}
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as err:
                retryable = err.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == retries - 1:
                    raise FetchError(f"HTTP {err.code} from the ACN-Data API") from None
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                if attempt == retries - 1:
                    raise FetchError(f"network error: {type(err).__name__}") from None
            time.sleep(delay)
            delay *= 2
        raise FetchError("request failed")

    return get


def fetch_month(site: str, month: str, get: Getter, *, attempts: int = 3) -> list[dict]:
    """Every session of one site that plugged in during one UTC month."""
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}")
    for _ in range(attempts):
        items: list[dict] = []
        page, total = 1, None
        while True:
            data = get(month_query(site, month, page))
            batch = data.get("_items") or []
            total = (data.get("_meta") or {}).get("total")
            for item in batch:
                item.pop("_links", None)
            items.extend(batch)
            if len(batch) < PAGE_SIZE or (total is not None and len(items) >= total):
                break
            page += 1
        ids = [item["_id"] for item in items]
        if total is not None and len(items) == total and len(set(ids)) == len(ids):
            return items
    raise FetchError(
        f"{site} {month}: received {len(items)} sessions ({len(set(ids))} distinct) "
        f"but the server reports {total}"
    )


def canonical_bytes(items: list[dict]) -> bytes:
    ordered = sorted(items, key=lambda item: item["_id"])
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fingerprint(items: list[dict]) -> str:
    return hashlib.sha256(canonical_bytes(items)).hexdigest()


def raw_path(root: Path, site: str, month: str) -> Path:
    return Path(root) / "raw" / site / f"{month}.json.gz"


def save_month(root: Path, site: str, month: str, items: list[dict]) -> Path:
    path = raw_path(root, site, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(canonical_bytes(items))
    return path


def read_month(root: Path, site: str, month: str) -> list[dict]:
    with gzip.open(raw_path(root, site, month), "rb") as f:
        return json.loads(f.read())


def months_between(first: str, last: str) -> list[str]:
    """YYYY-MM strings from `first` to `last`, both included."""
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def fetch_site(
    site: str,
    first: str,
    last: str,
    *,
    data_root: Path,
    fingerprint_dir: Path,
    get: Getter,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Download each month, store it, and write the site's fingerprint table.

    Progress lines report counts only, never session contents.
    """
    rows = []
    for month in months_between(first, last):
        items = fetch_month(site, month, get)
        save_month(data_root, site, month, items)
        rows.append({"month": month, "sessions": len(items), "sha256": fingerprint(items)})
        progress(f"{site} {month}: {len(items)} sessions")
    table = pd.DataFrame(rows, columns=["month", "sessions", "sha256"])
    write_fingerprints(fingerprint_dir, site, table)
    return table


def write_fingerprints(directory: Path, site: str, table: pd.DataFrame) -> Path:
    path = Path(directory) / f"{site}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return path


def verify_fingerprints(data_root: Path, fingerprint_dir: Path, site: str) -> pd.DataFrame:
    """Recompute every stored month's fingerprint and compare with the committed table."""
    expected = pd.read_csv(Path(fingerprint_dir) / f"{site}.csv", dtype={"month": str})
    found = []
    for month in expected["month"]:
        path = raw_path(data_root, site, month)
        if not path.exists():
            found.append({"month": month, "found_sessions": None, "found_sha256": None})
            continue
        items = read_month(data_root, site, month)
        found.append(
            {"month": month, "found_sessions": len(items), "found_sha256": fingerprint(items)}
        )
    report = expected.merge(pd.DataFrame(found), on="month")
    report["matches"] = report["sha256"] == report["found_sha256"]
    return report


# The independent public copy (SustainGym extracts of the Caltech site).

SNAPSHOT_COMMIT = "eac2a4d5ce4ccf44b13c290edac72532c68e30f5"
SNAPSHOT_FILES = {
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
    expected = SNAPSHOT_FILES.get(path.name)
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
    start = pd.Timestamp(first, tz="America/Los_Angeles")
    end = pd.Timestamp(last, tz="America/Los_Angeles") + pd.Timedelta(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def compare_snapshot(snapshot: pd.DataFrame, sessions: pd.DataFrame, name: str) -> dict:
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


def check_snapshots(
    folder: Path, sessions: pd.DataFrame, sealed_from: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Compare every pinned snapshot in `folder` with `sessions` (the cleaned Caltech table).

    A snapshot reaching into a sealed period (from `sealed_from` on) is listed but not compared.
    """
    rows = []
    for name in SNAPSHOT_FILES:
        if sealed_from is not None and window_of(name)[1] > sealed_from:
            rows.append({"file": name, "status": "sealed test period"})
            continue
        path = Path(folder) / name
        if not path.exists():
            raise SnapshotError(f"missing {name}; see the acn module docstring for where to get it")
        rows.append({**compare_snapshot(read_snapshot(path), sessions, name), "status": "compared"})
    return pd.DataFrame(rows)
