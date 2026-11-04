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
USER_AGENT = "ev-stay-promise (python urllib)"
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
