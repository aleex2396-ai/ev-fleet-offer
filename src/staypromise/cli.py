"""Command line: ``staypromise <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from staypromise import acn, data, snapshots
from staypromise.pipeline import run_demo

# The months each site has sessions in the public API (checked 4 Nov 2026).
SITE_MONTHS = {"caltech": ("2018-04", "2021-09"), "jpl": ("2018-09", "2021-09")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="staypromise",
        description="Reliable lower bounds on EV dwell time when driver behaviour changes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo",
        help="run the whole pipeline on made-up sessions (a plumbing check, not a result)",
    )
    demo.add_argument("--out", type=Path, default=Path("results/demo"))
    demo.add_argument("--seed", type=int, default=0)

    fetch = commands.add_parser(
        "fetch",
        help="download one site's sessions from ACN-Data, month by month (needs a token)",
    )
    fetch.add_argument("site", choices=sorted(SITE_MONTHS))
    fetch.add_argument("--data", type=Path, default=Path("data"))
    fetch.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

    verify = commands.add_parser(
        "verify", help="check downloaded months against the committed fingerprints"
    )
    verify.add_argument("site", choices=sorted(SITE_MONTHS))
    verify.add_argument("--data", type=Path, default=Path("data"))
    verify.add_argument("--fingerprints", type=Path, default=Path("fingerprints"))

    snap = commands.add_parser(
        "snapshot-check",
        help="compare the downloaded Caltech sessions with the pinned public SustainGym extracts",
    )
    snap.add_argument("snapshots", type=Path, help="folder holding the four pinned CSV files")
    snap.add_argument("--data", type=Path, default=Path("data"))

    args = parser.parse_args(argv)
    if args.command == "demo":
        manifest = run_demo(args.out, seed=args.seed)
        print(f"wrote {len(manifest['outputs'])} files and manifest.json to {args.out}")
    elif args.command == "fetch":
        first, last = SITE_MONTHS[args.site]
        table = acn.fetch_site(
            args.site,
            first,
            last,
            data_root=args.data,
            fingerprint_dir=args.fingerprints,
            get=acn.http_getter(acn.read_token()),
        )
        print(f"{args.site}: {int(table['sessions'].sum())} sessions in {len(table)} months")
    elif args.command == "verify":
        report = acn.verify_fingerprints(args.data, args.fingerprints, args.site)
        bad = report[~report["matches"]]
        print(f"{args.site}: {len(report) - len(bad)} of {len(report)} months match")
        if len(bad):
            print(bad[["month", "sessions", "found_sessions"]].to_string(index=False))
            return 1
    elif args.command == "snapshot-check":
        sessions, counts = data.load_sessions(args.data, "caltech")
        print("cleaning:", ", ".join(f"{k} {v}" for k, v in counts.items()))
        print(snapshots.check_all(args.snapshots, sessions).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
