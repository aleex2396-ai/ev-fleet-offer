"""Command line: ``staypromise <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from staypromise.pipeline import run_demo


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

    args = parser.parse_args(argv)
    if args.command == "demo":
        manifest = run_demo(args.out, seed=args.seed)
        print(f"wrote {len(manifest['outputs'])} files and manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
