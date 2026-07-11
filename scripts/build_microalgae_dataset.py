#!/usr/bin/env python3
"""Build the canonical timepoint microalgae processed dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_views(raw: str) -> set[str]:
    if raw == "all":
        return {"timepoint", "field"}
    views = {item.strip() for item in raw.split(",") if item.strip()}
    valid = {"timepoint", "field"}
    unknown = views - valid
    if unknown:
        raise ValueError(
            "only the active timepoint and field views are supported; "
            f"unknown or archived view(s): {', '.join(sorted(unknown))}"
        )
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description="Build microalgae_v1 processed data")
    parser.add_argument(
        "--version",
        default="microalgae_v1",
        choices=["microalgae_v1"],
        help="Processed dataset version to build.",
    )
    parser.add_argument(
        "--views",
        default="timepoint",
        help="Comma-separated views: timepoint,field, or all. Default: timepoint.",
    )
    args = parser.parse_args()

    views = parse_views(args.views)

    if "field" in views:
        run([sys.executable, "scripts/build_field_metadata.py"])

    if "timepoint" in views:
        run([sys.executable, "scripts/build_timegroup_data.py"])

    if "field" in views:
        run([sys.executable, "scripts/build_field_dataset.py"])

    run(["bash", "scripts/migrate_processed_layout.sh"])
    print(f"Processed data ready under data/processed/{args.version}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
