#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphoflux.data import DataFactory, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize CellFlux-ready data tables.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "crispr_hep.yaml"),
        help="Path to the dataset factory YAML config.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verifying that raw assets exist under data/raw.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    factory = DataFactory(cfg, project_root=ROOT)
    audit = factory.materialize(verify_assets=not args.no_verify)

    summary = {
        "manifest_rows": audit["manifest_rows"],
        "controls": audit["counts"]["controls"],
        "targets": audit["counts"]["targets"],
        "target_genes": audit["counts"]["target_genes"],
        "pairs": {
            split: values["n_pairs"]
            for split, values in audit["pairs"]["splits"].items()
        },
        "manifest": audit["outputs"]["manifest"],
        "audit_report": audit["outputs"]["audit_report"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
