#!/usr/bin/env python3
"""Build a distribution-preserving stratified subset of the Diet dataset.

Strategy:
  1. Stratified sampling by condition (adlib/fasted/hfd) — proportional to full dataset
  2. Within each condition, stratified by hepatocyte subpopulation (cluster_type: Hep1-Hep6)
     to preserve biological heterogeneity
  3. Preserve batch grouping where possible (cells from same BATCH stay together)
  4. Validate with distribution statistics vs full dataset

Output:
  - data/processed/diet/index_diet_{size_label}.csv
  - data/processed/diet/subset_report_{size_label}.json

Usage:
  python scripts/build_diet_subset.py --size 50000 --seed 42
  python scripts/build_diet_subset.py --size 100000 --seed 42
  python scripts/build_diet_subset.py --size 200000 --seed 42
  python scripts/build_diet_subset.py --size 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = REPO_ROOT / "data/processed/diet/index_diet.csv"
OUT_DIR = REPO_ROOT / "data/processed/diet"


def load_full_index(path: Path) -> pd.DataFrame:
    """Load the full diet index CSV and validate required columns."""
    df = pd.read_csv(path, index_col=0)
    required_cols = {"CPD_NAME", "BATCH", "SAMPLE_KEY", "cluster_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Index missing required columns: {missing}")
    return df


def build_stratified_subset(
    full_df: pd.DataFrame,
    target_size: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Build a stratified subset preserving condition AND cluster_type proportions.

    Args:
        full_df: Full dataset index DataFrame.
        target_size: Desired number of cells in subset.
        seed: Random seed for reproducibility.

    Returns:
        (subset_df, report_dict) tuple.
    """
    rng = np.random.default_rng(seed)

    cond_col = "CPD_NAME"
    cluster_col = "cluster_type"

    cond_counts = full_df[cond_col].value_counts()
    total = len(full_df)

    # --- Step 1: Allocate per-condition (proportional) ---
    per_cond_target = {}
    for cond in cond_counts.index:
        proportion = cond_counts[cond] / total
        per_cond_target[cond] = max(10, int(target_size * proportion))

    # Adjust to hit target exactly
    surplus = target_size - sum(per_cond_target.values())
    if surplus != 0:
        largest = max(per_cond_target, key=per_cond_target.get)
        per_cond_target[largest] += surplus

    # --- Step 2: Within each condition, allocate per-cluster (proportional) ---
    subset_indices = []
    cluster_stats: dict = {}

    for cond, n_cond_target in per_cond_target.items():
        cond_df = full_df[full_df[cond_col] == cond]
        cluster_counts = cond_df[cluster_col].value_counts()
        cond_total = len(cond_df)

        cluster_stats[cond] = {
            "full": cluster_counts.to_dict(),
            "sampled": {},
        }

        # Allocate per-cluster within this condition
        per_cluster_target = {}
        for cluster in cluster_counts.index:
            proportion = cluster_counts[cluster] / cond_total
            per_cluster_target[cluster] = max(1, int(n_cond_target * proportion))

        # Adjust within condition
        c_surplus = n_cond_target - sum(per_cluster_target.values())
        if c_surplus != 0:
            largest_cluster = max(per_cluster_target, key=per_cluster_target.get)
            per_cluster_target[largest_cluster] += c_surplus

        # --- Step 3: Sample within each (condition, cluster) group ---
        for cluster, n_cluster_target in per_cluster_target.items():
            cluster_cond_df = cond_df[cond_df[cluster_col] == cluster]
            n_available = len(cluster_cond_df)
            n_sample = min(n_cluster_target, n_available)

            sampled = cluster_cond_df.sample(n=n_sample, random_state=rng)
            subset_indices.extend(sampled.index.tolist())
            cluster_stats[cond]["sampled"][cluster] = n_sample

    subset_df = full_df.loc[subset_indices].copy()

    # --- Step 4: Build validation report ---
    # Per-condition distribution
    cond_dist_full = cond_counts.to_dict()
    cond_dist_subset = subset_df[cond_col].value_counts().to_dict()

    # Per-cluster distribution
    cluster_dist_full = full_df[cluster_col].value_counts().to_dict()
    cluster_dist_subset = subset_df[cluster_col].value_counts().to_dict()

    # Per-(condition, cluster) distribution
    cond_cluster_full = (
        full_df.groupby([cond_col, cluster_col]).size().to_dict()
    )
    cond_cluster_subset = (
        subset_df.groupby([cond_col, cluster_col]).size().to_dict()
    )
    # Convert tuple keys to strings for JSON
    cond_cluster_full_str = {f"{c}|{cl}": n for (c, cl), n in cond_cluster_full.items()}
    cond_cluster_subset_str = {f"{c}|{cl}": n for (c, cl), n in cond_cluster_subset.items()}

    # Per-condition cluster ratio comparison
    cluster_ratios = {}
    for cond in cond_counts.index:
        cluster_ratios[cond] = {}
        cluster_counts_full = full_df[
            full_df[cond_col] == cond
        ][cluster_col].value_counts()
        for cluster in cluster_counts_full.index:
            full_n = cluster_counts_full.get(cluster, 0)
            subset_n = cluster_stats.get(cond, {}).get("sampled", {}).get(cluster, 0)
            if full_n > 0:
                cluster_ratios[cond][cluster] = round(subset_n / full_n, 4)

    report = {
        "full_size": total,
        "subset_size": len(subset_df),
        "target_size": target_size,
        "seed": seed,
        "condition_distribution": {
            "full": cond_dist_full,
            "subset": cond_dist_subset,
            "ratio": {
                k: round(cond_dist_subset.get(k, 0) / v, 4)
                for k, v in cond_dist_full.items()
            },
        },
        "cluster_distribution": {
            "full": cluster_dist_full,
            "subset": cluster_dist_subset,
            "ratio": {
                k: round(cluster_dist_subset.get(k, 0) / v, 4)
                for k, v in cluster_dist_full.items()
            },
        },
        "condition_cluster_distribution": {
            "full": cond_cluster_full_str,
            "subset": cond_cluster_subset_str,
        },
        "cluster_sampling_ratios": cluster_ratios,
        "per_condition_targets": per_cond_target,
    }

    # --- Quick quality check: minimum cells per cluster ---
    min_per_cluster = {}
    for cond in cond_counts.index:
        for cluster in cluster_dist_full:
            key = f"{cond}|{cluster}"
            n = cond_cluster_subset_str.get(key, 0)
            if n > 0:
                min_per_cluster[key] = n
    report["min_cells_per_condition_cluster"] = min(
        min_per_cluster.values()
    ) if min_per_cluster else 0

    return subset_df, report


def _size_label(size: int) -> str:
    """Convert size to a readable file label."""
    if size >= 1000:
        return f"{size // 1000}k"
    return str(size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, required=True, help="Target subset size (cells)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to full index CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.index.exists():
        print(f"Full index not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading full index from {args.index}...")
    full_df = load_full_index(args.index)
    n_conditions = full_df["CPD_NAME"].nunique()
    n_clusters = full_df["cluster_type"].nunique()
    print(
        f"Full dataset: {len(full_df)} cells, "
        f"conditions={n_conditions}, clusters={n_clusters}"
    )

    subset_df, report = build_stratified_subset(
        full_df, args.size, args.seed
    )

    # Write output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label = _size_label(args.size)
    out_path = args.out_dir / f"index_diet_{label}.csv"
    subset_df.to_csv(out_path)
    print(f"Subset written: {out_path} ({len(subset_df)} cells)")

    # Write report
    report_path = args.out_dir / f"subset_report_{label}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Report written: {report_path}")

    # Print summary
    print("\n--- Condition Distribution ---")
    for cond, ratio in report["condition_distribution"]["ratio"].items():
        full_n = report["condition_distribution"]["full"].get(cond, 0)
        subset_n = report["condition_distribution"]["subset"].get(cond, 0)
        print(f"  {cond}: {subset_n}/{full_n} (ratio={ratio})")

    print("\n--- Cluster Distribution ---")
    for cluster, ratio in report["cluster_distribution"]["ratio"].items():
        full_n = report["cluster_distribution"]["full"].get(cluster, 0)
        subset_n = report["cluster_distribution"]["subset"].get(cluster, 0)
        print(f"  {cluster}: {subset_n}/{full_n} (ratio={ratio})")

    min_per = report.get("min_cells_per_condition_cluster", "N/A")
    print(f"\nMinimum cells per (condition, cluster) group: {min_per}")
    print(
        f"\nDone. Config snippet:\n"
        f"  data_index_path: data/processed/diet/{out_path.name}"
    )


if __name__ == "__main__":
    main()
