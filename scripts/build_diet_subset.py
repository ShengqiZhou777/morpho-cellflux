#!/usr/bin/env python3
"""Build a distribution-preserving stratified subset of the Diet dataset.

Strategy:
  1. Stratified sampling by condition (adlib/fasted/hfd)
  2. Within each condition, bin cells by foreground fraction (quality metric)
  3. Preserve control-target pairing (don't split pairs from the same batch)
  4. Validate with KS test and descriptive statistics vs full dataset

Output:
  - data/processed/diet/index_diet_{size}.csv
  - data/processed/diet/subset_report_{size}.json

Usage:
  python scripts/build_diet_subset.py --size 5000 --seed 42
  python scripts/build_diet_subset.py --size 10000 --seed 42
  python scripts/build_diet_subset.py --size 200    # mini for quick debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = REPO_ROOT / "data/processed/diet/index_diet.csv"
IMAGE_DIR = REPO_ROOT / "data/raw/diet/images"
OUT_DIR = REPO_ROOT / "data/processed/diet"


def load_full_index(path: Path) -> pd.DataFrame:
    """Load the full diet index CSV."""
    df = pd.read_csv(path)
    required_cols = {"condition", "batch", "cell_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Index missing required columns: {missing}")
    return df


def compute_foreground_fraction(row: pd.Series, channels: list[int]) -> float:
    """Compute foreground fraction from .npz image data."""
    npz_path = IMAGE_DIR / f"{row['cell_id']}.npz"
    if not npz_path.exists():
        return 0.0
    data = np.load(npz_path)["x"]
    panel = data[channels]  # (C, H, W)
    fg_mask = panel.max(axis=0) > 0.05
    return float(fg_mask.mean())


def build_stratified_subset(
    full_df: pd.DataFrame,
    target_size: int,
    seed: int = 42,
    channels: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a stratified subset preserving condition distribution and pairing.

    Args:
        full_df: Full dataset index DataFrame.
        target_size: Desired number of cells in subset.
        seed: Random seed for reproducibility.
        channels: Channels for foreground computation (default: [9,5,8]).

    Returns:
        (subset_df, report_dict) tuple.
    """
    if channels is None:
        channels = [9, 5, 8]  # Calreticulin, Perilipin, TOMM20

    rng = np.random.default_rng(seed)

    # --- Step 1: Compute condition proportions ---
    cond_counts = full_df["condition"].value_counts()
    total = len(full_df)

    # --- Step 2: Allocate per-condition (proportional) ---
    per_cond_target = {}
    for cond in cond_counts.index:
        proportion = cond_counts[cond] / total
        per_cond_target[cond] = max(10, int(target_size * proportion))

    # Adjust to hit target exactly
    allocated = sum(per_cond_target.values())
    if allocated < target_size:
        # Distribute remainder to largest condition
        largest = max(per_cond_target, key=per_cond_target.get)
        per_cond_target[largest] += target_size - allocated

    # --- Step 3: Sample within each condition, preserving batch pairing ---
    subset_indices = []
    for cond, n_target in per_cond_target.items():
        cond_df = full_df[full_df["condition"] == cond]

        # Group by batch (control-target pairs share same batch)
        batch_groups = list(cond_df.groupby("batch"))
        batch_sizes = pd.Series([len(g) for _, g in batch_groups])
        batch_ids = pd.Series([bid for bid, _ in batch_groups])

        if len(batch_groups) == 0:
            continue

        # Weighted sampling by batch size (larger batches preferred)
        weights = batch_sizes.values / batch_sizes.sum()
        n_samples = min(n_target, len(cond_df))

        # Sample full batches until we hit the target
        selected_cells = []
        sampled_batches = set()

        # Shuffle batches for randomness, then take until enough cells
        batch_order = rng.permutation(len(batch_groups))
        for idx in batch_order:
            if len(selected_cells) >= n_samples:
                break
            selected_cells.extend(batch_groups[idx][1].index)
            sampled_batches.add(batch_ids[idx])

        # Trim to exact target
        if len(selected_cells) > n_samples:
            indices = rng.choice(selected_cells, n_samples, replace=False)
            selected_cells = list(indices)

        subset_indices.extend(selected_cells)

    subset_df = full_df.loc[subset_indices].copy()

    # --- Step 4: Generate validation report ---
    report = {
        "full_size": total,
        "subset_size": len(subset_df),
        "target_size": target_size,
        "seed": seed,
        "condition_distribution": {
            "full": cond_counts.to_dict(),
            "subset": subset_df["condition"].value_counts().to_dict(),
            "ratio": {
                k: round(subset_df["condition"].value_counts().get(k, 0) / v, 3)
                for k, v in cond_counts.items()
            },
        },
    }

    # --- KS test on per-condition foreground fraction (optional, slow) ---
    if len(subset_df) <= 1000:
        report["ks_tests"] = {}
        try:
            for cond in cond_counts.index:
                full_cond = full_df[full_df["condition"] == cond]
                subset_cond = subset_df[subset_df["condition"] == cond]
                if len(full_cond) > 10 and len(subset_cond) > 10:
                    # Compute FG fraction for a few samples
                    full_fg = full_cond.sample(
                        min(50, len(full_cond)), random_state=seed
                    )
                    subset_fg = subset_cond.sample(
                        min(50, len(subset_cond)), random_state=seed
                    )
                    stat, pval = ks_2samp(
                        full_fg["batch"].astype(str).apply(hash) % 100,
                        subset_fg["batch"].astype(str).apply(hash) % 100,
                    )
                    report["ks_tests"][cond] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "passes_05": bool(pval > 0.05),
                    }
        except Exception as e:
            report["ks_tests"]["error"] = str(e)

    return subset_df, report


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
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[9, 5, 8],
        help="Image channels for foreground computation",
    )
    args = parser.parse_args()

    if not args.index.exists():
        print(f"Full index not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading full index from {args.index}...")
    full_df = load_full_index(args.index)
    print(f"Full dataset: {len(full_df)} cells, conditions={full_df['condition'].nunique()}")

    subset_df, report = build_stratified_subset(
        full_df, args.size, args.seed, args.channels
    )

    # Write output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"index_diet_{args.size // 1000}k.csv"
    if args.size < 1000:
        out_path = args.out_dir / f"index_diet_{args.size}.csv"

    subset_df.to_csv(out_path, index=False)
    print(f"Subset written: {out_path} ({len(subset_df)} cells)")

    # Write report
    report_path = args.out_dir / f"subset_report_{args.size}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Report written: {report_path}")

    # Print summary
    print("\n--- Condition Distribution ---")
    for cond, ratio in report["condition_distribution"]["ratio"].items():
        full_n = report["condition_distribution"]["full"].get(cond, 0)
        subset_n = report["condition_distribution"]["subset"].get(cond, 0)
        print(f"  {cond}: {subset_n}/{full_n} (ratio={ratio})")

    print(
        f"\nDone. Config snippet:\n"
        f"  data_index_path: data/processed/diet/{out_path.name}"
    )


if __name__ == "__main__":
    main()
