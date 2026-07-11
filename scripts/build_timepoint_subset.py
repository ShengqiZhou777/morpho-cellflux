#!/usr/bin/env python3
"""Build a balanced quick-training subset for microalgae timepoint data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "timepoint"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "timepoint_quick"


def _sample_group(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(group) <= n:
        return group
    return group.sample(n=n, random_state=seed)


def _sample_by_label(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [_sample_group(group, cap, seed) for _, group in df.groupby("CPD_NAME", sort=True)],
        ignore_index=True,
    )


def build_subset(
    source_view: Path,
    output_view: Path,
    train_per_label: int,
    test_per_label: int,
    seed: int,
) -> dict[str, object]:
    index_path = source_view / "index.csv"
    embedding_path = source_view / "embedding.csv"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)

    idx = pd.read_csv(index_path, index_col=0)
    if not {"SPLIT", "ANNOT", "CPD_NAME"}.issubset(idx.columns):
        raise ValueError("timepoint index must contain SPLIT, ANNOT, and CPD_NAME")

    pieces: list[pd.DataFrame] = []
    caps = {"train": train_per_label, "test": test_per_label}
    for split, cap in caps.items():
        split_df = idx[idx["SPLIT"] == split]
        for annot in ("treated", "negative_control"):
            annot_df = split_df[split_df["ANNOT"] == annot]
            pieces.append(_sample_by_label(annot_df, cap, seed))

    subset = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["SPLIT", "BATCH", "ANNOT", "CPD_NAME", "SAMPLE_KEY"], kind="stable")
        .reset_index(drop=True)
    )
    labels = sorted(subset.loc[subset["ANNOT"] == "treated", "CPD_NAME"].unique())

    emb = pd.read_csv(embedding_path, index_col=0)
    missing = sorted(set(labels) - set(emb.index))
    if missing:
        raise ValueError(f"Subset labels missing from embedding.csv: {missing[:10]}")
    emb = emb.loc[labels]

    output_view.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_view / "index.csv")
    emb.to_csv(output_view / "embedding.csv")

    train_treated = subset[(subset["SPLIT"] == "train") & (subset["ANNOT"] == "treated")]
    test_treated = subset[(subset["SPLIT"] == "test") & (subset["ANNOT"] == "treated")]
    train_counts = train_treated.groupby("CPD_NAME").size()
    test_counts = test_treated.groupby("CPD_NAME").size()
    summary = {
        "source_view": str(source_view.relative_to(REPO_ROOT)),
        "index_rows": int(len(subset)),
        "train_rows": int((subset["SPLIT"] == "train").sum()),
        "test_rows": int((subset["SPLIT"] == "test").sum()),
        "train_treated_rows": int(len(train_treated)),
        "test_treated_rows": int(len(test_treated)),
        "n_labels": int(len(labels)),
        "embedding_dim": int(emb.shape[1]),
        "train_per_label_cap": int(train_per_label),
        "test_per_label_cap": int(test_per_label),
        "seed": int(seed),
        "train_treated_per_label_min": int(train_counts.min()),
        "train_treated_per_label_median": float(train_counts.median()),
        "train_treated_per_label_max": int(train_counts.max()),
        "test_treated_per_label_min": int(test_counts.min()),
        "test_treated_per_label_median": float(test_counts.median()),
        "test_treated_per_label_max": int(test_counts.max()),
    }
    with (output_view / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-view", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-per-label", type=int, default=256)
    parser.add_argument("--test-per-label", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_subset(
        source_view=args.source_view,
        output_view=args.output_view,
        train_per_label=args.train_per_label,
        test_per_label=args.test_per_label,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
