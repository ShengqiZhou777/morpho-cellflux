#!/usr/bin/env python3
"""Build a tiny local microalgae fixture for smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "smoke" / "microalgae_v1"


def _write_image(path: Path, seed: int, treated: bool) -> None:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:128, 0:128]
    cx = 42 + (seed % 5) * 8
    cy = 48 + (seed % 4) * 7
    radius = 18 + (seed % 3) * 3
    cell = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2)))
    channel_shift = np.array([0.30, 0.52, 0.72], dtype=np.float32)
    if treated:
        channel_shift = np.array([0.70, 0.45, 0.35], dtype=np.float32)
        cell += 0.35 * np.exp(-(((x - 82) ** 2 + (y - 72) ** 2) / (2 * 14**2)))
    noise = rng.normal(0.0, 0.025, size=(128, 128, 3)).astype(np.float32)
    image = np.clip(cell[..., None] * channel_shift + noise, 0.0, 1.0)
    Image.fromarray((image * 255).astype(np.uint8), mode="RGB").save(path)


def build_fixture(root: Path = DEFAULT_ROOT) -> None:
    image_root = root / "images"
    view_root = root / "views" / "timepoint"
    image_root.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_id = 0
    for split, pairs in [("train", 4), ("test", 2)]:
        for pair_idx in range(pairs):
            batch = f"{split}_batch_{pair_idx % 2}"
            for annot, treated in [("negative_control", False), ("treated", True)]:
                filename = f"{split}_{pair_idx}_{annot}.png"
                _write_image(image_root / filename, seed=sample_id + 17, treated=treated)
                rows.append(
                    {
                        "SAMPLE_KEY": filename,
                        "SPLIT": split,
                        "ANNOT": annot,
                        "CPD_NAME": "smoke_timepoint_a" if treated else "control",
                        "BATCH": batch,
                    }
                )
                sample_id += 1

    index = pd.DataFrame(rows)
    index.to_csv(view_root / "index.csv")

    embedding = pd.DataFrame(
        [[0.15, 0.35, 0.55, 0.75]],
        index=["smoke_timepoint_a"],
        columns=["time_sin", "time_cos", "rna_pc1", "protein_pc1"],
    )
    embedding.to_csv(view_root / "embedding.csv")

    summary = {
        "dataset": "microalgae_smoke",
        "rows": len(index),
        "train_rows": int((index["SPLIT"] == "train").sum()),
        "test_rows": int((index["SPLIT"] == "test").sum()),
        "image_root": str(image_root.relative_to(ROOT)),
        "view_root": str(view_root.relative_to(ROOT)),
    }
    (view_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    build_fixture(args.root)
    print(f"Smoke fixture ready: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
