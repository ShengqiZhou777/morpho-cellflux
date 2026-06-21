"""Copy-control baseline for Morpho-CellFlux method comparisons.

This baseline returns a same-batch control image as the generated image for each
treated target cell. It is the biological null used by the gap-closure metric:
`gap_closed = 0` means "no better than copying a control cell".

The output layout intentionally matches CellFlux eval runs so
`scripts/aggregate_eval.py` can evaluate it without baseline-specific logic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _read_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"SAMPLE_KEY": str})
    required = {"SAMPLE_KEY", "CPD_NAME", "ANNOT", "BATCH", "SPLIT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["SAMPLE_KEY"] = df["SAMPLE_KEY"].astype(str)
    return df


def _panel_png(image_dir: Path, sample_key: str, channels: list[int]) -> Image.Image:
    npz_path = image_dir / f"{sample_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    arr = np.load(npz_path)["x"][channels]
    arr = np.moveaxis(arr, 0, -1)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _condition_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = df[(df["SPLIT"] == split) & (df["ANNOT"] != "negative_control")].copy()
    if rows.empty:
        raise ValueError(f"No treated rows found for split={split!r}")
    return rows


def _control_pool(df: pd.DataFrame, split: str, fallback_all_splits: bool) -> pd.DataFrame:
    ctrl = df[(df["SPLIT"] == split) & (df["ANNOT"] == "negative_control")].copy()
    if ctrl.empty and fallback_all_splits:
        ctrl = df[df["ANNOT"] == "negative_control"].copy()
    if ctrl.empty:
        raise ValueError("No negative_control rows found for copy-control baseline")
    return ctrl


def _choose_control(
    target: pd.Series,
    controls: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
    same_batch = controls[controls["BATCH"] == target["BATCH"]]
    pool = same_batch if not same_batch.empty else controls
    idx = int(rng.integers(0, len(pool)))
    return str(pool.iloc[idx]["SAMPLE_KEY"])


def build_copy_control(
    config: dict,
    output_dir: Path,
    split: str,
    seed: int,
    max_samples: int | None,
    fallback_all_splits: bool,
) -> None:
    channels = [int(c) for c in config.get("channels", [5, 9, 10])]
    image_dir = _repo_path(config["image_path"])
    index_path = _repo_path(config["data_index_path"])
    embedding_path = _repo_path(config["embedding_path"])

    df = _read_index(index_path)
    targets = _condition_rows(df, split)
    if max_samples is not None:
        targets = targets.head(max_samples)
    controls = _control_pool(df, split, fallback_all_splits)
    rng = np.random.default_rng(seed)

    epoch_dir = output_dir / "fid_samples" / "epoch-0"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    trt2ctrl: dict[str, str] = {}
    written = 0
    skipped = 0
    for _, target in targets.iterrows():
        target_id = str(target["SAMPLE_KEY"])
        condition = str(target["CPD_NAME"])
        control_id = _choose_control(target, controls, rng)
        try:
            image = _panel_png(image_dir, control_id, channels)
        except FileNotFoundError:
            skipped += 1
            continue

        condition_dir = epoch_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        image.save(condition_dir / f"{target_id}.png")
        trt2ctrl[target_id] = control_id
        written += 1

    args = {
        "baseline_method": "copy_control",
        "use_initial": 1,
        "channels": channels,
        "image_path": str(config["image_path"]),
        "data_index_path": str(config["data_index_path"]),
        "embedding_path": str(config["embedding_path"]),
        "dataset_name": config.get("dataset_name", "perturbmulti"),
        "task_name": config.get("task_name"),
        "split": split,
        "seed": seed,
        "max_samples": max_samples,
        "written": written,
        "skipped_missing_controls": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(args, indent=2) + "\n")
    mapping = json.dumps(trt2ctrl, indent=2) + "\n"
    (output_dir / "fid_samples" / "trt2ctrl_idx.json").write_text(mapping)
    (epoch_dir / "trt2ctrl_idx.json").write_text(mapping)

    print(f"copy-control wrote {written} generated samples to {epoch_dir}")
    if skipped:
        print(f"skipped {skipped} targets because the sampled control npz was missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Morpho-CellFlux YAML config")
    parser.add_argument("--output", required=True, help="Output run directory")
    parser.add_argument("--split", default="test", help="Index split to generate/evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--strict-split-controls",
        action="store_true",
        help="Use controls only from --split; by default falls back to all splits if needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(_repo_path(args.config))
    build_copy_control(
        config=config,
        output_dir=_repo_path(args.output),
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        fallback_all_splits=not args.strict_split_controls,
    )


if __name__ == "__main__":
    main()
