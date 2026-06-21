"""Shared helpers for external baseline adapters.

External methods differ in how they train and generate images, but their final
outputs must match the CellFlux eval contract consumed by
`scripts/aggregate_eval.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
# Legacy fallback only; public configs should define `channels`.
DEFAULT_CHANNELS = [5, 9, 10]


def repo_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | os.PathLike[str]) -> dict:
    with repo_path(path).open() as f:
        return yaml.safe_load(f)


def read_index(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_csv(repo_path(path), dtype={"SAMPLE_KEY": str})
    required = {"SAMPLE_KEY", "CPD_NAME", "ANNOT", "BATCH", "SPLIT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["SAMPLE_KEY"] = df["SAMPLE_KEY"].astype(str)
    return df


def channels_from_config(config: dict) -> list[int]:
    return [int(c) for c in config.get("channels", DEFAULT_CHANNELS)]


def treated_rows(df: pd.DataFrame, split: str, max_samples: int | None = None) -> pd.DataFrame:
    out = df[(df["SPLIT"] == split) & (df["ANNOT"] != "negative_control")].copy()
    if max_samples is not None:
        out = out.head(max_samples).copy()
    if out.empty:
        raise ValueError(f"No treated rows found for split={split!r}")
    return out


def control_rows(df: pd.DataFrame, split: str, fallback_all_splits: bool = True) -> pd.DataFrame:
    ctrl = df[(df["SPLIT"] == split) & (df["ANNOT"] == "negative_control")].copy()
    if ctrl.empty and fallback_all_splits:
        ctrl = df[df["ANNOT"] == "negative_control"].copy()
    if ctrl.empty:
        raise ValueError("No negative_control rows found")
    return ctrl


def choose_control(target: pd.Series, controls: pd.DataFrame, rng: np.random.Generator) -> str:
    same_batch = controls[controls["BATCH"] == target["BATCH"]]
    pool = same_batch if not same_batch.empty else controls
    return str(pool.iloc[int(rng.integers(0, len(pool)))]["SAMPLE_KEY"])


def build_pairs(
    df: pd.DataFrame,
    split: str,
    seed: int,
    max_samples: int | None = None,
    fallback_all_splits: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    targets = treated_rows(df, split, max_samples)
    controls = control_rows(df, split, fallback_all_splits=fallback_all_splits)
    rng = np.random.default_rng(seed)
    pairs = {
        str(target["SAMPLE_KEY"]): choose_control(target, controls, rng)
        for _, target in targets.iterrows()
    }
    return targets, pairs


def panel_array(image_dir: str | os.PathLike[str], sample_key: str, channels: Iterable[int]) -> np.ndarray:
    npz_path = repo_path(image_dir) / f"{sample_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    arr = np.load(npz_path)["x"][list(channels)]
    arr = np.moveaxis(arr, 0, -1)
    return np.clip(arr, 0.0, 1.0)


def panel_png(image_dir: str | os.PathLike[str], sample_key: str, channels: Iterable[int]) -> Image.Image:
    arr = panel_array(image_dir, sample_key, channels)
    return Image.fromarray((arr * 255).round().astype(np.uint8), mode="RGB")


def image_to_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def write_fid_image(output_dir: str | os.PathLike[str], condition: str, target_id: str, image: Image.Image | np.ndarray) -> None:
    out = repo_path(output_dir) / "fid_samples" / "epoch-0" / str(condition)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_to_uint8(image), mode="RGB").save(out / f"{target_id}.png")


def write_eval_contract(
    output_dir: str | os.PathLike[str],
    trt2ctrl: dict[str, str],
    args: dict,
) -> None:
    out = repo_path(output_dir)
    epoch_dir = out / "fid_samples" / "epoch-0"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.dumps(trt2ctrl, indent=2) + "\n"
    (out / "args.json").write_text(json.dumps(args, indent=2) + "\n")
    (out / "fid_samples" / "trt2ctrl_idx.json").write_text(mapping)
    (epoch_dir / "trt2ctrl_idx.json").write_text(mapping)
