"""Export Morpho-CellFlux data into common external-baseline formats.

This script is the first adapter layer for external baselines. It reads the same
YAML config used by Morpho-CellFlux and exports the selected 3-channel panel as:

1. PNG image folders for PhenDiff / StarGAN / generic image-to-image baselines.
2. IMPA-style HWC `.npy` files plus an index CSV with `STATE` labels.

The export preserves the original sample IDs and writes condition mappings so
generated outputs can later be converted back into the shared `fid_samples`
layout used by `scripts/aggregate_eval.py`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
# Legacy fallback only; public configs should define `channels`.
DEFAULT_CHANNELS = [5, 9, 10]
_WORKER_IMAGE_DIR: Path | None = None
_WORKER_OUTPUT_DIR: Path | None = None
_WORKER_CHANNELS: list[int] | None = None
_WORKER_COND2ID: dict[str, int] | None = None
_WORKER_SKIP_EXISTING = True


def repo_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def panel_array(image_dir: Path, sample_key: str, channels: list[int]) -> np.ndarray:
    npz_path = image_dir / f"{sample_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    arr = np.load(npz_path)["x"][channels]
    arr = np.moveaxis(arr, 0, -1)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def condition_order(df: pd.DataFrame) -> list[str]:
    controls = sorted(df.loc[df["ANNOT"] == "negative_control", "CPD_NAME"].astype(str).unique())
    treated = sorted(df.loc[df["ANNOT"] != "negative_control", "CPD_NAME"].astype(str).unique())
    return controls + [c for c in treated if c not in controls]


def safe_condition(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def impa_sample_key(sample_key: str) -> str:
    # IMPA's bbbc021 loader parses SAMPLE_KEY as p/w/file.npy.
    return f"p0_w0_{sample_key}"


def write_imagefolder_record(
    arr: np.ndarray,
    row: pd.Series,
    out_root: Path,
    split: str,
    condition: str,
) -> str:
    rel = Path("imagefolder") / split / safe_condition(condition) / f"{row.SAMPLE_KEY}.png"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return str(rel)


def write_impa_record(arr: np.ndarray, row: pd.Series, out_root: Path) -> str:
    rel = Path("impa_npy") / "images" / "p0" / "w0" / f"{row.SAMPLE_KEY}.npy"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return str(rel)


def imagefolder_rel(split: str, condition: str, sample_key: str) -> Path:
    return Path("imagefolder") / split / safe_condition(condition) / f"{sample_key}.png"


def impa_rel(sample_key: str) -> Path:
    return Path("impa_npy") / "images" / "p0" / "w0" / f"{sample_key}.npy"


def complete_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_records(row: dict, png_rel: Path, npy_rel: Path, cond2id: dict[str, int]) -> tuple[dict, dict]:
    sample_key = str(row["SAMPLE_KEY"])
    condition = str(row["CPD_NAME"])
    annot = str(row["ANNOT"])
    state = "control" if annot == "negative_control" else "trt"
    batch = str(row["BATCH"])
    rec = {
        "sample_key": sample_key,
        "split": str(row["SPLIT"]),
        "condition": condition,
        "condition_id": cond2id[condition],
        "state": state,
        "annot": annot,
        "batch": int(batch) if batch.isdigit() else batch,
        "png": str(png_rel),
        "npy": str(npy_rel),
    }

    # IMPA-compatible CSV. Keep original columns and add STATE + parser-safe key.
    impa_rec = dict(row)
    impa_rec["ORIG_SAMPLE_KEY"] = sample_key
    impa_rec["SAMPLE_KEY"] = impa_sample_key(sample_key)
    impa_rec["STATE"] = state
    impa_rec["BATCH"] = batch
    impa_rec["CPD_NAME"] = condition
    if "DOSE" not in impa_rec:
        impa_rec["DOSE"] = 1.0
    return rec, impa_rec


def init_worker(
    image_dir: str,
    output_dir: str,
    channels: list[int],
    cond2id: dict[str, int],
    skip_existing: bool,
) -> None:
    global _WORKER_IMAGE_DIR, _WORKER_OUTPUT_DIR, _WORKER_CHANNELS, _WORKER_COND2ID, _WORKER_SKIP_EXISTING
    _WORKER_IMAGE_DIR = Path(image_dir)
    _WORKER_OUTPUT_DIR = Path(output_dir)
    _WORKER_CHANNELS = channels
    _WORKER_COND2ID = cond2id
    _WORKER_SKIP_EXISTING = skip_existing


def export_record_worker(row: dict) -> tuple[dict | None, dict | None, str | None]:
    if _WORKER_IMAGE_DIR is None or _WORKER_OUTPUT_DIR is None or _WORKER_CHANNELS is None or _WORKER_COND2ID is None:
        raise RuntimeError("export worker was not initialized")

    sample_key = str(row["SAMPLE_KEY"])
    split = str(row["SPLIT"])
    condition = str(row["CPD_NAME"])
    png_rel = imagefolder_rel(split, condition, sample_key)
    npy_rel = impa_rel(sample_key)
    png_path = _WORKER_OUTPUT_DIR / png_rel
    npy_path = _WORKER_OUTPUT_DIR / npy_rel
    rec, impa_rec = build_records(row, png_rel, npy_rel, _WORKER_COND2ID)

    if _WORKER_SKIP_EXISTING and complete_file(png_path) and complete_file(npy_path):
        return rec, impa_rec, None

    try:
        arr = panel_array(_WORKER_IMAGE_DIR, sample_key, _WORKER_CHANNELS)
    except FileNotFoundError as exc:
        return None, None, str(exc)

    if not _WORKER_SKIP_EXISTING or not complete_file(png_path):
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(png_path)
    if not _WORKER_SKIP_EXISTING or not complete_file(npy_path):
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, arr)
    return rec, impa_rec, None


def write_phendiff_adapter_files(output_dir: Path, conds: list[str], channels: list[int]) -> None:
    class_to_idx = {safe_condition(cond): i for i, cond in enumerate(sorted(safe_condition(c) for c in conds))}
    (output_dir / "phendiff_class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n")
    denoiser = {
        "_class_name": "CondUNet2DModel",
        "_diffusers_version": "0.18.2",
        "act_fn": "silu",
        "attention_head_dim": 8,
        "block_out_channels": [64, 128, 256],
        "center_input_sample": False,
        "class_embed_type": None,
        "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
        "downsample_padding": 1,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": len(channels),
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "norm_eps": 1e-5,
        "norm_num_groups": 32,
        "num_class_embeds": len(class_to_idx),
        "out_channels": len(channels),
        "resnet_time_scale_shift": "default",
        "sample_size": 128,
        "time_embedding_type": "positional",
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    }
    (output_dir / "phendiff_denoiser_config.json").write_text(json.dumps(denoiser, indent=2) + "\n")


def write_impa_embedding(config_embedding_path: Path, output_dir: Path) -> str:
    dst = output_dir / "impa_embedding.csv"
    shutil.copyfile(config_embedding_path, dst)
    return str(dst.relative_to(output_dir))


def export_dataset(
    config_path: Path,
    output_dir: Path,
    benchmark: str,
    splits: set[str],
    max_rows: int | None,
    workers: int,
    skip_existing: bool,
    progress_every: int,
) -> None:
    cfg = load_config(config_path)
    channels = [int(c) for c in cfg.get("channels", DEFAULT_CHANNELS)]
    image_dir = repo_path(cfg["image_path"])
    index_path = repo_path(cfg["data_index_path"])
    embedding_path = repo_path(cfg["embedding_path"])

    df = pd.read_csv(index_path, index_col=0, dtype={"SAMPLE_KEY": str})
    df["SAMPLE_KEY"] = df["SAMPLE_KEY"].astype(str)
    if splits:
        df = df[df["SPLIT"].isin(splits)].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        raise ValueError("No rows selected for export")

    output_dir.mkdir(parents=True, exist_ok=True)
    conds = condition_order(df)
    cond2id = {cond: i for i, cond in enumerate(conds)}

    metadata = []
    impa_rows = []
    missing = []
    total = len(df)
    row_dicts = (row._asdict() for row in df.itertuples(index=False))
    start = time.monotonic()

    def consume(i: int, result: tuple[dict | None, dict | None, str | None]) -> None:
        rec, impa_rec, missing_file = result
        if missing_file is not None:
            missing.append(missing_file)
        else:
            metadata.append(rec)
            impa_rows.append(impa_rec)
        if progress_every > 0 and (i % progress_every == 0 or i == total):
            elapsed = max(time.monotonic() - start, 1e-6)
            rate = i / elapsed
            print(
                f"[export {benchmark}] {i}/{total} rows, "
                f"exported={len(metadata)}, missing={len(missing)}, "
                f"rate={rate:.1f} rows/s",
                flush=True,
            )

    print(
        f"[export {benchmark}] rows={total} workers={workers} "
        f"skip_existing={int(skip_existing)} output={output_dir}",
        flush=True,
    )
    if workers <= 1:
        init_worker(str(image_dir), str(output_dir), channels, cond2id, skip_existing)
        for i, row in enumerate(row_dicts, start=1):
            consume(i, export_record_worker(row))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(str(image_dir), str(output_dir), channels, cond2id, skip_existing),
        ) as executor:
            for i, result in enumerate(executor.map(export_record_worker, row_dicts, chunksize=128), start=1):
                consume(i, result)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(impa_rows).to_csv(output_dir / "impa_index.csv")
    (output_dir / "conditions.json").write_text(json.dumps(cond2id, indent=2) + "\n")
    (output_dir / "missing_files.json").write_text(json.dumps(missing, indent=2) + "\n")
    impa_embedding_rel = write_impa_embedding(embedding_path, output_dir)
    write_phendiff_adapter_files(output_dir, conds, channels)

    # HuggingFace imagefolder-compatible metadata per split.
    for split, split_df in meta_df.groupby("split"):
        split_root = output_dir / "imagefolder" / split
        records = []
        for rec in split_df.to_dict("records"):
            file_name = str(Path(rec["png"]).relative_to(Path("imagefolder") / split))
            records.append({
                "file_name": file_name,
                "text": rec["condition"],
                "label": rec["condition_id"],
                "sample_key": rec["sample_key"],
                "state": rec["state"],
            })
        with (split_root / "metadata.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    manifest = {
        "benchmark": benchmark,
        "config_path": str(config_path),
        "image_path": cfg["image_path"],
        "data_index_path": cfg["data_index_path"],
        "embedding_path": cfg["embedding_path"],
        "channels": channels,
        "rows_exported": len(metadata),
        "missing_files": len(missing),
        "conditions": cond2id,
        "formats": {
            "imagefolder": "imagefolder/<split>/<condition>/<sample_key>.png",
            "impa_npy": "impa_npy/images/p0/w0/<sample_key>.npy + impa_index.csv",
            "impa_embedding": impa_embedding_rel,
            "phendiff_class_to_idx": "phendiff_class_to_idx.json",
            "phendiff_denoiser_config": "phendiff_denoiser_config.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("EXPORT_WORKERS", "1")))
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {s for s in args.splits.split(",") if s}
    export_dataset(
        config_path=repo_path(args.config),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        splits=splits,
        max_rows=args.max_rows,
        workers=max(args.workers, 1),
        skip_existing=not args.overwrite_existing,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
