#!/usr/bin/env python3
"""Build field-level metadata from localized raw microalgae field images.

This script stays close to the actual acquisition unit of the dataset:
one microscopy field-of-view image plus its segmentation mask and
instance-level summary table.

Outputs under `data/processed/microalgae_v1/views/field/`:
1. `metadata.csv`  : EXIF + acquisition order + basic field counts
2. `summary.csv`   : field-level aggregated cell morphology statistics
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "microalgae_v1"
TIMECOURSE_ROOT = RAW_ROOT / "field_images"
OUT = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "field"

SUMMARY_FILE = "All_Cells_Summary_V3.csv"
SUMMARY_FALLBACKS = ("All_Cells_Summary_V3.csv", "All_Cells_Summary_V2.csv", "All_Cells_Summary.csv")

FIELD_FEATURES = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "solidity",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "mean_intensity",
    "std_intensity",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
]

TIME_BIN_MINUTES = 5


def _time_bin_minutes(actual_h: float) -> int:
    return int(round((float(actual_h) * 60.0) / TIME_BIN_MINUTES) * TIME_BIN_MINUTES)


def _read_exif(image_path: Path) -> dict[str, object]:
    img = Image.open(image_path)
    exif = {TAGS.get(k, str(k)): v for k, v in img.getexif().items()}
    return {
        "image_width": img.size[0],
        "image_height": img.size[1],
        "make": exif.get("Make"),
        "model": exif.get("Model"),
        "capture_datetime": exif.get("DateTime"),
        "x_resolution": exif.get("XResolution"),
        "y_resolution": exif.get("YResolution"),
        "resolution_unit": exif.get("ResolutionUnit"),
        "orientation": exif.get("Orientation"),
        "software": exif.get("Software"),
    }


def _load_field_summary(state_dir: Path) -> pd.DataFrame:
    for name in SUMMARY_FALLBACKS:
        path = state_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No field summary CSV found under {state_dir}")


def build_field_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for time_dir in sorted(TIMECOURSE_ROOT.iterdir(), key=lambda p: p.name):
        if not time_dir.is_dir():
            continue
        time_label = time_dir.name
        nominal_time_h = int(time_label[:-1]) if time_label.endswith("h") else float(time_label)

        for cond_dir in sorted(time_dir.iterdir(), key=lambda p: p.name):
            if not cond_dir.is_dir():
                continue

            image_dir = cond_dir / "images"
            mask_dir = cond_dir / "masks"
            summary_df = _load_field_summary(cond_dir).copy()
            summary_df["field_id"] = summary_df["file"].astype(str).str.replace("_mask.png", "", regex=False)

            grouped = summary_df.groupby("field_id", sort=True)
            for image_path in sorted(image_dir.glob("*.jpg")):
                field_id = image_path.stem
                mask_path = mask_dir / f"{field_id}_mask.png"
                exif = _read_exif(image_path)

                if field_id in grouped.groups:
                    field_cells = grouped.get_group(field_id)
                    n_cells = int(len(field_cells))
                    min_label = int(field_cells["label"].min())
                    max_label = int(field_cells["label"].max())
                else:
                    field_cells = pd.DataFrame(columns=summary_df.columns)
                    n_cells = 0
                    min_label = None
                    max_label = None

                metadata_rows.append(
                    {
                        "field_id": field_id,
                        "condition": cond_dir.name,
                        "time_label": time_label,
                        "nominal_time_h": nominal_time_h,
                        "image_relpath": str(image_path.relative_to(TIMECOURSE_ROOT)),
                        "mask_relpath": str(mask_path.relative_to(TIMECOURSE_ROOT)) if mask_path.exists() else None,
                        "summary_file": SUMMARY_FILE if (cond_dir / SUMMARY_FILE).exists() else None,
                        "n_cells": n_cells,
                        "min_instance_label": min_label,
                        "max_instance_label": max_label,
                        **exif,
                    }
                )

                if n_cells == 0:
                    continue

                agg = {"field_id": field_id, "condition": cond_dir.name, "time_label": time_label, "nominal_time_h": nominal_time_h}
                for feature in FIELD_FEATURES:
                    agg[f"{feature}_mean"] = float(field_cells[feature].mean())
                    agg[f"{feature}_std"] = float(field_cells[feature].std(ddof=0))
                summary_rows.append(agg)

    metadata = pd.DataFrame(metadata_rows).sort_values(["condition", "nominal_time_h", "field_id"], kind="stable")
    metadata["capture_datetime"] = pd.to_datetime(metadata["capture_datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
    global_start = metadata["capture_datetime"].min()
    metadata["actual_time_h"] = ((metadata["capture_datetime"] - global_start).dt.total_seconds() / 3600.0).astype(float)
    metadata["time_bin_min"] = metadata["actual_time_h"].apply(_time_bin_minutes).astype(int)
    metadata["time_bin_h"] = (metadata["time_bin_min"] / 60.0).astype(float)
    metadata["capture_date"] = metadata["capture_datetime"].dt.date.astype("string")
    metadata["capture_time"] = metadata["capture_datetime"].dt.time.astype("string")
    metadata["capture_order_in_state"] = (
        metadata.sort_values(["condition", "nominal_time_h", "capture_datetime", "field_id"], kind="stable")
        .groupby(["condition", "nominal_time_h"])
        .cumcount()
        + 1
    )
    metadata["seconds_from_state_start"] = (
        metadata["capture_datetime"]
        - metadata.groupby(["condition", "nominal_time_h"])["capture_datetime"].transform("min")
    ).dt.total_seconds()
    metadata["capture_datetime"] = metadata["capture_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    field_summary = pd.DataFrame(summary_rows).sort_values(["condition", "nominal_time_h", "field_id"], kind="stable")
    field_summary = field_summary.merge(
        metadata[
            [
                "field_id",
                "condition",
                "nominal_time_h",
                "capture_datetime",
                "capture_order_in_state",
                "seconds_from_state_start",
                "actual_time_h",
                "time_bin_min",
                "time_bin_h",
                "n_cells",
            ]
        ],
        on=["field_id", "condition", "nominal_time_h"],
        how="left",
    )
    return metadata.reset_index(drop=True), field_summary.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata, field_summary = build_field_frames()

    metadata_path = OUT / "metadata.csv"
    summary_path = OUT / "summary.csv"
    metadata.to_csv(metadata_path, index=False)
    field_summary.to_csv(summary_path, index=False)

    print(f"wrote {metadata_path} with shape {metadata.shape}")
    print(f"wrote {summary_path} with shape {field_summary.shape}")
    print(metadata.head(5).to_string())


if __name__ == "__main__":
    main()
