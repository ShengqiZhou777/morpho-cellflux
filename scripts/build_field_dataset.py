#!/usr/bin/env python3
"""Build field-level microalgae generation artifacts from localized raw images.

This keeps the training task aligned with the actual acquisition unit:
one raw microscopy field-of-view image conditioned on acquisition timing and
field-level morphology summary.

Outputs under `data/processed/microalgae_v1/views/field/`:
1. `index.csv`      : ctrl/trt-compatible pair index for the existing loader
2. `embedding.csv`  : one embedding row per target field
3. `targets.csv`    : target-field metadata and derived coarse labels
4. `prompts.csv`    : prompt strings for later text-conditioning work
5. `summary.json`   : build summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_field_metadata import build_field_frames


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "microalgae_v1"
OUT = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "field"

FIELD_METADATA_PATH = OUT / "metadata.csv"
FIELD_SUMMARY_PATH = OUT / "summary.csv"

TEST_MODULUS = 7
TEST_RESIDUE = 0

FIELD_SUMMARY_FEATURES = [
    "area_mean",
    "area_std",
    "perimeter_mean",
    "perimeter_std",
    "circularity_mean",
    "circularity_std",
    "aspect_ratio_mean",
    "aspect_ratio_std",
    "solidity_mean",
    "solidity_std",
    "major_axis_mean",
    "major_axis_std",
    "minor_axis_mean",
    "minor_axis_std",
    "eccentricity_mean",
    "eccentricity_std",
    "mean_intensity_mean",
    "mean_intensity_std",
    "std_intensity_mean",
    "std_intensity_std",
    "texture_contrast_mean",
    "texture_contrast_std",
    "texture_homogeneity_mean",
    "texture_homogeneity_std",
    "texture_energy_mean",
    "texture_energy_std",
    "texture_correlation_mean",
    "texture_correlation_std",
]


def _state_key(condition: str, time_h: float | int) -> str:
    return f"{condition.lower()}_{int(float(time_h))}h"


def _field_key(condition: str, time_h: float | int, field_id: str) -> str:
    return f"{condition.lower()}_{int(float(time_h))}h_{field_id}"


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - float(series.mean())) / std).astype(np.float32)


def _compute_tertiles(series: pd.Series) -> tuple[float, float]:
    q1, q2 = series.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    if q1 == q2:
        q1 = float(series.median())
        q2 = q1
    return float(q1), float(q2)


def _categorize(series: pd.Series, thresholds: tuple[float, float], labels: tuple[str, str, str]) -> pd.Series:
    lo, hi = thresholds
    return pd.Series(
        np.where(series <= lo, labels[0], np.where(series <= hi, labels[1], labels[2])),
        index=series.index,
    )


def _load_field_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if FIELD_METADATA_PATH.exists() and FIELD_SUMMARY_PATH.exists():
        metadata = pd.read_csv(FIELD_METADATA_PATH)
        summary = pd.read_csv(FIELD_SUMMARY_PATH)
        required = {"actual_time_h", "time_bin_min", "time_bin_h"}
        if not required.issubset(metadata.columns):
            metadata, summary = build_field_frames()
    else:
        metadata, summary = build_field_frames()
    return metadata.copy(), summary.copy()


def _prepare_fields(metadata: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    merged = summary.merge(
        metadata[
            [
                "field_id",
                "condition",
                "nominal_time_h",
                "image_relpath",
                "mask_relpath",
                "image_width",
                "image_height",
                "max_instance_label",
                "capture_datetime",
                "capture_order_in_state",
                "seconds_from_state_start",
                "actual_time_h",
                "time_bin_min",
                "time_bin_h",
            ]
        ],
        on=[
            "field_id",
            "condition",
            "nominal_time_h",
            "capture_datetime",
            "capture_order_in_state",
            "seconds_from_state_start",
            "actual_time_h",
            "time_bin_min",
            "time_bin_h",
        ],
        how="left",
        validate="1:1",
    )
    merged = merged.loc[merged["image_relpath"].notna()].copy()
    merged["field_id"] = merged["field_id"].astype(str)
    merged["state_id"] = merged.apply(lambda row: _state_key(row["condition"], row["nominal_time_h"]), axis=1)
    merged["field_key"] = merged.apply(
        lambda row: _field_key(row["condition"], row["nominal_time_h"], row["field_id"]),
        axis=1,
    )
    merged["sample_key"] = merged["image_relpath"]
    merged["condition_id"] = merged["condition"].map({"Dark": 0, "Light": 1}).astype(int)
    max_actual_time_h = float(merged["actual_time_h"].max())
    if max_actual_time_h <= 0:
        raise ValueError("Field metadata actual_time_h must span a positive duration")
    merged["cond_light"] = (merged["condition"] == "Light").astype(np.float32)
    merged["cond_dark"] = (merged["condition"] == "Dark").astype(np.float32)
    merged["time_norm"] = (merged["actual_time_h"] / max_actual_time_h).astype(np.float32)

    merged = merged.sort_values(
        ["condition", "nominal_time_h", "capture_order_in_state", "field_id"],
        kind="stable",
    ).reset_index(drop=True)
    merged["state_rank"] = merged.groupby("state_id").cumcount()
    merged["SPLIT"] = np.where((merged["state_rank"] % TEST_MODULUS) == TEST_RESIDUE, "test", "train")

    state_sizes = merged.groupby("state_id")["capture_order_in_state"].transform("max").clip(lower=1)
    state_jitter = merged.groupby("state_id")["seconds_from_state_start"].transform("max").fillna(0.0).clip(lower=1.0)
    merged["capture_order_norm"] = ((merged["capture_order_in_state"] - 1) / state_sizes).astype(np.float32)
    merged["capture_time_norm"] = (merged["seconds_from_state_start"].fillna(0.0) / state_jitter).astype(np.float32)
    merged["n_cells_z"] = _safe_zscore(merged["n_cells"])

    for col in FIELD_SUMMARY_FEATURES:
        merged[f"{col}_z"] = _safe_zscore(merged[col])

    return merged


def _assign_sources(fields: pd.DataFrame) -> pd.DataFrame:
    fields = fields.copy()
    source_rows: list[dict[str, object]] = []
    source_pools = {
        condition: pool.sort_values(["capture_order_in_state", "field_id"], kind="stable").reset_index(drop=True)
        for condition, pool in fields.loc[fields["nominal_time_h"] == 0].groupby("condition")
    }
    if set(source_pools) != {"Dark", "Light"}:
        raise ValueError("Expected both Dark and Light 0h source pools to be present")

    for row in fields.itertuples(index=False):
        pool = source_pools[row.condition]
        source_idx = int((int(row.capture_order_in_state) - 1) % len(pool))
        source = pool.iloc[source_idx]
        if row.nominal_time_h == 0 and source["field_key"] == row.field_key and len(pool) > 1:
            source = pool.iloc[(source_idx + 1) % len(pool)]
        source_rows.append(
            {
                "target_field_key": row.field_key,
                "source_field_key": source["field_key"],
                "source_field_id": source["field_id"],
                "source_sample_key": source["sample_key"],
                "source_capture_order_in_state": int(source["capture_order_in_state"]),
                "source_seconds_from_state_start": float(source["seconds_from_state_start"]),
                "source_actual_time_h": float(source["actual_time_h"]),
                "source_time_bin_min": int(source["time_bin_min"]),
                "source_time_bin_h": float(source["time_bin_h"]),
            }
        )

    return fields.merge(pd.DataFrame(source_rows), left_on="field_key", right_on="target_field_key", how="left", validate="1:1")


def _build_embeddings(targets: pd.DataFrame) -> pd.DataFrame:
    merged = targets.copy()
    feature_cols = ["cond_light", "cond_dark", "time_norm", "capture_order_norm", "capture_time_norm", "n_cells_z"]
    feature_cols += [f"{col}_z" for col in FIELD_SUMMARY_FEATURES]

    embedding = merged.set_index("field_key")[feature_cols].sort_index()
    return embedding.astype(np.float32)


def _build_labels_and_prompts(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    density_thr = _compute_tertiles(targets["n_cells"])
    size_thr = _compute_tertiles(targets["area_mean"])
    brightness_thr = _compute_tertiles(targets["mean_intensity_mean"])
    texture_thr = _compute_tertiles(targets["texture_contrast_mean"])

    labeled = targets.copy()
    labeled["density_label"] = _categorize(targets["n_cells"], density_thr, ("sparse", "balanced", "dense"))
    labeled["size_label"] = _categorize(targets["area_mean"], size_thr, ("small_cell", "medium_cell", "large_cell"))
    labeled["brightness_label"] = _categorize(
        targets["mean_intensity_mean"], brightness_thr, ("dim", "balanced_brightness", "bright")
    )
    labeled["texture_label"] = _categorize(
        targets["texture_contrast_mean"], texture_thr, ("smooth", "mixed_texture", "granular")
    )
    labeled["capture_phase"] = _categorize(
        targets["capture_time_norm"], (1.0 / 3.0, 2.0 / 3.0), ("early_capture", "mid_capture", "late_capture")
    )
    labeled["prompt"] = labeled.apply(
        lambda row: (
            f"raw microalgae microscopy field, {row['condition'].lower()} condition, "
            f"{int(row['nominal_time_h'])}h time point, {row['density_label']} cell field, "
            f"{row['size_label'].replace('_', ' ')}, {row['brightness_label'].replace('_', ' ')}, "
            f"{row['texture_label'].replace('_', ' ')}, {row['capture_phase'].replace('_', ' ')}"
        ),
        axis=1,
    )

    target_cols = [
        "field_key",
        "state_id",
        "condition",
        "condition_id",
        "nominal_time_h",
        "field_id",
        "sample_key",
        "mask_relpath",
        "SPLIT",
        "source_field_key",
        "source_field_id",
        "source_sample_key",
        "source_actual_time_h",
        "source_time_bin_min",
        "source_time_bin_h",
        "cond_light",
        "cond_dark",
        "time_norm",
        "n_cells",
        "max_instance_label",
        "capture_order_in_state",
        "capture_order_norm",
        "seconds_from_state_start",
        "capture_time_norm",
        "actual_time_h",
        "time_bin_min",
        "time_bin_h",
        "density_label",
        "size_label",
        "brightness_label",
        "texture_label",
        "capture_phase",
    ]
    target_table = labeled[target_cols].sort_values(["condition", "nominal_time_h", "field_id"], kind="stable")

    prompt_table = labeled[["field_key", "state_id", "prompt"]].rename(columns={"field_key": "target_field_key"})
    prompt_table = prompt_table.sort_values("target_field_key", kind="stable").reset_index(drop=True)
    return target_table.reset_index(drop=True), prompt_table


def _build_pair_index(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = targets.sort_values(["condition", "nominal_time_h", "capture_order_in_state", "field_id"], kind="stable")

    for row in ordered.itertuples(index=False):
        pair_id = f"pair_{row.field_key}"
        common = {
            "CPD_NAME": row.field_key,
            "BATCH": pair_id,
            "SPLIT": row.SPLIT,
            "STATE": "field_pair",
            "SOURCE_KEY": row.source_field_key,
            "TARGET_KEY": row.field_key,
            "source_condition": row.condition,
            "source_time": 0,
            "target_condition": row.condition,
            "target_time": int(row.nominal_time_h),
            "condition_id": int(row.condition_id),
            "source_actual_time_h": float(row.source_actual_time_h),
            "target_actual_time_h": float(row.actual_time_h),
            "source_time_bin_min": int(row.source_time_bin_min),
            "target_time_bin_min": int(row.time_bin_min),
            "source_time_bin_h": float(row.source_time_bin_h),
            "target_time_bin_h": float(row.time_bin_h),
            "target_nominal_time_h": int(row.nominal_time_h),
            "source_field_id": row.source_field_id,
            "target_field_id": row.field_id,
            "target_n_cells": int(row.n_cells),
            "target_mask_relpath": row.mask_relpath,
            "target_max_instance_label": int(row.max_instance_label),
            "target_capture_order_in_state": int(row.capture_order_in_state),
            "target_seconds_from_state_start": float(row.seconds_from_state_start),
        }
        rows.append(
            {
                "SAMPLE_KEY": row.source_sample_key,
                "ANNOT": "negative_control",
                "PAIR_ROLE": "source",
                **common,
            }
        )
        rows.append(
            {
                "SAMPLE_KEY": row.sample_key,
                "ANNOT": "treated",
                "PAIR_ROLE": "target",
                **common,
            }
        )

    index_df = pd.DataFrame(rows)
    return index_df


def build_generation_artifacts() -> dict[str, pd.DataFrame]:
    metadata, summary = _load_field_frames()
    fields = _prepare_fields(metadata, summary)
    targets = _assign_sources(fields)

    embedding = _build_embeddings(targets)
    target_table, prompt_table = _build_labels_and_prompts(targets)
    index_df = _build_pair_index(target_table)

    return {
        "index": index_df,
        "embedding": embedding,
        "targets": target_table,
        "prompts": prompt_table,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = build_generation_artifacts()

    index_path = OUT / "index.csv"
    embedding_path = OUT / "embedding.csv"
    targets_path = OUT / "targets.csv"
    prompts_path = OUT / "prompts.csv"
    summary_path = OUT / "summary.json"

    artifacts["index"].to_csv(index_path)
    artifacts["embedding"].to_csv(embedding_path)
    artifacts["targets"].to_csv(targets_path, index=False)
    artifacts["prompts"].to_csv(prompts_path, index=False)

    summary = {
        "n_pairs": int(len(artifacts["index"]) // 2),
        "n_rows_index": int(len(artifacts["index"])),
        "n_target_fields": int(len(artifacts["embedding"])),
        "embedding_dim": int(artifacts["embedding"].shape[1]),
        "splits": artifacts["targets"]["SPLIT"].value_counts().to_dict(),
        "states": (
            artifacts["targets"]
            .groupby(["condition", "nominal_time_h", "SPLIT"])
            .size()
            .rename("n_fields")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {index_path} with shape {artifacts['index'].shape}")
    print(f"wrote {embedding_path} with shape {artifacts['embedding'].shape}")
    print(f"wrote {targets_path} with shape {artifacts['targets'].shape}")
    print(f"wrote {prompts_path} with shape {artifacts['prompts'].shape}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
