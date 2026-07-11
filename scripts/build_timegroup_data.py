#!/usr/bin/env python3
"""Build 5-minute EXIF-binned training data for microalgae single cells.

The single-cell lane maps 0h source populations to target populations grouped
by condition and rounded EXIF acquisition time. Labels are intentionally
coarser than per-field timestamps while remaining acquisition-time aware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_field_metadata import build_field_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "microalgae_v1"
OUT = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "timepoint"

IMAGE_ROOT = RAW_ROOT / "single_cell_images"
FIELD_METADATA = REPO_ROOT / "data" / "processed" / "microalgae_v1" / "views" / "field" / "metadata.csv"
CELL_ALLOWLIST = OUT / "cell_allowlist.csv"
FUSIONODE_DATA = Path(os.environ.get("FUSIONODE_DATA", "/home/shockley/myproject/FusionODE/data"))
FUSIONODE_CELL_SELECTION = FUSIONODE_DATA / "cell_groups_3way_seed7.csv"

TRAIN_FRACTION = 0.85
SEED = 42
TIME_BIN_MINUTES = 5


def _time_bin_minutes(actual_h: float) -> int:
    return int(round((float(actual_h) * 60.0) / TIME_BIN_MINUTES) * TIME_BIN_MINUTES)


def _time_bin_label(condition: str, bin_min: int) -> str:
    hours = int(bin_min) // 60
    minutes = int(bin_min) % 60
    return f"{condition.lower()}_{hours:03d}h{minutes:02d}m"


def _enumerate_single_cell_images(image_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_path in sorted(image_root.glob("*h/*/*.png")):
        condition = image_path.parent.name
        time_label = image_path.parent.parent.name
        if condition not in {"Dark", "Light"} or not time_label.endswith("h"):
            continue
        cell_id = image_path.stem
        field_id = cell_id.rsplit("_", 1)[0]
        rows.append(
            {
                "sample_relpath": str(image_path.relative_to(image_root)),
                "cell_id": cell_id,
                "field_id": field_id,
                "condition": condition,
                "time": int(time_label[:-1]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No single-cell PNG files found under {image_root}")
    return pd.DataFrame(rows)


def _load_cell_allowlist() -> pd.DataFrame:
    if CELL_ALLOWLIST.exists():
        allowlist = pd.read_csv(CELL_ALLOWLIST)
    elif FUSIONODE_CELL_SELECTION.exists():
        allowlist = pd.read_csv(FUSIONODE_CELL_SELECTION)
        CELL_ALLOWLIST.parent.mkdir(parents=True, exist_ok=True)
        keep_cols = [c for c in ["condition", "time", "cell_id", "group_id", "group"] if c in allowlist.columns]
        allowlist[keep_cols].to_csv(CELL_ALLOWLIST, index=False)
    else:
        raise FileNotFoundError(
            "Missing FusionODE cell quality allowlist. Expected either "
            f"{CELL_ALLOWLIST} or {FUSIONODE_CELL_SELECTION}."
        )

    required = {"condition", "time", "cell_id"}
    missing = required - set(allowlist.columns)
    if missing:
        raise ValueError(f"Cell allowlist is missing required columns: {sorted(missing)}")

    allowlist = allowlist.copy()
    allowlist["condition"] = allowlist["condition"].astype(str)
    allowlist["time"] = allowlist["time"].astype(float).astype(int)
    allowlist["cell_id"] = allowlist["cell_id"].astype(str).str.replace(".png", "", regex=False)
    return allowlist


def _filter_to_allowlist(cells: pd.DataFrame, allowlist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(cells)
    filtered = cells.merge(
        allowlist[["condition", "time", "cell_id"]],
        on=["condition", "time", "cell_id"],
        how="inner",
        validate="1:1",
    )
    return filtered, before - len(filtered)


def build_timegroup_embeddings(
    field_metadata: pd.DataFrame, leave_out_hours: list[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build embeddings and field-to-timegroup mapping."""
    fm = field_metadata.copy()
    fm["capture_dt"] = pd.to_datetime(fm["capture_datetime"])
    global_start = fm["capture_dt"].min()
    global_span = (fm["capture_dt"].max() - global_start).total_seconds()
    if global_span <= 0:
        raise ValueError("Field metadata capture times must span a positive duration")

    fm["actual_h"] = (fm["capture_dt"] - global_start).dt.total_seconds() / 3600.0
    fm["time_bin_min"] = fm["actual_h"].apply(_time_bin_minutes).astype(int)
    fm["time_bin_h"] = (fm["time_bin_min"] / 60.0).astype(float)
    fm["timegroup_key"] = fm.apply(lambda r: _time_bin_label(r["condition"], int(r["time_bin_min"])), axis=1)

    fm_tgt = fm[fm["nominal_time_h"] > 0].copy()
    if leave_out_hours:
        fm_tgt = fm_tgt[~fm_tgt["nominal_time_h"].isin(leave_out_hours)]

    # One embedding per (condition, 5-minute EXIF bin), using only acquisition
    # time and light/dark state derived from localized field metadata.
    tg_unique = (
        pd.concat([fm[fm["nominal_time_h"] == 0], fm_tgt])
        .groupby(["condition", "time_bin_min"], as_index=False)["actual_h"]
        .mean()
        .sort_values(["condition", "time_bin_min"], kind="stable")
    )

    rows = []
    for _, row in tg_unique.iterrows():
        cond = row["condition"]
        bin_min = int(row["time_bin_min"])
        ah = row["actual_h"]
        time_norm = (ah * 3600) / global_span

        d = {
            "timegroup_key": _time_bin_label(cond, bin_min),
            "cond_light": 1.0 if cond == "Light" else 0.0,
            "cond_dark": 1.0 if cond == "Dark" else 0.0,
            "time_norm": time_norm,
            "time_bin_h": bin_min / 60.0,
        }
        rows.append(d)

    emb = pd.DataFrame(rows).set_index("timegroup_key").sort_index().astype(np.float32)

    # Map: field_id -> timegroup_key
    field_map = pd.concat([fm[fm["nominal_time_h"] == 0], fm_tgt])[
        [
            "field_id",
            "condition",
            "timegroup_key",
            "actual_h",
            "time_bin_min",
            "time_bin_h",
            "nominal_time_h",
        ]
    ].copy()
    field_map["field_id"] = field_map["field_id"].astype(str)

    return emb, field_map


def build_timegroup_index(
    cells: pd.DataFrame,
    field_map: pd.DataFrame,
    leave_out_hours: list[int] | None = None,
) -> pd.DataFrame:
    """Build ctrl/trt index with CPD_NAME = timegroup_key."""
    rng = np.random.RandomState(SEED)
    cells = cells.copy()

    # Map each crop to its exact source field metadata row.
    cells = cells.merge(
        field_map,
        left_on=["field_id", "condition", "time"],
        right_on=["field_id", "condition", "nominal_time_h"],
        how="inner",
        validate="m:1",
    )

    controls_all = cells[cells["time"] == 0].copy()
    ctrl_by_cond = {
        "Dark": controls_all[controls_all["condition"] == "Dark"],
        "Light": controls_all[controls_all["condition"] == "Light"],
    }

    treated = cells[cells["time"] > 0].copy()
    if leave_out_hours:
        treated = treated[~treated["time"].isin([float(h) for h in leave_out_hours])]

    train_rows, test_rows = [], []

    for tg_key, grp in treated.groupby("timegroup_key"):
        cond = grp["condition"].iloc[0]
        controls = ctrl_by_cond[cond]
        n = len(grp)
        indices = np.arange(n)
        rng.shuffle(indices)
        split_at = max(1, int(n * TRAIN_FRACTION))

        for split_name, chosen in (("train", indices[:split_at]), ("test", indices[split_at:])):
            if len(chosen) == 0:
                continue
            for ci in chosen:
                row = grp.iloc[ci]
                rec = {
                    "SAMPLE_KEY": row["sample_relpath"],
                    "CPD_NAME": tg_key,
                    "ANNOT": "treated",
                    "BATCH": cond,
                    "SPLIT": split_name,
                    "sgRNA": tg_key,
                    "cluster_type": "NA",
                    "condition_id": 0,
                    "target_condition": cond,
                    "target_time": int(row["time"]),
                    "source_time": 0,
                    "source_actual_time_h": float("nan"),
                    "target_actual_time_h": float(row["actual_h"]),
                    "source_time_bin_min": pd.NA,
                    "target_time_bin_min": int(row["time_bin_min"]),
                    "source_time_bin_h": float("nan"),
                    "target_time_bin_h": float(row["time_bin_h"]),
                    "target_nominal_time_h": int(row["time"]),
                }
                (train_rows if split_name == "train" else test_rows).append(rec)

            ctrl_idx = rng.choice(len(controls), size=len(chosen), replace=True)
            for target_i, ci in zip(chosen, ctrl_idx, strict=True):
                row = grp.iloc[target_i]
                crow = controls.iloc[ci]
                rec = {
                    "SAMPLE_KEY": crow["sample_relpath"],
                    "CPD_NAME": tg_key,
                    "ANNOT": "negative_control",
                    "BATCH": cond,
                    "SPLIT": split_name,
                    "sgRNA": tg_key,
                    "cluster_type": "NA",
                    "condition_id": 0,
                    "target_condition": cond,
                    "target_time": int(row["time"]),
                    "source_time": 0,
                    "source_actual_time_h": float(crow["actual_h"]),
                    "target_actual_time_h": float(row["actual_h"]),
                    "source_time_bin_min": int(crow["time_bin_min"]),
                    "target_time_bin_min": int(row["time_bin_min"]),
                    "source_time_bin_h": float(crow["time_bin_h"]),
                    "target_time_bin_h": float(row["time_bin_h"]),
                    "target_nominal_time_h": int(row["time"]),
                }
                (train_rows if split_name == "train" else test_rows).append(rec)

    idx = pd.DataFrame(train_rows + test_rows)
    return idx.sort_values(["SPLIT", "BATCH", "ANNOT", "SAMPLE_KEY"], kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leave-out", type=int, nargs="*", default=None)
    parser.add_argument("--suffix", type=str, default="timegroup")
    args = parser.parse_args()

    leave_out = args.leave_out or []
    suffix = args.suffix
    OUT.mkdir(parents=True, exist_ok=True)

    cells_all = _enumerate_single_cell_images(IMAGE_ROOT)
    allowlist = _load_cell_allowlist()
    cells, n_filtered = _filter_to_allowlist(cells_all, allowlist)
    if cells.empty:
        raise ValueError("Cell allowlist removed all single-cell images")

    if FIELD_METADATA.exists():
        fm = pd.read_csv(FIELD_METADATA)
    else:
        fm, _ = build_field_frames()
        FIELD_METADATA.parent.mkdir(parents=True, exist_ok=True)
        fm.to_csv(FIELD_METADATA, index=False)

    print("Building timegroup embeddings...")
    emb, field_map = build_timegroup_embeddings(fm, leave_out)
    n_labels = emb.shape[0]
    n_times = emb["time_norm"].nunique()
    print(f"  Labels: {n_labels} (5-minute EXIF bins)")
    print(f"  Unique time_norm: {n_times}")

    print("Building index...")
    idx = build_timegroup_index(cells, field_map, leave_out)

    idx_path = OUT / "index.csv"
    emb_path = OUT / "embedding.csv"
    idx.to_csv(idx_path)
    emb.to_csv(emb_path)

    treated = idx[idx["ANNOT"] == "treated"]
    label_counts = treated.groupby("CPD_NAME").size()
    sparse_labels = label_counts[label_counts < 2].sort_values()

    s = {
        "n_labels": int(emb.shape[0]),
        "unique_time_norm": int(n_times),
        "embedding_dim": int(emb.shape[1]),
        "index_rows": int(len(idx)),
        "train_rows": int((idx.SPLIT == "train").sum()),
        "test_rows": int((idx.SPLIT == "test").sum()),
        "raw_cell_images": int(len(cells_all)),
        "allowed_cell_images": int(len(cells)),
        "filtered_cell_images": int(n_filtered),
        "cell_allowlist": str(CELL_ALLOWLIST.relative_to(REPO_ROOT)),
        "leave_out_hours": leave_out,
        "time_source": "EXIF 5-minute bins from localized field metadata",
        "time_bin_minutes": TIME_BIN_MINUTES,
        "labels_with_lt_2_treated_samples": sparse_labels.to_dict(),
    }
    json.dump(s, (OUT / "summary.json").open("w"), indent=2)

    print(f"\n{'='*50}")
    for k, v in s.items():
        print(f"  {k}: {v}")
    if sparse_labels.empty:
        print("  warning: none")
    else:
        print(f"  warning: {len(sparse_labels)} labels have fewer than 2 treated samples")


if __name__ == "__main__":
    main()
