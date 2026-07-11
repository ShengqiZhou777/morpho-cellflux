"""Aggregate evaluation for microalgae phenotype transport.

DEPRECATED: this metric collapses each cell to a single scalar
(fg_mean_intensity) and its PGC divides by d(src,tgt) which is ~0 for
microalgae, making it unstable and blind to morphology/diversity. Superseded by
phenoflux.eval.distribution_eval (full morphology distributions + identity
baseline guard). Kept for backward reference only.

Per-condition metrics:
  - PGC (Phenotypic Gap Closure): 1 - d(gen,tgt)/d(src,tgt) where d is Wasserstein distance
    Measures how much of the source→target morphology gap the model closes
  - Energy distance as alternative to Wasserstein
  - Direction correlation: Pearson(gen-src, tgt-src) for pixel-level shift alignment

Usage:
  python phenoflux/eval/aggregate_microalgae.py <eval_run_dir> [min_n] [epoch]

Reads:
  - fid_samples/epoch-<N>/<condition>/<cell_id>.png  (RGB generated images)
  - fid_samples/trt2ctrl_idx.json  (treated→control pairing)
  - data/raw/microalgae_v1/single_cell_images/<time>h/<condition>/<cell_id>.png

Outputs:
  - aggregate_eval_by_condition.csv
  - aggregate_eval_summary.json
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, wasserstein_distance, energy_distance


def load_rgb_image(path: str) -> np.ndarray | None:
    """Load RGB image and normalize to [0,1], return (H,W,3) or None if missing."""
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB")) / 255.0


def fg_mean_intensity(img: np.ndarray) -> float:
    """Mean pixel intensity over foreground (max-over-channels > 0.05)."""
    flat = img.reshape(-1, 3)
    fg = flat.max(axis=1) > 0.05
    sel = flat[fg] if fg.sum() > 20 else flat
    return float(sel.mean())


def compute_metrics(gen_vals: np.ndarray, tgt_vals: np.ndarray, src_vals: np.ndarray) -> dict:
    """Compute PGC and direction correlation for a condition's cell population.

    Args:
        gen_vals: (N,) generated cell intensities
        tgt_vals: (N,) real target intensities
        src_vals: (N,) source control intensities

    Returns:
        {wd_gen, wd_src, pgc_wd, ed_gen, ed_src, pgc_ed, dir_corr}
    """
    result = {}

    # Wasserstein distance
    wd_gen = wasserstein_distance(gen_vals, tgt_vals)
    wd_src = wasserstein_distance(src_vals, tgt_vals) if len(src_vals) >= 5 else np.nan
    result["wd_gen"] = wd_gen
    result["wd_src"] = wd_src
    result["pgc_wd"] = 1 - wd_gen / wd_src if wd_src > 0 else np.nan

    # Energy distance
    ed_gen = energy_distance(gen_vals, tgt_vals)
    ed_src = energy_distance(src_vals, tgt_vals) if len(src_vals) >= 5 else np.nan
    result["ed_gen"] = ed_gen
    result["ed_src"] = ed_src
    result["pgc_ed"] = 1 - ed_gen / ed_src if ed_src > 0 else np.nan

    # Direction correlation (pairwise shift alignment)
    if len(gen_vals) >= 3 and len(src_vals) >= 3:
        dg = gen_vals - src_vals
        dr = tgt_vals - src_vals
        if dg.std() > 1e-6 and dr.std() > 1e-6:
            result["dir_corr"] = pearsonr(dg, dr)[0]
        else:
            result["dir_corr"] = np.nan
    else:
        result["dir_corr"] = np.nan

    return result


def main():
    argv = sys.argv[1:]
    run_dir = Path(argv[0])
    min_n = int(argv[1]) if len(argv) > 1 else 5
    epoch_arg = argv[2] if len(argv) > 2 else None

    # Load run config
    args_json = run_dir / "args.json"
    if args_json.exists():
        args = json.load(open(args_json))
        image_root = Path(args.get("image_path", "data/raw/microalgae_v1/single_cell_images"))
    else:
        image_root = Path("data/raw/microalgae_v1/single_cell_images")

    # Find epoch directory
    if epoch_arg:
        epoch_dir = run_dir / "fid_samples" / f"epoch-{epoch_arg}"
    else:
        epoch_dirs = sorted(
            (run_dir / "fid_samples").glob("epoch-*"),
            key=lambda p: int(p.name.split("-")[-1])
        )
        epoch_dir = epoch_dirs[-1] if epoch_dirs else None

    if not epoch_dir or not epoch_dir.exists():
        print(f"No epoch directory found in {run_dir}/fid_samples/")
        sys.exit(1)

    # Load pairing (prefer per-epoch, fallback to global)
    pairing_epoch = epoch_dir / "trt2ctrl_idx.json"
    pairing_global = run_dir / "fid_samples" / "trt2ctrl_idx.json"
    pairing_path = pairing_epoch if pairing_epoch.exists() else pairing_global

    trt2ctrl = json.load(open(pairing_path)) if pairing_path.exists() else {}

    # Collect per-cell metrics
    rows = []
    for cond_dir in sorted(epoch_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name

        for png in cond_dir.glob("*.png"):
            cell_id = png.stem
            ctrl_id = trt2ctrl.get(cell_id)

            # Load images
            gen_img = load_rgb_image(str(png))
            tgt_candidates = list(image_root.glob(f"*/*/{cell_id}.png"))
            src_candidates = list(image_root.glob(f"*/*/{ctrl_id}.png")) if ctrl_id else []
            tgt_img = load_rgb_image(str(tgt_candidates[0])) if tgt_candidates else None
            src_img = load_rgb_image(str(src_candidates[0])) if src_candidates else None

            if gen_img is None or tgt_img is None:
                continue

            # Compute foreground mean intensity
            gen_val = fg_mean_intensity(gen_img)
            tgt_val = fg_mean_intensity(tgt_img)
            src_val = fg_mean_intensity(src_img) if src_img is not None else np.nan

            rows.append({
                "condition": condition,
                "cell_id": cell_id,
                "gen": gen_val,
                "tgt": tgt_val,
                "src": src_val,
            })

    if not rows:
        print("No valid cells found.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Aggregate by condition
    summary = {"conditions": {}}

    print(f"\nRun: {run_dir}")
    print(f"Epoch: {epoch_dir.name}")
    print(f"Total cells: {len(df)}")
    print(f"\n{'Condition':<20} {'N':>6} {'W(gen,tgt)':>11} {'W(src,tgt)':>11} {'PGC':>8} {'Dir-Corr':>9}")
    print("-" * 75)

    for condition, grp in df.groupby("condition"):
        if len(grp) < min_n:
            continue

        gen = grp["gen"].dropna().to_numpy()
        tgt = grp["tgt"].dropna().to_numpy()
        src = grp["src"].dropna().to_numpy()

        if len(gen) < 5 or len(tgt) < 5:
            continue

        metrics = compute_metrics(gen, tgt, src)
        summary["conditions"][condition] = metrics

        pgc = metrics["pgc_wd"]
        dcorr = metrics["dir_corr"]

        print(f"{condition:<20} {len(grp):>6} {metrics['wd_gen']:>11.4f} "
              f"{metrics['wd_src']:>11.4f} {pgc:>8.3f} {dcorr:>9.3f}")

    # Pooled metrics (all conditions together)
    gen_all = df["gen"].dropna().to_numpy()
    tgt_all = df["tgt"].dropna().to_numpy()
    src_all = df["src"].dropna().to_numpy()

    if len(gen_all) >= 5 and len(tgt_all) >= 5:
        pooled = compute_metrics(gen_all, tgt_all, src_all)
        summary["pooled"] = pooled

        print(f"\n{'POOLED':<20} {len(df):>6} {pooled['wd_gen']:>11.4f} "
              f"{pooled['wd_src']:>11.4f} {pooled['pgc_wd']:>8.3f} {pooled['dir_corr']:>9.3f}")

    # Save outputs
    df_out = df.groupby("condition").agg({
        "gen": "mean",
        "tgt": "mean",
        "src": "mean",
        "cell_id": "count",
    }).rename(columns={"cell_id": "n_cells"})

    out_csv = run_dir / "aggregate_eval_by_condition.csv"
    out_json = run_dir / "aggregate_eval_summary.json"

    df_out.to_csv(out_csv)
    json.dump(summary, open(out_json, "w"), indent=2)

    print(f"\nSaved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
