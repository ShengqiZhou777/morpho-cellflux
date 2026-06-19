"""Aggregate, per-perturbation evaluation of a CellFlux perturbmulti eval run.

Publication-oriented metrics (carried by aggregates, not cherry-picked images):
  For each channel (Perilipin, Alb, polyT) and each perturbation gene, compute the
  mean intensity of generated / real-target / source(control) cells, then across
  genes report:
    - Pearson/Spearman( per-gene generated mean , per-gene real-target mean )
        -> does the model reproduce the per-perturbation morphology ranking?
    - (control-init only) sign-agreement & corr of (gen-src) vs (tgt-src)
        -> does generation move in the real perturbation direction?

Usage:  python scripts/aggregate_eval.py <eval_run_dir> [min_n]
The run dir must contain fid_samples/<epoch>/<gene>/<target_id>.png and
fid_samples/trt2ctrl_idx.json (produced by train.py --save_fid_samples).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default image dir; overridden per-run from the run's args.json image_path in main()
# so eval reads the SAME npz cells the run trained on (e.g. diet vs CRISPR).
IMG = os.path.join(REPO_ROOT, "data/raw/extracted_images")
CHANNELS = [("Perilipin", 0), ("Calreticulin", 1), ("pS6RP", 2)]  # RGB order == npz[[5,9,10]]
NPZ_CH = [5, 9, 10]
LEAD = {"Perilipin": ["Insig1", "Pten", "Eif2s1", "Aars"],       # lipid/steatosis
        "Calreticulin": ["Sel1l", "Atp2a2", "Dnajb9"],            # UPR
        "pS6RP": ["Pten", "Tsc1", "Tsc2", "Mtor", "Cdc37"]}       # mTOR


def fg_means(img_chw_or_hwc, as_hwc):
    """Per-channel foreground means; fg = pixels where max-over-channels > 0.05."""
    a = img_chw_or_hwc
    if as_hwc:                       # (H,W,3) generated png in [0,1]
        flat = a.reshape(-1, 3)
    else:                            # (3,H,W) real npz panel in [0,1]
        flat = a.reshape(3, -1).T
    fg = flat.max(axis=1) > 0.05
    sel = flat[fg] if fg.sum() > 20 else flat
    return sel.mean(axis=0)          # [3]


def npz_panel_means(cid):
    p = f"{IMG}/{cid}.npz"
    if not os.path.exists(p):
        return None
    return fg_means(np.load(p)["x"][NPZ_CH], as_hwc=False)


def main():
    global IMG
    run = sys.argv[1].rstrip("/")
    min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    use_initial = None
    aj = f"{run}/args.json"
    if os.path.exists(aj):
        _a = json.load(open(aj))
        use_initial = _a.get("use_initial")
        _ip = _a.get("image_path")
        if _ip:
            IMG = _ip if os.path.isabs(_ip) else os.path.join(REPO_ROOT, _ip)
    if len(sys.argv) > 3:
        epoch_dir = f"{run}/fid_samples/epoch-{sys.argv[3]}"
    else:
        epoch_dir = sorted(glob.glob(f"{run}/fid_samples/epoch-*"),
                           key=lambda p: int(p.split("-")[-1]))[-1]
    # Prefer the per-epoch pairing (epoch-<e>/trt2ctrl_idx.json) so old epochs keep their
    # OWN treated->control pairing; fall back to the global file for runs evaluated before
    # the per-epoch fix landed.
    tp_epoch = f"{epoch_dir}/trt2ctrl_idx.json"
    tp_global = f"{run}/fid_samples/trt2ctrl_idx.json"
    tp = tp_epoch if os.path.exists(tp_epoch) else tp_global
    trt2ctrl = json.load(open(tp)) if os.path.exists(tp) else {}

    rows = []
    for gdir in sorted(glob.glob(f"{epoch_dir}/*")):
        gene = os.path.basename(gdir)
        for png in glob.glob(f"{gdir}/*.png"):
            tid = os.path.splitext(os.path.basename(png))[0]
            cid = trt2ctrl.get(tid)
            gen = fg_means(np.asarray(Image.open(png).convert("RGB")) / 255.0, as_hwc=True)
            tgt = npz_panel_means(tid)
            src = npz_panel_means(cid) if cid else None
            if tgt is None:
                continue
            rec = {"gene": gene}
            for name, k in CHANNELS:
                rec[f"gen_{name}"] = gen[k]
                rec[f"tgt_{name}"] = tgt[k]
                rec[f"src_{name}"] = src[k] if src is not None else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    g = df.groupby("gene").mean(numeric_only=True)
    g["n"] = df.groupby("gene").size()
    gg = g[g["n"] >= min_n]
    print(f"run: {run}")
    print(f"use_initial: {use_initial}   epoch dir: {os.path.basename(epoch_dir)}")
    print(f"images: {len(df)}   genes(n>={min_n}): {len(gg)}\n")

    print(f"{'channel':10} {'Pearson(gen,tgt)':>17} {'Spearman':>9} "
          f"{'dir-corr(Δ)':>12} {'sign-agree':>10}")
    summary = {}
    for name, _ in CHANNELS:
        pe = pearsonr(gg[f"gen_{name}"], gg[f"tgt_{name}"])[0]
        sp = spearmanr(gg[f"gen_{name}"], gg[f"tgt_{name}"])[0]
        dcorr, sign = np.nan, np.nan
        if use_initial in (1, 2):
            sub = gg[[f"gen_{name}", f"tgt_{name}", f"src_{name}"]].dropna()
            if len(sub) > 2:
                dg = sub[f"gen_{name}"] - sub[f"src_{name}"]
                dr = sub[f"tgt_{name}"] - sub[f"src_{name}"]
                dcorr = pearsonr(dg, dr)[0]
                sign = float((np.sign(dg) == np.sign(dr)).mean())
        summary[name] = dict(pearson=pe, spearman=sp, dir_corr=dcorr, sign_agree=sign)
        print(f"{name:10} {pe:17.3f} {sp:9.3f} {dcorr:12.3f} {sign:10.2f}")

    for name, _ in CHANNELS:
        genes = LEAD.get(name, [])
        cols = [c for c in ["n", f"src_{name}", f"gen_{name}", f"tgt_{name}"] if c in g.columns]
        sub = g.loc[g.index.isin(genes), cols]
        if len(sub):
            print(f"\nlead genes for {name}:")
            print(sub.to_string())

    g.to_csv(f"{run}/aggregate_eval_by_gene.csv")
    json.dump(summary, open(f"{run}/aggregate_eval_summary.json", "w"), indent=2)
    print(f"\nsaved: {run}/aggregate_eval_by_gene.csv + aggregate_eval_summary.json")


if __name__ == "__main__":
    main()
