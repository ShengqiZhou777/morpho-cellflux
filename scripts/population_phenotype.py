"""Population phenotype analysis: does the model reproduce a real perturbation effect
at the DISTRIBUTION level (where subtle in-vivo CRISPR effects actually live)?

Per cell, a phenotype scalar from one panel channel; compare REAL control vs GENERATED
vs REAL KO distributions for a gene set, with Mann-Whitney directional p-values. This is
the honest "visible effect" figure for subtle data: single cells are dominated by
cell-to-cell variance, but the population shift is real and recovered by the model.

Phenotypes (channel, reducer):
  lipid  = Perilipin puncta  (R / npz[5], top-2% foreground mean)  -- steatosis genes up
  upr    = Calreticulin mean (G / npz[9], foreground mean)         -- UPR genes up
  mtor   = pS6RP mean        (B / npz[10], foreground mean)        -- mTOR genes

Usage: python scripts/population_phenotype.py <run_dir> <phenotype> <gene1,gene2,..> [epoch]
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default image dir; overridden per-run from the run's args.json image_path in main()
# so eval reads the SAME npz cells the run trained on (e.g. diet vs CRISPR).
IMG = os.path.join(REPO_ROOT, "data/raw/crispr/images")
# phenotype -> (RGB index, npz channel, reducer name)
PHENO = {"lipid": (0, 5, "puncta"), "upr": (1, 9, "mean"), "mtor": (2, 10, "mean")}
LABEL = {"lipid": "Perilipin puncta (top-2% mean)",
         "upr": "Calreticulin mean", "mtor": "pS6RP mean"}


def reduce_plane(r, how):
    fg = r[r > 0.05]
    if fg.size < 50:
        return None
    if how == "puncta":
        return float(np.mean(np.sort(fg)[-max(1, int(0.02 * fg.size)):]))
    return float(fg.mean())


def main():
    global IMG
    run = sys.argv[1].rstrip("/")
    pheno = sys.argv[2]
    genes = sys.argv[3].split(",")
    rgb_i, npz_c, how = PHENO[pheno]
    aj = f"{run}/args.json"
    if os.path.exists(aj):
        _ip = json.load(open(aj)).get("image_path")
        if _ip:
            IMG = _ip if os.path.isabs(_ip) else os.path.join(REPO_ROOT, _ip)
    if len(sys.argv) > 4:
        ed = f"{run}/fid_samples/epoch-{sys.argv[4]}"
    else:
        ed = sorted(glob.glob(f"{run}/fid_samples/epoch-*"),
                    key=lambda p: int(p.split("-")[-1]))[-1]
    tpe = f"{ed}/trt2ctrl_idx.json"
    tp = tpe if os.path.exists(tpe) else f"{run}/fid_samples/trt2ctrl_idx.json"
    trt2ctrl = json.load(open(tp))

    def from_npz(cid):
        p = f"{IMG}/{cid}.npz"
        return None if not os.path.exists(p) else reduce_plane(np.load(p)["x"][npz_c], how)

    def from_png(png):
        a = (np.asarray(Image.open(png).convert("RGB")) / 255.0)[:, :, rgb_i]
        return reduce_plane(a, how)

    ctrl, gen, ko, seen = [], [], [], set()
    for g in genes:
        for png in glob.glob(f"{ed}/{g}/*.png"):
            tid = os.path.splitext(os.path.basename(png))[0]
            cid = trt2ctrl.get(tid)
            v = from_png(png);  (v is not None) and gen.append(v)
            kv = from_npz(tid); (kv is not None) and ko.append(kv)
            if cid and cid not in seen:
                cv = from_npz(cid)
                if cv is not None:
                    ctrl.append(cv); seen.add(cid)

    def stat(a):
        a = np.array(a); return f"n={len(a)} mean={a.mean():.3f} med={np.median(a):.3f}"
    print(f"phenotype={pheno} genes={genes} epoch={os.path.basename(ed)}")
    for nm, a in [("REAL control", ctrl), ("GENERATED", gen), ("REAL KO", ko)]:
        print(f"  {nm:13}: {stat(a)}")
    p_real = mannwhitneyu(ctrl, ko, alternative="less")[1]
    p_gen = mannwhitneyu(ctrl, gen, alternative="less")[1]
    print(f"  Mann-Whitney ctrl<KO  p={p_real:.2e}")
    print(f"  Mann-Whitney ctrl<gen p={p_gen:.2e}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    parts = ax.violinplot([ctrl, gen, ko], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#888", "#2b6cb0", "#c0392b"][i]); pc.set_alpha(.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["REAL\ncontrol", "GENERATED\n(model)", "REAL\nKO"])
    ax.set_ylabel(LABEL[pheno])
    ax.set_title(f"{pheno} phenotype {genes}\nctrl<KO p={p_real:.1e}  |  ctrl<gen p={p_gen:.1e}")
    fig.tight_layout()
    out = f"{run}/population_{pheno}_{'-'.join(genes)}_{os.path.basename(ed)}.jpg"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
