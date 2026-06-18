"""Per-gene MEAN-image figure: visualize the population-level perturbation shift
that Delta-direction measures (GPU-free, from saved fid_samples).

Per lead gene, average many generated cells / their source-control cells / their
real-target cells into a mean RGB image, so the subtle per-cell effect (drowned by
cell-to-cell variance in a single-cell montage) becomes visible at the population
level. This is the figure that MATCHES our data: single-cell interpolation can't
show weak CRISPR effects, but the mean shift can. Pair with aggregate_eval Delta-dir.

Rows = genes; cols = [control mean | generated mean | target mean | gen-ctrl diff].
RGB = panel2 [5,9,10] = Perilipin / Calreticulin / pS6RP.

Usage: python scripts/mean_image_figure.py <run_dir> <gene1,gene2,...> [epoch]
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(REPO_ROOT, "data/raw/extracted_images")
NPZ_CH = [5, 9, 10]  # panel2: Perilipin(R), Calreticulin(G), pS6RP(B)


def npz_rgb(cid):
    p = f"{IMG}/{cid}.npz"
    if not os.path.exists(p):
        return None
    a = np.load(p)["x"][NPZ_CH]                 # (3,H,W) in [0,1]
    return np.clip(np.transpose(a, (1, 2, 0)), 0, 1)


def png_rgb(png):
    return np.asarray(Image.open(png).convert("RGB")) / 255.0


def mean_stack(imgs):
    return np.mean(np.stack(imgs, 0), 0) if imgs else None


def main():
    run = sys.argv[1].rstrip("/")
    genes = sys.argv[2].split(",")
    if len(sys.argv) > 3:
        epoch_dir = f"{run}/fid_samples/epoch-{sys.argv[3]}"
    else:
        epoch_dir = sorted(glob.glob(f"{run}/fid_samples/epoch-*"),
                           key=lambda p: int(p.split("-")[-1]))[-1]
    trt2ctrl = json.load(open(f"{run}/fid_samples/trt2ctrl_idx.json"))

    rows = []
    for gene in genes:
        gens, srcs, tgts = [], [], []
        for png in glob.glob(f"{epoch_dir}/{gene}/*.png"):
            tid = os.path.splitext(os.path.basename(png))[0]
            cid = trt2ctrl.get(tid)
            tgt = npz_rgb(tid)
            src = npz_rgb(cid) if cid else None
            if tgt is None or src is None:
                continue
            gens.append(png_rgb(png)); srcs.append(src); tgts.append(tgt)
        if not gens:
            continue
        rows.append((gene, len(gens), mean_stack(srcs), mean_stack(gens), mean_stack(tgts)))

    n = len(rows)
    titles = ["control mean", "generated mean", "target mean", "gen - ctrl (x4)"]
    fig, axes = plt.subplots(n, 4, figsize=(8.5, 2.2 * n))
    axes = np.atleast_2d(axes)
    for i, (gene, k, src, gen, tgt) in enumerate(rows):
        diff = np.clip(0.5 + 4.0 * (gen - src), 0, 1)   # amplified gen-ctrl shift
        for j, im in enumerate([src, gen, tgt, diff]):
            axes[i, j].imshow(im); axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
        axes[i, 0].text(-0.16, 0.5, f"{gene}\n(n={k})", transform=axes[i, 0].transAxes,
                        va="center", ha="right", fontsize=9)
    fig.suptitle(f"{os.path.basename(epoch_dir)}  per-gene MEAN  "
                 f"RGB=Perilipin/Calreticulin/pS6RP", fontsize=10)
    fig.tight_layout()
    out = f"{run}/mean_image_{os.path.basename(epoch_dir)}.jpg"
    fig.savefig(out, dpi=120)
    print(f"saved: {out}  ({n} genes: {[r[0] for r in rows]})")


if __name__ == "__main__":
    main()
