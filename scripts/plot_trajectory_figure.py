"""Render a CellFlux-style perturbation-trajectory figure from interpolation .npz files.

Input: `sample_*_traj.npz` written by `eval_loop.save_interpolation_grid` (--interpolate mode),
each holding ONE cell:
    control (C,H,W) | trajectory (T,C,H,W) | target (C,H,W) | gene (int) | t_grid (T)   -- all [-1,1]
Lays them out (one row per cell) as:
    Control --> [ Interpolation frames ] --> Predicted  ⇢  Perturbed
with CellFlux-style colored boxes, arrows, group labels and a legend.

Honest-figure options (data is UNPAIRED -- the per-cell `target` is an ARBITRARY real perturbed
cell, NOT the true counterfactual of `control`):
  --montage-dir/--montage-keys/--montage-channels : replace the single "Perturbed" cell with a
      montage of real perturbed cells, i.e. the phenotype POPULATION (recommended -- avoids
      implying a one-to-one target).
  --channel N [--channel-name NAME] : render only one panel channel (grayscale) to foreground the
      responsive marker (e.g. Perilipin for lipid), the honest signal for weak perturbations.

Usage:
  python scripts/plot_trajectory_figure.py outputs/<run>/interpolation --out fig.png
  python scripts/plot_trajectory_figure.py <dir> --out peri.png --channel 1 --channel-name Perilipin
  python scripts/plot_trajectory_figure.py <dir> --out montage.png \
      --montage-dir data/raw/diet_extracted_images --montage-channels 9 5 8 \
      --montage-keys <id1> <id2> <id3> <id4>
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL = {"Control": "#E8932E", "Interpolation": "#7E57C2",
       "Predicted": "#D7484B", "Perturbed": "#43A047"}
ORDER = ["Control", "Predicted", "Interpolation", "Perturbed"]  # legend order (matches CellFlux)


def to_rgb(img, channel=None, cmap=None):
    """(C,H,W) in [-1,1] -> displayable. channel=int -> single channel; if cmap given, return
    a 2-D array (imshow colormaps it), else a grayscale (H,W,3)."""
    a = (np.clip(np.asarray(img, dtype=np.float32), -1.0, 1.0) + 1.0) / 2.0
    if channel is not None:
        g = a[channel]
        return g if cmap else np.stack([g, g, g], axis=-1)
    if a.shape[0] < 3:
        z = np.zeros((3,) + a.shape[1:], np.float32)
        z[: a.shape[0]] = a
        a = z
    return np.transpose(a[:3], (1, 2, 0))


def subsample(T, k):
    return list(range(T)) if T <= k else list(np.linspace(0, T - 1, k).astype(int))


def load_raw_cell(npz_path, channels):
    """Load a raw extracted_images npz cell ([0,1]) -> selected channels in [-1,1]."""
    x = np.load(npz_path)["x"][list(channels)]
    return x * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="a sample_*_traj.npz file or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, default=None, help="render only this panel channel (grayscale)")
    ap.add_argument("--channel-name", default=None)
    ap.add_argument("--cmap", default=None, help="colormap for single-channel mode (e.g. magma); default grayscale")
    ap.add_argument("--max-frames", type=int, default=7, help="interpolation frames to show")
    ap.add_argument("--max-rows", type=int, default=4)
    ap.add_argument("--gene-names", default=None, help="optional json {gene_id: name}")
    ap.add_argument("--montage-dir", default=None)
    ap.add_argument("--montage-keys", nargs="*", default=None)
    ap.add_argument("--montage-channels", nargs="*", type=int, default=None)
    args = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(args.input, "sample_*_traj.npz")))
             if os.path.isdir(args.input) else [args.input])[: args.max_rows]
    assert paths, f"no sample_*_traj.npz found under {args.input}"
    gene_names = json.load(open(args.gene_names)) if args.gene_names else {}
    cells = [dict(np.load(p, allow_pickle=True)) for p in paths]

    montage = None
    if args.montage_dir and args.montage_keys:
        montage = [load_raw_cell(os.path.join(args.montage_dir, f"{k}.npz"), args.montage_channels)
                   for k in args.montage_keys]

    def groups_of(c):
        ti = subsample(c["trajectory"].shape[0], args.max_frames)
        interp = [c["trajectory"][i] for i in ti]
        pert = montage if montage is not None else [np.asarray(c["target"])]
        return [("Control", [np.asarray(c["control"])]),
                ("Interpolation", interp),
                ("Predicted", [c["trajectory"][-1]]),
                ("Perturbed", pert)]

    # shared x-layout from the first row's structure
    FS, GIN, GGRP, RGAP, PAD = 1.0, 0.16, 0.85, 0.75, 0.06
    spans, x = [], 0.0
    for name, imgs in groups_of(cells[0]):
        centers = []
        for _ in imgs:
            centers.append(x + FS / 2); x += FS + GIN
        spans.append((name, centers[0] - FS / 2, centers[-1] + FS / 2, centers))
        x += GGRP - GIN
    total_w = x - (GGRP - GIN)
    n_rows = len(cells); row_h = FS + RGAP

    fig, ax = plt.subplots(figsize=(max(8, total_w), n_rows * row_h + 1.4))
    ax.set_xlim(-0.5, total_w + 0.2); ax.set_ylim(-1.4, n_rows * row_h)
    ax.axis("off")

    for r, c in enumerate(cells):
        y0 = (n_rows - 1 - r) * row_h
        yc = y0 + FS / 2
        for (name, imgs), (gname, gx0, gx1, centers) in zip(groups_of(c), spans):
            for img, xcen in zip(imgs, centers):
                _cm = args.cmap if (args.channel is not None and args.cmap) else None
                ax.imshow(to_rgb(img, args.channel, args.cmap),
                          extent=[xcen - FS / 2, xcen + FS / 2, y0, y0 + FS],
                          cmap=_cm, vmin=0.0 if _cm else None, vmax=1.0 if _cm else None,
                          zorder=2, aspect="auto")
            ax.add_patch(FancyBboxPatch(
                (gx0 - PAD, y0 - PAD), (gx1 - gx0) + 2 * PAD, FS + 2 * PAD,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fill=False, edgecolor=COL[name], lw=2.5, zorder=3))
        for i in range(len(spans) - 1):
            x1b, n1 = spans[i][2], spans[i][0]
            x2a = spans[i + 1][1]
            ax.annotate("", xy=(x2a - 0.03, yc), xytext=(x1b + 0.03, yc),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#555555",
                                        linestyle="--" if n1 == "Predicted" else "-"), zorder=1)
        gid = int(c["gene"]) if c.get("gene") is not None else None
        lbl = gene_names.get(str(gid), f"gene {gid}") if gid is not None else ""
        ax.text(-0.45, yc, lbl, ha="right", va="center", fontsize=10, rotation=90)

    for name, gx0, gx1, _ in spans:
        ax.text((gx0 + gx1) / 2, -0.32, name, ha="center", va="top",
                fontsize=11, color=COL[name], fontweight="bold")
    fig.legend(handles=[plt.Line2D([0], [0], color=COL[k], lw=4, label=k) for k in ORDER],
               loc="lower center", ncol=4, frameon=False, fontsize=11)
    ttl = "Perturbation trajectory" + (f"  —  {args.channel_name} channel" if args.channel_name else "")
    ax.set_title(ttl, fontsize=12)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}  ({n_rows} row(s), {len(spans)} groups"
          + (", montage perturbed" if montage is not None else "") + ")")


if __name__ == "__main__":
    main()
