"""Delta-direction SCATTER: the honest visualization of what Delta-direction measures.

For each channel, plot per-gene (x = target_mean - src_mean, y = generated_mean - src_mean).
Points on the y=x diagonal => model moved that channel in the REAL perturbation direction.
The Pearson correlation of each panel IS the Delta-dir-corr reported by aggregate_eval.
This replaces blurry spatial mean-images: our perturbation effect lives in per-channel
intensity shifts, not pixel layout, so this is the figure that actually shows the signal.

Reads <run>/aggregate_eval_by_gene.csv (run aggregate_eval.py for the target epoch first).
Usage: python scripts/delta_scatter.py <run_dir> [min_n]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CH = ["Perilipin", "Calreticulin", "pS6RP"]


def main():
    run = sys.argv[1].rstrip("/")
    min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    g = pd.read_csv(f"{run}/aggregate_eval_by_gene.csv").set_index("gene")
    g = g[g["n"] >= min_n]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, ch in zip(axes, CH):
        dr = g[f"tgt_{ch}"] - g[f"src_{ch}"]      # real shift
        dg = g[f"gen_{ch}"] - g[f"src_{ch}"]      # generated shift
        r = pearsonr(dr, dg)[0]
        ax.axhline(0, color="0.8", lw=.8); ax.axvline(0, color="0.8", lw=.8)
        lim = max(abs(dr).max(), abs(dg).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=.8, alpha=.5)  # y=x
        ax.scatter(dr, dg, s=28, c="#2b6cb0", edgecolor="w", linewidth=.5, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("real shift  (target - control)", fontsize=9)
        ax.set_ylabel("generated shift  (gen - control)", fontsize=9)
        ax.set_title(f"{ch}   Delta-dir r = {r:.2f}", fontsize=11)
        ax.set_aspect("equal", "box")
    fig.suptitle(f"Per-gene perturbation-direction recovery (n={len(g)} genes)  "
                 f"-- points on diagonal = correct direction", fontsize=11)
    fig.tight_layout()
    out = f"{run}/delta_scatter.jpg"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}  ({len(g)} genes)")


if __name__ == "__main__":
    main()
