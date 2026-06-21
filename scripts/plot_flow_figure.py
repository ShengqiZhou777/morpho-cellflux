"""Render a flow-trajectory figure (redesigned, distinct from the CellFlux box layout).

Input: `sample_*_traj.npz` from `eval_loop.save_interpolation_grid` (--interpolate), each one cell:
    control (C,H,W) | trajectory (T,C,H,W) | target (C,H,W) | gene (int) | t_grid (T)  -- all [-1,1].

Design (intentionally NOT the CellFlux 4-colored-box / horizontal-arrow / bottom-legend motif):
  * a continuous film-strip of the ODE trajectory with a FLOW-TIME gradient bar (t: 0 -> 1) under it,
  * the responsive channel rendered with a perceptual sci colormap (default 'magma') so lipid puncta pop,
  * the single arbitrary "target" replaced by a MONTAGE of real perturbed cells (the phenotype POPULATION),
  * a quantitative readout: mean channel intensity vs flow-time, climbing from control toward the
    real-population band -- ties the picture to the gap_closed metric and keeps it honest.

Usage:
  python scripts/plot_flow_figure.py outputs/runs/diet/main/interpolation --out fig.png \
      --channel 1 --channel-name Perilipin --cmap magma \
      --montage-dir data/raw/diet_extracted_images --montage-channels 9 5 8 \
      --montage-keys <id1> <id2> <id3> <id4> --cond-name HFD
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

INK = "#2B2B2B"
ACCENT = "#C0392B"      # generated/endpoint accent
POP = "#1B7837"         # real-population accent
MUTED = "#8A8A8A"


def to_disp(img, channel, cmap):
    """(C,H,W) in [-1,1] -> displayable. Returns (array, is_rgb)."""
    a = (np.clip(np.asarray(img, np.float32), -1.0, 1.0) + 1.0) / 2.0
    if channel is not None:
        return a[channel], False          # 2-D, colormapped by imshow
    if a.shape[0] < 3:
        z = np.zeros((3,) + a.shape[1:], np.float32); z[: a.shape[0]] = a; a = z
    return np.transpose(a[:3], (1, 2, 0)), True


def load_raw(npz_path, channels):
    return np.load(npz_path)["x"][list(channels)] * 2.0 - 1.0


def fg_mean(img_chw, channel):
    """Mean intensity of `channel` over the cell foreground (pixels above a low quantile)."""
    a = (np.clip(np.asarray(img_chw, np.float32), -1.0, 1.0) + 1.0) / 2.0
    g = a[channel]
    fg = a.max(0) > 0.08
    return float(g[fg].mean()) if fg.any() else float(g.mean())


def subsample(T, k):
    return list(range(T)) if T <= k else list(np.linspace(0, T - 1, k).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--channel-name", default=None)
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=2)
    ap.add_argument("--cond-name", default="perturbed")
    ap.add_argument("--gene-names", default=None)
    ap.add_argument("--montage-dir", default=None)
    ap.add_argument("--montage-keys", nargs="*", default=None)
    ap.add_argument("--montage-channels", nargs="*", type=int, default=None)
    ap.add_argument("--curve-from", default=None,
                    help="dir of sample_*_traj.npz to draw ALL intensity curves (faint) + bold mean")
    ap.add_argument("--band-mean", type=float, default=None, help="real-population channel mean (band center)")
    ap.add_argument("--band-std", type=float, default=None, help="real-population channel std (band half-width)")
    ap.add_argument("--control-level", type=float, default=None, help="real control channel mean (dotted line)")
    args = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(args.input, "sample_*_traj.npz")))
             if os.path.isdir(args.input) else [args.input])[: args.max_rows]
    assert paths, f"no sample_*_traj.npz under {args.input}"
    gene_names = json.load(open(args.gene_names)) if args.gene_names else {}
    cells = [dict(np.load(p, allow_pickle=True)) for p in paths]

    montage = None
    if args.montage_dir and args.montage_keys:
        montage = [load_raw(os.path.join(args.montage_dir, f"{k}.npz"), args.montage_channels)
                   for k in args.montage_keys]

    ch, cmap = args.channel, args.cmap
    n = len(cells)
    # ---- figure scaffold: image band (top) + quantitative curve (bottom) ----
    fig = plt.figure(figsize=(13.5, 2.55 * n + 2.4), facecolor="white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.55 * n, 2.2], hspace=0.32)
    axI = fig.add_subplot(gs[0]); axI.axis("off")
    axC = fig.add_subplot(gs[1])

    FS, GUT, BLOCKGAP, ROWH = 1.0, 0.05, 0.7, 1.5
    # geometry: [control] gap [strip....] gap [montage 2xK]
    x_ctrl = 0.0
    strip0 = FS + BLOCKGAP
    ti0 = subsample(cells[0]["trajectory"].shape[0], args.max_frames)
    nf = len(ti0)
    strip_w = nf * FS + (nf - 1) * GUT
    mont0 = strip0 + strip_w + BLOCKGAP
    mont_cols = int(np.ceil(len(montage) / 2)) if montage else 1
    MS = 0.92
    mont_w = mont_cols * MS + (mont_cols - 1) * 0.05
    total_w = mont0 + mont_w
    axI.set_xlim(-0.6, total_w + 0.2); axI.set_ylim(-0.9, n * ROWH)

    def imshow(ax, img, x0, y0, w, h, frame=None, lw=1.0):
        disp, rgb = to_disp(img, ch, cmap)
        ax.imshow(disp if rgb else disp, extent=[x0, x0 + w, y0, y0 + h],
                  cmap=None if rgb else cmap, vmin=None if rgb else 0.0,
                  vmax=None if rgb else 1.0, zorder=2, aspect="auto",
                  interpolation="bilinear")
        if frame:
            ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=frame, lw=lw, zorder=3))

    lipid_curves = []
    for r, c in enumerate(cells):
        y0 = (n - 1 - r) * ROWH + 0.18
        traj = c["trajectory"]; tg = np.asarray(c["t_grid"]); ti = subsample(traj.shape[0], args.max_frames)
        # control
        imshow(axI, c["control"], x_ctrl, y0, FS, FS, frame=MUTED, lw=1.2)
        # film-strip
        x = strip0
        for j, idx in enumerate(ti):
            last = (j == len(ti) - 1)
            imshow(axI, traj[idx], x, y0, FS, FS,
                   frame=ACCENT if last else None, lw=2.0)
            x += FS + GUT
        # montage of real population
        if montage:
            for m, mc in enumerate(montage):
                cx = mont0 + (m // 2) * (MS + 0.05)
                cy = y0 + (1 - (m % 2)) * (FS - MS)
                imshow(axI, mc, cx, cy, MS, MS, frame=POP, lw=1.4)
        # row label
        gid = int(c["gene"]) if c.get("gene") is not None else None
        lbl = gene_names.get(str(gid), args.cond_name)
        axI.text(-0.5, y0 + FS / 2, lbl, ha="right", va="center", fontsize=11,
                 color=INK, rotation=90, fontweight="bold")
        # lipid trend along the FULL trajectory (not subsampled)
        if ch is not None:
            lipid_curves.append((tg, np.array([fg_mean(traj[k], ch) for k in range(traj.shape[0])])))

    # ---- flow-time gradient bar under the strip ----
    grad = np.linspace(0, 1, 256)[None, :]
    bar_y = -0.55
    axI.imshow(grad, extent=[strip0, strip0 + strip_w, bar_y, bar_y + 0.16],
               cmap="viridis", aspect="auto", zorder=2)
    axI.add_patch(plt.Rectangle((strip0, bar_y), strip_w, 0.16, fill=False, ec=INK, lw=0.8, zorder=3))
    axI.text(strip0, bar_y - 0.06, "t = 0", ha="left", va="top", fontsize=9, color=INK)
    axI.text(strip0 + strip_w, bar_y - 0.06, "t = 1", ha="right", va="top", fontsize=9, color=INK)
    axI.text(strip0 + strip_w / 2, bar_y - 0.06, "generative flow time",
             ha="center", va="top", fontsize=9.5, color=INK, style="italic")
    axI.text(x_ctrl + FS / 2, y_top := (n - 1) * ROWH + 0.18 + FS + 0.07, "control",
             ha="center", va="bottom", fontsize=10, color=MUTED, fontweight="bold")
    axI.text(strip0 + strip_w - FS / 2, y_top, "generated", ha="center", va="bottom",
             fontsize=10, color=ACCENT, fontweight="bold")
    if montage:
        axI.text(mont0 + mont_w / 2, y_top, f"real {args.cond_name} population",
                 ha="center", va="bottom", fontsize=10, color=POP, fontweight="bold")
    cn = args.channel_name or "channel"
    axI.set_title(f"Generative perturbation flow  ·  {cn}", fontsize=13.5, color=INK, pad=10)

    # ---- bottom: channel-intensity readout vs flow time ----
    curves = lipid_curves
    if args.curve_from and ch is not None:
        cf = sorted(glob.glob(os.path.join(args.curve_from, "sample_*_traj.npz")))
        curves = []
        for p in cf:
            tr = dict(np.load(p, allow_pickle=True))["trajectory"]
            tg = np.asarray(dict(np.load(p, allow_pickle=True))["t_grid"])
            curves.append((tg, np.array([fg_mean(tr[k], ch) for k in range(tr.shape[0])])))
    if curves:
        for tg, ys in curves:
            axC.plot(tg, ys, "-", lw=1.0, color=ACCENT, alpha=0.28, zorder=2)
        # bold mean across samples that share the modal grid length
        L = max(len(ys) for _, ys in curves)
        same = [ys for _, ys in curves if len(ys) == L]
        if same:
            tg0 = next(tg for tg, ys in curves if len(ys) == L)
            axC.plot(tg0, np.mean(same, 0), "-o", ms=4, lw=2.6, color=ACCENT,
                     zorder=4, label=f"generated (mean, n={len(same)})")
        ctrl = args.control_level if args.control_level is not None else np.mean([ys[0] for _, ys in curves])
        axC.axhline(ctrl, ls=":", color=MUTED, lw=1.5, zorder=3, label="real control (adlib)")
        if args.band_mean is not None:
            bm, bs = args.band_mean, (args.band_std or 0.0)
            axC.axhspan(bm - bs, bm + bs, color=POP, alpha=0.15, zorder=1)
            axC.axhline(bm, ls="--", color=POP, lw=1.8, zorder=3, label=f"real {args.cond_name} (pop.)")
        elif montage:
            band = np.array([fg_mean(mc, args.channel) for mc in montage])
            axC.axhspan(band.mean() - band.std(), band.mean() + band.std(), color=POP, alpha=0.15)
            axC.axhline(band.mean(), ls="--", color=POP, lw=1.8, label=f"real {args.cond_name}")
        axC.set_xlabel("generative flow time  t", fontsize=11, color=INK)
        axC.set_ylabel(f"{cn} intensity", fontsize=11, color=INK)
        axC.set_xlim(0, 1)
        axC.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=1)
        for s in ("top", "right"):
            axC.spines[s].set_visible(False)
        axC.tick_params(colors=INK, labelsize=9)
    else:
        axC.axis("off")

    plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {args.out}  ({n} row(s), {nf} strip frames, "
          f"{'montage' if montage else 'single-target'})")


if __name__ == "__main__":
    main()
