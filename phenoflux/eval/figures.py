"""Diet marker distribution figure from saved fid_samples.

This is the reproducible version of the Diet 5K marker-distribution check.
It compares generated images against real target npz crops for all generated
PNGs, and computes paired control gap-closure only for target ids that have a
treated->control mapping.

Usage:
  python scripts/diet_marker_distribution_figure.py \
    --run-dir outputs/runs/diet/fid5k \
    --epoch 12 \
    --out-dir outputs/figures/diet \
    --prefix diet_fid5k
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
CH_NAMES = [
    "Alb", "polyT", "rRNA", "M6PR", "CathB", "Perilipin", "Sqstm1", "LC3b",
    "TOMM20", "Calreticulin", "pS6RP", "Na/K-ATPase", "SNAP23", "TOM70", "Rab7",
    "mtRNA", "Vimentin", "Gapdh",
]
DEFAULT_CHANNELS = [9, 5, 8]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory containing fid_samples/")
    parser.add_argument("--epoch", default=None, help="Epoch number. Defaults to latest epoch-* dir.")
    parser.add_argument("--out-dir", default="outputs/figures/diet", help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument(
        "--image-path",
        default=None,
        help="Raw npz image directory. Defaults to args.json image_path.",
    )
    parser.add_argument(
        "--cap-per-condition",
        type=int,
        default=None,
        help="Optional deterministic cap per diet condition for balanced summaries.",
    )
    return parser.parse_args()


def load_run_config(run_dir):
    args_path = Path(run_dir) / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open() as f:
        return json.load(f)


def resolve_epoch_dir(run_dir, epoch):
    fid_root = Path(run_dir) / "fid_samples"
    if epoch is not None:
        epoch_dir = fid_root / f"epoch-{epoch}"
        if not epoch_dir.exists():
            raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")
        return epoch_dir

    epoch_dirs = sorted(
        fid_root.glob("epoch-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch-* dirs under {fid_root}")
    return epoch_dirs[-1]


def load_mapping(run_dir, epoch_dir):
    epoch_mapping = epoch_dir / "trt2ctrl_idx.json"
    global_mapping = Path(run_dir) / "fid_samples" / "trt2ctrl_idx.json"
    mapping_path = epoch_mapping if epoch_mapping.exists() else global_mapping
    if not mapping_path.exists():
        return {}, None
    with mapping_path.open() as f:
        return json.load(f), mapping_path


def fg_means(pixels):
    """Foreground mean per channel; pixels is [P,C] in [0,1]."""
    foreground = pixels.max(axis=1) > 0.05
    selected = pixels[foreground] if foreground.sum() > 20 else pixels
    return selected.mean(axis=0)


def npz_panel_means(cell_id, image_dir, channels):
    if not cell_id:
        return None
    path = image_dir / f"{cell_id}.npz"
    if not path.exists():
        return None
    panel = np.load(path)["x"][channels]
    return fg_means(panel.reshape(len(channels), -1).T)


def png_panel_means(path, n_channels):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return fg_means(arr.reshape(-1, n_channels))


def iter_rows(epoch_dir, image_dir, channels, mapping, cap_per_condition):
    rows = []
    for cond_dir in sorted(p for p in epoch_dir.iterdir() if p.is_dir()):
        pngs = sorted(cond_dir.glob("*.png"))
        if cap_per_condition is not None:
            pngs = pngs[:cap_per_condition]
        for png in pngs:
            target_id = png.stem
            control_id = mapping.get(target_id)
            gen = png_panel_means(png, len(channels))
            target = npz_panel_means(target_id, image_dir, channels)
            control = npz_panel_means(control_id, image_dir, channels) if control_id else None
            if target is None:
                continue
            rows.append(
                {
                    "condition": cond_dir.name,
                    "target_id": target_id,
                    "control_id": control_id or "",
                    "generated": gen,
                    "target": target,
                    "control": control,
                }
            )
    return rows


def summarize(rows, marker_names):
    summary_rows = []
    for condition in sorted({row["condition"] for row in rows}):
        cond_rows = [row for row in rows if row["condition"] == condition]
        for idx, marker in enumerate(marker_names):
            gen = np.array([row["generated"][idx] for row in cond_rows], dtype=float)
            target = np.array([row["target"][idx] for row in cond_rows], dtype=float)
            paired = [row for row in cond_rows if row["control"] is not None]
            if paired:
                control = np.array([row["control"][idx] for row in paired], dtype=float)
                gen_paired = np.array([row["generated"][idx] for row in paired], dtype=float)
                target_paired = np.array([row["target"][idx] for row in paired], dtype=float)
                w_gen_target = wasserstein_distance(gen_paired, target_paired)
                w_control_target = wasserstein_distance(control, target_paired)
                gap_closed = (
                    1.0 - w_gen_target / w_control_target
                    if w_control_target > 0
                    else float("nan")
                )
                control_mean = float(control.mean())
            else:
                w_gen_target = float("nan")
                w_control_target = float("nan")
                gap_closed = float("nan")
                control_mean = float("nan")
            summary_rows.append(
                {
                    "condition": condition,
                    "marker": marker,
                    "n_generated": len(cond_rows),
                    "n_paired_control": len(paired),
                    "generated_mean": float(gen.mean()),
                    "target_mean": float(target.mean()),
                    "control_mean": control_mean,
                    "generated_minus_target": float(gen.mean() - target.mean()),
                    "wasserstein_generated_target": float(w_gen_target),
                    "wasserstein_control_target": float(w_control_target),
                    "gap_closed": float(gap_closed),
                }
            )
    return summary_rows


def write_summary(summary_rows, out_csv, out_json, metadata):
    fieldnames = [
        "condition", "marker", "n_generated", "n_paired_control", "generated_mean",
        "target_mean", "control_mean", "generated_minus_target",
        "wasserstein_generated_target", "wasserstein_control_target", "gap_closed",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    with out_json.open("w") as f:
        json.dump({"metadata": metadata, "summary": summary_rows}, f, indent=2)


def plot_distributions(rows, marker_names, out_path, title):
    conditions = sorted({row["condition"] for row in rows})
    fig, axes = plt.subplots(
        len(conditions),
        len(marker_names),
        figsize=(4.2 * len(marker_names), 3.1 * len(conditions)),
        squeeze=False,
    )
    colors = {"control": "#6b7280", "generated": "#2563eb", "target": "#dc2626"}

    for i, condition in enumerate(conditions):
        cond_rows = [row for row in rows if row["condition"] == condition]
        for j, marker in enumerate(marker_names):
            ax = axes[i][j]
            control = [row["control"][j] for row in cond_rows if row["control"] is not None]
            generated = [row["generated"][j] for row in cond_rows]
            target = [row["target"][j] for row in cond_rows]
            values_for_bins = [*generated, *target, *(control or generated)]
            lo = min(values_for_bins)
            hi = max(values_for_bins)
            if hi <= lo:
                lo -= 1e-3
                hi += 1e-3
            bins = np.linspace(lo, hi, 36)
            if len(control) >= 5:
                ax.hist(
                    control,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.7,
                    color=colors["control"],
                    label="control",
                )
            ax.hist(
                generated,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.9,
                color=colors["generated"],
                label="generated",
            )
            ax.hist(
                target,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.9,
                color=colors["target"],
                label="target",
            )
            ax.set_title(f"{condition} / {marker}", fontsize=10)
            ax.set_xlabel("foreground mean intensity")
            ax.set_ylabel("density")
            if i == 0 and j == len(marker_names) - 1:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_shift(summary_rows, marker_names, out_path, title):
    conditions = sorted({row["condition"] for row in summary_rows})
    x = np.arange(len(marker_names))
    width = 0.24
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.4 * len(conditions), 3.6), squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        cond = [row for row in summary_rows if row["condition"] == condition]
        by_marker = {row["marker"]: row for row in cond}
        control = [by_marker[m]["control_mean"] for m in marker_names]
        generated = [by_marker[m]["generated_mean"] for m in marker_names]
        target = [by_marker[m]["target_mean"] for m in marker_names]
        ax.bar(x - width, control, width, label="control", color="#9ca3af")
        ax.bar(x, generated, width, label="generated", color="#3b82f6")
        ax.bar(x + width, target, width, label="target", color="#ef4444")
        ax.set_title(condition)
        ax.set_xticks(x)
        ax.set_xticklabels(marker_names, rotation=25, ha="right")
        ax.set_ylabel("foreground mean intensity")
        ax.set_ylim(bottom=0)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir)
    channels = config.get("channels") or DEFAULT_CHANNELS
    marker_names = [CH_NAMES[channel] for channel in channels]
    image_path = args.image_path or config.get("image_path") or "data/raw/diet/images"
    image_dir = Path(image_path)
    if not image_dir.is_absolute():
        image_dir = REPO_ROOT / image_dir
    epoch_dir = resolve_epoch_dir(run_dir, args.epoch)
    mapping, mapping_path = load_mapping(run_dir, epoch_dir)

    rows = iter_rows(epoch_dir, image_dir, channels, mapping, args.cap_per_condition)
    if not rows:
        raise RuntimeError(f"No usable generated PNGs found under {epoch_dir}")

    n_pngs = sum(1 for _ in epoch_dir.glob("*/*.png"))
    n_paired = sum(1 for row in rows if row["control"] is not None)
    mapping_complete = n_paired == len(rows)
    prefix = args.prefix or f"{run_dir.name}_{epoch_dir.name}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = summarize(rows, marker_names)
    metadata = {
        "run_dir": str(run_dir),
        "epoch_dir": str(epoch_dir),
        "image_dir": str(image_dir),
        "channels": channels,
        "markers": marker_names,
        "mapping_path": str(mapping_path) if mapping_path else None,
        "n_pngs_on_disk": n_pngs,
        "n_rows_used": len(rows),
        "n_mapping_entries": len(mapping),
        "n_paired_control_rows": n_paired,
        "mapping_complete_for_used_rows": mapping_complete,
        "cap_per_condition": args.cap_per_condition,
    }

    out_csv = out_dir / f"{prefix}_marker_distribution_summary.csv"
    out_json = out_dir / f"{prefix}_marker_distribution_summary.json"
    out_dist = out_dir / f"{prefix}_marker_distributions.png"
    out_shift = out_dir / f"{prefix}_mean_shift.png"
    write_summary(summary_rows, out_csv, out_json, metadata)
    plot_distributions(
        rows,
        marker_names,
        out_dist,
        f"{prefix}: Diet marker distributions ({', '.join(marker_names)})",
    )
    plot_mean_shift(
        summary_rows,
        marker_names,
        out_shift,
        f"{prefix}: foreground marker means",
    )

    if not mapping_complete:
        print(
            "warning: treated->control mapping is incomplete; "
            "distribution plots use all generated/target rows, paired gap_closed uses only mapped rows"
        )
    print(f"rows used: {len(rows)} / pngs on disk: {n_pngs}; paired controls: {n_paired}")
    print(f"saved: {out_dist}")
    print(f"saved: {out_shift}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
