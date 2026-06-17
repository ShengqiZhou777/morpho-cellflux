#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_NAMES = [
    "Alb",
    "polyT",
    "rRNA",
    "M6PR",
    "CathB",
    "Perilipin",
    "Sqstm1",
    "LC3b",
    "TOMM20",
    "Calreticulin",
    "pS6RP",
    "NaKATPase",
    "SNAP23",
    "TOM70",
    "Rab7",
    "mtRNA",
    "Vimentin",
    "Gapdh",
]

FOCUS_CHANNELS = [0, 5, 7, 8, 14]
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose source/generated/target CellFlux preview NPZ files."
    )
    parser.add_argument(
        "preview_npz",
        nargs="+",
        help="One or more preview NPZ files containing source, generated, target arrays.",
    )
    parser.add_argument(
        "--out",
        default="outputs/diagnostics/preview_diagnostics",
        help="Output path prefix. Writes .json and .md.",
    )
    parser.add_argument(
        "--channels",
        default="0,5,7,8,14",
        help="Comma-separated channels for focused morphology metrics.",
    )
    parser.add_argument(
        "--channel-names",
        default=None,
        help=(
            "Optional comma-separated names for --channels. Use this for "
            "panel previews whose channel indices are not the original "
            "18-channel indices."
        ),
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.03,
        help="Foreground pixel fraction for puncta top-mass metrics.",
    )
    return parser.parse_args()


def cell_mask(image: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    return image.max(axis=0) > float(threshold)


def mean_filter2d(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    pad = kernel // 2
    padded = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(1, 2))
    return windows.mean(axis=(-1, -2))


def dog(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    return image - mean_filter2d(image, kernel=kernel)


def masked_values(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = channel[mask]
    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def safe_std(values: np.ndarray) -> float:
    return float(values.std()) if values.size else float("nan")


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = a[:, mask] - b[:, mask]
    return float(np.mean(diff * diff)) if diff.size else float("nan")


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(a[:, mask] - b[:, mask])
    return float(np.mean(diff)) if diff.size else float("nan")


def pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a[:, mask].reshape(-1)
    y = b[:, mask].reshape(-1)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > EPS else float("nan")


def top_mass(channel: np.ndarray, mask: np.ndarray, fraction: float) -> float:
    values = masked_values(channel, mask)
    if values.size == 0:
        return float("nan")
    k = max(1, int(round(values.size * float(fraction))))
    top = np.partition(values, values.size - k)[-k:]
    return float(top.mean())


def count_proxy(channel: np.ndarray, mask: np.ndarray, target_channel: np.ndarray) -> float:
    target_values = masked_values(target_channel, mask)
    values = masked_values(channel, mask)
    if values.size == 0 or target_values.size == 0:
        return float("nan")
    threshold = np.percentile(target_values, 97.0)
    return float(np.mean(values >= threshold))


def channel_stats(
    source: np.ndarray,
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    channels: list[int],
    channel_names: dict[int, str],
    top_fraction: float,
) -> list[dict[str, Any]]:
    source_dog = dog(source)
    generated_dog = dog(generated)
    target_dog = dog(target)
    rows: list[dict[str, Any]] = []
    for channel in channels:
        src_values = masked_values(source[channel], mask)
        gen_values = masked_values(generated[channel], mask)
        tgt_values = masked_values(target[channel], mask)
        src_dog_values = masked_values(source_dog[channel], mask)
        gen_dog_values = masked_values(generated_dog[channel], mask)
        tgt_dog_values = masked_values(target_dog[channel], mask)

        gen_mean = safe_mean(gen_values)
        tgt_mean = safe_mean(tgt_values)
        gen_std = safe_std(gen_values)
        tgt_std = safe_std(tgt_values)
        gen_dog_energy = safe_mean(np.abs(gen_dog_values))
        tgt_dog_energy = safe_mean(np.abs(tgt_dog_values))
        gen_top = top_mass(generated[channel], mask, top_fraction)
        tgt_top = top_mass(target[channel], mask, top_fraction)

        rows.append(
            {
                "channel": channel,
                "name": channel_names[channel],
                "mean_ratio_gen_target": gen_mean / (tgt_mean + EPS),
                "std_ratio_gen_target": gen_std / (tgt_std + EPS),
                "dog_energy_ratio_gen_target": gen_dog_energy / (tgt_dog_energy + EPS),
                "top_mass_ratio_gen_target": gen_top / (tgt_top + EPS),
                "target_threshold_count_ratio": count_proxy(
                    generated[channel],
                    mask,
                    target[channel],
                )
                / (count_proxy(target[channel], mask, target[channel]) + EPS),
                "dog_energy_ratio_source_target": safe_mean(np.abs(src_dog_values))
                / (tgt_dog_energy + EPS),
                "top_mass_ratio_source_target": top_mass(source[channel], mask, top_fraction)
                / (tgt_top + EPS),
            }
        )
    return rows


def summarize_preview(
    path: Path,
    channels: list[int],
    channel_names: dict[int, str],
    top_fraction: float,
) -> dict[str, Any]:
    data = np.load(path)
    source = data["source"].astype(np.float32)
    generated = data["generated"].astype(np.float32)
    target = data["target"].astype(np.float32)
    if source.shape != generated.shape or source.shape != target.shape:
        raise ValueError(f"array shape mismatch in {path}")

    sample_summaries: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    for sample_idx in range(source.shape[0]):
        src = source[sample_idx]
        gen = generated[sample_idx]
        tgt = target[sample_idx]
        mask = cell_mask(tgt)
        gen_target_mse = masked_mse(gen, tgt, mask)
        gen_source_mse = masked_mse(gen, src, mask)
        source_target_mse = masked_mse(src, tgt, mask)
        sample_rows = channel_stats(
            src,
            gen,
            tgt,
            mask,
            channels,
            channel_names=channel_names,
            top_fraction=top_fraction,
        )
        focus_rows.extend(sample_rows)
        sample_summaries.append(
            {
                "sample": sample_idx,
                "mask_pixels": int(mask.sum()),
                "gen_target_mse": gen_target_mse,
                "gen_source_mse": gen_source_mse,
                "source_target_mse": source_target_mse,
                "gen_target_mae": masked_mae(gen, tgt, mask),
                "gen_source_mae": masked_mae(gen, src, mask),
                "gen_target_pearson": pearson(gen, tgt, mask),
                "gen_source_pearson": pearson(gen, src, mask),
                "mse_closer_to_target": gen_target_mse < gen_source_mse,
                "relative_target_progress": 1.0
                - (gen_target_mse / (source_target_mse + EPS)),
                "mean_dog_energy_ratio": safe_mean(
                    np.asarray([row["dog_energy_ratio_gen_target"] for row in sample_rows])
                ),
                "mean_top_mass_ratio": safe_mean(
                    np.asarray([row["top_mass_ratio_gen_target"] for row in sample_rows])
                ),
                "mean_std_ratio": safe_mean(
                    np.asarray([row["std_ratio_gen_target"] for row in sample_rows])
                ),
            }
        )

    def aggregate(key: str) -> float:
        return safe_mean(np.asarray([row[key] for row in sample_summaries], dtype=np.float64))

    channel_aggregates: list[dict[str, Any]] = []
    for channel in channels:
        rows = [row for row in focus_rows if row["channel"] == channel]
        channel_aggregates.append(
            {
                "channel": channel,
                "name": channel_names[channel],
                "std_ratio_gen_target": safe_mean(
                    np.asarray([row["std_ratio_gen_target"] for row in rows])
                ),
                "dog_energy_ratio_gen_target": safe_mean(
                    np.asarray([row["dog_energy_ratio_gen_target"] for row in rows])
                ),
                "top_mass_ratio_gen_target": safe_mean(
                    np.asarray([row["top_mass_ratio_gen_target"] for row in rows])
                ),
                "target_threshold_count_ratio": safe_mean(
                    np.asarray([row["target_threshold_count_ratio"] for row in rows])
                ),
                "dog_energy_ratio_source_target": safe_mean(
                    np.asarray([row["dog_energy_ratio_source_target"] for row in rows])
                ),
            }
        )

    return {
        "path": str(path),
        "run": path.parents[1].name,
        "step": path.stem.replace("step_", ""),
        "samples": int(source.shape[0]),
        "channels": channels,
        "aggregate": {
            "gen_target_mse": aggregate("gen_target_mse"),
            "gen_source_mse": aggregate("gen_source_mse"),
            "source_target_mse": aggregate("source_target_mse"),
            "relative_target_progress": aggregate("relative_target_progress"),
            "gen_target_pearson": aggregate("gen_target_pearson"),
            "gen_source_pearson": aggregate("gen_source_pearson"),
            "mean_dog_energy_ratio": aggregate("mean_dog_energy_ratio"),
            "mean_top_mass_ratio": aggregate("mean_top_mass_ratio"),
            "mean_std_ratio": aggregate("mean_std_ratio"),
        },
        "channel_aggregates": channel_aggregates,
        "sample_summaries": sample_summaries,
    }


def markdown_report(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Preview Diagnostics")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(
        "| run | step | gen-target MSE | gen-source MSE | target progress | "
        "dog energy ratio | top mass ratio | std ratio | corr target | corr source |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        agg = result["aggregate"]
        lines.append(
            f"| {result['run']} | {result['step']} | "
            f"{agg['gen_target_mse']:.5f} | {agg['gen_source_mse']:.5f} | "
            f"{agg['relative_target_progress']:.3f} | "
            f"{agg['mean_dog_energy_ratio']:.3f} | "
            f"{agg['mean_top_mass_ratio']:.3f} | "
            f"{agg['mean_std_ratio']:.3f} | "
            f"{agg['gen_target_pearson']:.3f} | {agg['gen_source_pearson']:.3f} |"
        )
    lines.append("")
    lines.append("## Focus Channel Ratios")
    lines.append("")
    lines.append(
        "Ratios are generated/target unless the column name says source/target. "
        "Values far below 1 mean generated is smoother or weaker than target."
    )
    lines.append("")
    lines.append(
        "| run | step | channel | std | DoG energy | top mass | target peak count | source DoG |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for result in results:
        for row in result["channel_aggregates"]:
            lines.append(
                f"| {result['run']} | {result['step']} | {row['name']} | "
                f"{row['std_ratio_gen_target']:.3f} | "
                f"{row['dog_energy_ratio_gen_target']:.3f} | "
                f"{row['top_mass_ratio_gen_target']:.3f} | "
                f"{row['target_threshold_count_ratio']:.3f} | "
                f"{row['dog_energy_ratio_source_target']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    channels = [int(item.strip()) for item in args.channels.split(",") if item.strip()]
    if args.channel_names:
        names = [item.strip() for item in args.channel_names.split(",") if item.strip()]
        if len(names) != len(channels):
            raise ValueError("--channel-names must have the same length as --channels")
        channel_names = dict(zip(channels, names))
    else:
        channel_names = {channel: CHANNEL_NAMES[channel] for channel in channels}
    results = [
        summarize_preview(Path(path), channels, channel_names, float(args.top_fraction))
        for path in args.preview_npz
    ]

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(results, indent=2))
    out_prefix.with_suffix(".md").write_text(markdown_report(results))
    print(f"wrote {out_prefix.with_suffix('.json')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
