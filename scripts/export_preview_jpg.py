#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


DEFAULT_CHANNEL_NAMES = [
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


RGB_PRESETS = {
    "default": (0, 1, 2),
    "lipid_function": (5, 0, 1),
    "er_secretory": (9, 3, 17),
    "mito_autophagy": (8, 7, 14),
}


def rgb_presets_for_channel_count(channel_count: int) -> dict[str, tuple[int, int, int]]:
    if channel_count == 3:
        return {
            "default": (0, 1, 2),
            "lipid_function": (0, 1, 2),
        }
    return {
        name: channels
        for name, channels in RGB_PRESETS.items()
        if max(channels) < channel_count
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CellFlux preview NPZ files to JPG grids.")
    parser.add_argument("preview_npz", help="Preview file containing source, target, and generated arrays.")
    parser.add_argument("--out-dir", default=None, help="Output directory for JPG files.")
    parser.add_argument("--sample", type=int, default=0, help="Sample index in the preview batch.")
    parser.add_argument(
        "--channel-names",
        default=None,
        help="Optional comma-separated names for the 18 channels.",
    )
    parser.add_argument(
        "--rgb",
        default=None,
        help="Comma-separated channel indices for one RGB composite.",
    )
    parser.add_argument(
        "--preset",
        default="all",
        choices=["all", *RGB_PRESETS.keys()],
        help="Named RGB composite to export when --rgb is not set.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Nearest-neighbor display scale for 128x128 crops.",
    )
    return parser.parse_args()


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def make_rgb(arr: np.ndarray, channels: tuple[int, int, int]) -> Image.Image:
    planes = [normalize_channel(arr[idx]) for idx in channels]
    return Image.fromarray(np.stack(planes, axis=-1), mode="RGB")


def make_channel_grid(arr: np.ndarray, names: list[str], scale: int) -> Image.Image:
    channels, height, width = arr.shape
    cols = 6
    rows = int(np.ceil(channels / cols))
    label_h = 18
    tile_w = width * scale
    tile_h = height * scale + label_h
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for ch in range(channels):
        row, col = divmod(ch, cols)
        gray = Image.fromarray(normalize_channel(arr[ch]), mode="L").resize(
            (tile_w, height * scale),
            Image.Resampling.NEAREST,
        )
        rgb = Image.merge("RGB", (gray, gray, gray))
        x = col * tile_w
        y = row * tile_h
        canvas.paste(rgb, (x, y + label_h))
        label = names[ch] if ch < len(names) else f"ch{ch:02d}"
        draw.text((x + 4, y + 3), f"{ch:02d} {label}", fill=(0, 0, 0), font=font)
    return canvas


def stack_triptych(source: Image.Image, generated: Image.Image, target: Image.Image, scale: int) -> Image.Image:
    width, height = source.size
    label_h = 22
    gap = 10
    canvas = Image.new("RGB", (width * 3 + gap * 2, height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (name, img) in enumerate(
        [("source control", source), ("generated", generated), ("target perturbation", target)]
    ):
        x = idx * (width + gap)
        draw.text((x + 4, 5), name, fill=(0, 0, 0), font=font)
        canvas.paste(img.resize((width, height), Image.Resampling.NEAREST), (x, label_h))
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)


def main() -> None:
    args = parse_args()
    preview_path = Path(args.preview_npz)
    out_dir = Path(args.out_dir) if args.out_dir else preview_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(preview_path)
    sample = int(args.sample)
    source = data["source"][sample]
    target = data["target"][sample]
    generated = data["generated"][sample]
    channel_count = int(source.shape[0])
    if args.channel_names:
        names = [item.strip() for item in args.channel_names.split(",")]
    elif "channel_names" in data:
        names = [str(item) for item in data["channel_names"].tolist()]
    else:
        names = DEFAULT_CHANNEL_NAMES[:channel_count]
    if len(names) != channel_count:
        raise ValueError(f"got {len(names)} channel names, expected {channel_count}")

    presets = rgb_presets_for_channel_count(channel_count)
    if args.rgb:
        rgb_specs = {"custom": tuple(int(item.strip()) for item in args.rgb.split(","))}
        if len(rgb_specs["custom"]) != 3:
            raise ValueError("--rgb must contain exactly three channel indices")
    elif args.preset == "all":
        rgb_specs = presets
    else:
        if args.preset not in presets:
            raise ValueError(
                f"preset {args.preset!r} is not valid for {channel_count} channels"
            )
        rgb_specs = {args.preset: presets[args.preset]}

    for name, arr in [("source", source), ("target", target), ("generated", generated)]:
        grid = make_channel_grid(arr, names, args.scale)
        grid.save(out_dir / f"sample_{sample:02d}_{name}_channels.jpg", quality=95)

    for preset, rgb_channels in rgb_specs.items():
        source_rgb = make_rgb(source, rgb_channels)
        target_rgb = make_rgb(target, rgb_channels)
        generated_rgb = make_rgb(generated, rgb_channels)
        triptych = stack_triptych(source_rgb, generated_rgb, target_rgb, args.scale)
        channels = "_".join(str(ch) for ch in rgb_channels)
        triptych.save(
            out_dir / f"sample_{sample:02d}_{preset}_{channels}_triptych.jpg",
            quality=95,
        )
    print(f"wrote JPG previews to {out_dir}")


if __name__ == "__main__":
    main()
