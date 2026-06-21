"""Export StarGAN generations to the shared CellFlux eval layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapter_common import (
    build_pairs,
    channels_from_config,
    load_config,
    panel_array,
    read_index,
    repo_path,
    write_eval_contract,
    write_fid_image,
)


STARGAN_ROOT = repo_path("baselines/external/stargan")
sys.path.insert(0, str(STARGAN_ROOT))

from model import Generator  # noqa: E402


def safe_condition(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def load_class_to_idx(data_dir: Path) -> dict[str, int]:
    path = data_dir / "phendiff_class_to_idx.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing class map: {path}. Run export_all_baseline_data.sh first.")
    return json.loads(path.read_text())


def control_tensor(image_dir: str, sample_key: str, channels: list[int], img_size: int, device: torch.device) -> torch.Tensor:
    arr = panel_array(image_dir, sample_key, channels).astype("float32")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (x * 2.0 - 1.0).to(device)


def onehot(idx: int, dim: int, device: torch.device) -> torch.Tensor:
    y = torch.zeros(1, dim, device=device)
    y[0, idx] = 1.0
    return y


def export_stargan(
    config_path: Path,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    benchmark: str,
    split: str,
    seed: int,
    max_samples: int | None,
    image_size: int,
    g_conv_dim: int,
    g_repeat_num: int,
    device_name: str,
) -> None:
    config = load_config(config_path)
    channels = channels_from_config(config)
    df = read_index(config["data_index_path"])
    targets, trt2ctrl = build_pairs(df, split=split, seed=seed, max_samples=max_samples)
    class_to_idx = load_class_to_idx(data_dir)

    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    model = Generator(g_conv_dim, len(class_to_idx), g_repeat_num).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    written = 0
    skipped: list[str] = []
    for _, row in targets.iterrows():
        target_id = str(row["SAMPLE_KEY"])
        condition = str(row["CPD_NAME"])
        class_name = safe_condition(condition)
        if class_name not in class_to_idx:
            skipped.append(f"{target_id}: unknown class {condition}")
            continue
        try:
            x_ctrl = control_tensor(config["image_path"], trt2ctrl[target_id], channels, image_size, device)
        except FileNotFoundError as exc:
            skipped.append(f"{target_id}: {exc}")
            continue
        c_trg = onehot(class_to_idx[class_name], len(class_to_idx), device)
        with torch.no_grad():
            x_fake = model(x_ctrl, c_trg)
        image = ((x_fake.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) / 2.0).clip(0, 1)
        write_fid_image(output_dir, condition, target_id, image)
        written += 1

    args = {
        "baseline_method": "StarGAN",
        "benchmark": benchmark,
        "use_initial": 1,
        "channels": channels,
        "image_path": config["image_path"],
        "data_index_path": config["data_index_path"],
        "embedding_path": config["embedding_path"],
        "dataset_name": config.get("dataset_name", "perturbmulti"),
        "task_name": config.get("task_name"),
        "split": split,
        "seed": seed,
        "max_samples": max_samples,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "image_size": image_size,
        "g_conv_dim": g_conv_dim,
        "g_repeat_num": g_repeat_num,
        "written": written,
        "skipped": skipped,
    }
    write_eval_contract(output_dir, trt2ctrl, args)
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"StarGAN wrote {written} samples to {output_dir / 'fid_samples' / 'epoch-0'}")
    if skipped:
        print(f"StarGAN skipped {len(skipped)} samples; see {output_dir / 'skipped.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="StarGAN *-G.ckpt file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--g-conv-dim", type=int, default=64)
    parser.add_argument("--g-repeat-num", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_stargan(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        checkpoint=repo_path(args.checkpoint),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        image_size=args.image_size,
        g_conv_dim=args.g_conv_dim,
        g_repeat_num=args.g_repeat_num,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
