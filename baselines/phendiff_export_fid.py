"""Export PhenDiff generations to the shared CellFlux eval layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


PHENDIFF_ROOT = repo_path("baselines/external/phendiff")
sys.path.insert(0, str(PHENDIFF_ROOT))

from src.pipeline_conditional_ddim.pipeline_conditionial_ddim import ConditionalDDIMPipeline  # noqa: E402


def safe_condition(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def load_class_to_idx(data_dir: Path) -> dict[str, int]:
    path = data_dir / "phendiff_class_to_idx.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing PhenDiff class map: {path}. Run export_all_baseline_data.sh first.")
    return json.loads(path.read_text())


def start_tensor(
    image_dir: str,
    sample_key: str,
    channels: list[int],
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    arr = panel_array(image_dir, sample_key, channels)
    img = Image.fromarray((arr * 255).round().astype(np.uint8), mode="RGB")
    if img.size != size:
        img = img.resize(size, resample=Image.BILINEAR)
    x = np.asarray(img).astype("float32") / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return (x * 2.0 - 1.0).to(device)


def export_phendiff(
    config_path: Path,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    benchmark: str,
    split: str,
    seed: int,
    max_samples: int | None,
    num_inference_steps: int,
    guidance: float | None,
    frac_diffusion_skipped: float,
    device_name: str,
) -> None:
    config = load_config(config_path)
    channels = channels_from_config(config)
    df = read_index(config["data_index_path"])
    targets, trt2ctrl = build_pairs(df, split=split, seed=seed, max_samples=max_samples)
    class_to_idx = load_class_to_idx(data_dir)

    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    pipe = ConditionalDDIMPipeline.from_pretrained(checkpoint)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    sample_size = pipe.unet.config.sample_size
    if isinstance(sample_size, int):
        size = (sample_size, sample_size)
    else:
        size = (int(sample_size[1]), int(sample_size[0]))

    generator = torch.Generator(device=device).manual_seed(seed)
    written = 0
    skipped: list[str] = []
    for _, row in targets.iterrows():
        target_id = str(row["SAMPLE_KEY"])
        condition = str(row["CPD_NAME"])
        class_name = safe_condition(condition)
        if class_name not in class_to_idx:
            skipped.append(f"{target_id}: unknown class {condition}")
            continue
        control_id = trt2ctrl[target_id]
        try:
            x0 = start_tensor(config["image_path"], control_id, channels, size=size, device=device)
        except FileNotFoundError as exc:
            skipped.append(f"{target_id}: {exc}")
            continue

        label = torch.tensor([class_to_idx[class_name]], device=device).long()
        with torch.no_grad():
            image = pipe(
                class_labels=label,
                class_emb=None,
                w=guidance,
                generator=generator,
                num_inference_steps=num_inference_steps,
                output_type="numpy",
                start_image=x0,
                add_forward_noise_to_image=True,
                frac_diffusion_skipped=frac_diffusion_skipped,
            ).images[0]
        write_fid_image(output_dir, condition, target_id, image)
        written += 1

    args = {
        "baseline_method": "PhenDiff",
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
        "num_inference_steps": num_inference_steps,
        "guidance": guidance,
        "frac_diffusion_skipped": frac_diffusion_skipped,
        "written": written,
        "skipped": skipped,
    }
    write_eval_contract(output_dir, trt2ctrl, args)
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"PhenDiff wrote {written} samples to {output_dir / 'fid_samples' / 'epoch-0'}")
    if skipped:
        print(f"PhenDiff skipped {len(skipped)} samples; see {output_dir / 'skipped.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="PhenDiff full_pipeline_save directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--frac-diffusion-skipped", type=float, default=0.55)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_phendiff(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        checkpoint=repo_path(args.checkpoint),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        num_inference_steps=args.num_inference_steps,
        guidance=args.guidance,
        frac_diffusion_skipped=args.frac_diffusion_skipped,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
