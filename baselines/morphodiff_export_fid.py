"""Export MorphoDiff generations to the shared CellFlux eval layout.

Inference: control image → VAE encode → DDIM inversion (partial) →
DDIM denoising with CFG → VAE decode → generated image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm

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


class DietPerturbationEncoder(nn.Module):
    """Mirror of the training encoder for inference."""

    def __init__(self, num_conditions: int, embed_dim: int = 512, cross_attn_dim: int = 768, num_tokens: int = 77):
        super().__init__()
        self.condition_embed = nn.Embedding(num_conditions, embed_dim)
        self.proj = nn.Linear(embed_dim, cross_attn_dim)
        self.num_tokens = num_tokens

    def forward(self, class_labels: torch.LongTensor) -> torch.Tensor:
        emb = self.condition_embed(class_labels)
        emb = self.proj(emb)
        emb = emb.unsqueeze(1).repeat(1, self.num_tokens, 1)
        return emb


def control_tensor(image_dir: str, sample_key: str, channels: list[int], img_size: int, device: torch.device) -> torch.Tensor:
    arr = panel_array(image_dir, sample_key, channels).astype("float32")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (x * 2.0 - 1.0).to(device)  # → [-1, 1]


@torch.no_grad()
def export_morphodiff(
    config_path: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    benchmark: str,
    split: str,
    seed: int,
    max_samples: int | None,
    guidance: float,
    frac_diffusion_skipped: float,
    num_inference_steps: int,
    image_size: int,
    device_name: str,
) -> None:
    config = load_config(config_path)
    channels = channels_from_config(config)
    df = read_index(config["data_index_path"])
    targets, trt2ctrl = build_pairs(df, split=split, seed=seed, max_samples=max_samples)
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")

    # Load condition map
    pert_ckpt = torch.load(checkpoint_dir / "pert_encoder.pt", map_location=device)
    condition_map = pert_ckpt["condition_map"]
    idx_to_condition = {v: k for k, v in condition_map.items()}
    num_conditions = len(condition_map)
    class_to_idx = json.loads((data_dir / "phendiff_class_to_idx.json").read_text())

    # Load VAE
    vae_id = (checkpoint_dir / "vae_model_id.txt").read_text().strip()
    vae = AutoencoderKL.from_pretrained(vae_id).to(device)
    vae.eval()

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(checkpoint_dir / "unet").to(device)
    unet.eval()

    # Load Perturbation Encoder
    pert_encoder = DietPerturbationEncoder(num_conditions=num_conditions).to(device)
    pert_encoder.load_state_dict(pert_ckpt["pert_encoder"])
    pert_encoder.eval()

    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained(checkpoint_dir / "scheduler")
    scheduler.set_timesteps(num_inference_steps)

    torch.manual_seed(seed)
    written = 0
    skipped: list[str] = []

    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="MorphoDiff export"):
        target_id = str(row["SAMPLE_KEY"])
        condition = str(row["CPD_NAME"])
        if condition not in class_to_idx:
            skipped.append(f"{target_id}: unknown class {condition}")
            continue
        try:
            x_ctrl = control_tensor(config["image_path"], trt2ctrl[target_id], channels, image_size, device)
        except FileNotFoundError as exc:
            skipped.append(f"{target_id}: {exc}")
            continue

        # VAE encode control image
        latent = vae.encode(x_ctrl).latent_dist.sample()
        latent = latent * vae.config.scaling_factor

        # Prepare timesteps with skip
        init_timestep = int(scheduler.config.num_train_timesteps * (1 - frac_diffusion_skipped))
        timesteps = scheduler.timesteps[scheduler.timesteps <= init_timestep].to(device)

        # Add noise at init_timestep
        noise = torch.randn_like(latent)
        latent = scheduler.add_noise(latent, noise, timesteps[0].repeat(1))

        # Target condition embedding
        target_idx = class_to_idx[condition]
        cond_emb = pert_encoder(torch.tensor([target_idx], device=device).long()).float()

        # Unconditional embedding for CFG
        uncond_emb = torch.ones_like(cond_emb)

        # DDIM denoising with CFG
        for t in timesteps:
            # Conditional
            cond_pred = unet(latent, t.repeat(1), cond_emb, return_dict=False)[0]
            # Unconditional
            uncond_pred = unet(latent, t.repeat(1), uncond_emb, return_dict=False)[0]
            # CFG: Imagen equation
            guided_pred = uncond_pred + guidance * (cond_pred - uncond_pred)

            latent = scheduler.step(guided_pred, t, latent).prev_sample

        # VAE decode
        latent = latent / vae.config.scaling_factor
        image = vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        write_fid_image(output_dir, condition, target_id, image_np)
        written += 1

    args_info = {
        "baseline_method": "MorphoDiff",
        "benchmark": benchmark,
        "use_initial": 1,
        "channels": channels,
        "image_path": config["image_path"],
        "data_index_path": config["data_index_path"],
        "split": split, "seed": seed, "max_samples": max_samples,
        "checkpoint": str(checkpoint_dir),
        "data_dir": str(data_dir),
        "guidance": guidance,
        "frac_diffusion_skipped": frac_diffusion_skipped,
        "num_inference_steps": num_inference_steps,
        "written": written, "skipped": skipped,
    }
    write_eval_contract(output_dir, trt2ctrl, args_info)
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"MorphoDiff wrote {written} samples to {output_dir / 'fid_samples' / 'epoch-0'}")
    if skipped:
        print(f"MorphoDiff skipped {len(skipped)} samples; see {output_dir / 'skipped.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--frac-diffusion-skipped", type=float, default=0.55)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    export_morphodiff(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        checkpoint_dir=repo_path(args.checkpoint),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        guidance=args.guidance,
        frac_diffusion_skipped=args.frac_diffusion_skipped,
        num_inference_steps=args.num_inference_steps,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
