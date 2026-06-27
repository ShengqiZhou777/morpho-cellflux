#!/usr/bin/env python3
"""VRAM calibrator: find optimal batch size for each baseline method.

Usage:
  python baselines/calibrate_batch_size.py <method> [--target-vram-gb 22] [--min-bs 4] [--max-bs 128]

Methods: phendiff, impa, stargan, morphodiff

Strategy:
  1. Load the model architecture specific to each method
  2. Run synthetic forward+backward with increasing batch sizes
  3. Measure peak VRAM via torch.cuda.max_memory_allocated()
  4. Return the largest batch size that stays under target VRAM

This is model-only measurement without dataset I/O overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def repo_path(rel: str) -> Path:
    return Path(__file__).resolve().parent.parent / rel


def vram_used_gb() -> float:
    """Return currently allocated GPU memory in GB (via nvidia-smi, most reliable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return float(out) / 1024.0
    except Exception:
        return 0.0


def free_vram_gb() -> float:
    """Return free GPU memory in GB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return float(out) / 1024.0
    except Exception:
        return 0.0


def calibrate_model(
    model_factory,
    input_factory,
    optimizer_factory,
    device: torch.device,
    target_vram_gb: float,
    min_bs: int,
    max_bs: int,
    steps: int = 5,
) -> int:
    """Binary-search batch size for a given model architecture.

    Args:
        model_factory: fn(bs) -> dict of nn.Module
        input_factory: fn(bs, device) -> dict of tensors (model inputs)
        optimizer_factory: fn(models) -> optimizer
        target_vram_gb: target VRAM usage ceiling
        min_bs, max_bs: search range
        steps: how many fwd+bwd steps per calibration point

    Returns:
        Best batch size that stays under target_vram_gb.
    """
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    target_bytes = target_vram_gb * 1024**3
    safety_margin = 0.90  # leave 10% headroom for allocator overhead

    def test_bs(bs: int) -> tuple[bool, float]:
        """Return (fits, peak_vram_gb)."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            models = model_factory(bs)
            inputs = input_factory(bs, device)
            optimizer = optimizer_factory(models)

            for _ in range(steps):
                optimizer.zero_grad()
                outputs = _forward(models, inputs)
                if isinstance(outputs, torch.Tensor):
                    loss = outputs.mean()
                elif isinstance(outputs, dict):
                    loss = sum(v.mean() for v in outputs.values() if isinstance(v, torch.Tensor))
                else:
                    loss = sum(o.mean() for o in outputs if isinstance(o, torch.Tensor))
                loss.backward()
                optimizer.step()

            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_gb = peak_bytes / (1024**3)
            fits = peak_bytes < target_bytes * safety_margin

            # Cleanup
            del models, inputs, optimizer, outputs, loss
            torch.cuda.empty_cache()

            return fits, peak_gb

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False, float("inf")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False, float("inf")
            raise

    best_bs = min_bs
    best_vram = 0.0

    # Try candidate batch sizes from max down to min
    candidates = [max_bs, 64, 48, 40, 32, 24, 20, 16, 12, 8, 6, 4]
    candidates = sorted(set(c for c in candidates if min_bs <= c <= max_bs), reverse=True)

    for bs in candidates:
        fits, peak = test_bs(bs)
        print(f"  BS={bs:3d}: {'FITS' if fits else 'OOM'}, peak={peak:.1f} GB, free={free_vram_gb():.1f} GB")
        if fits:
            if peak > best_vram:
                best_bs = bs
                best_vram = peak
            # If we found one that uses >70% of target, we're good
            if peak > target_vram_gb * 0.70:
                break
        else:
            # OOM at high BS, continue to lower
            pass

    return best_bs


def _forward(models: dict, inputs: dict):
    """Model-specific forward pass dispatcher."""
    method = models.get("_method", "unknown")

    if method == "phendiff":
        unet = models["unet"]
        return unet(sample=inputs["x"], timestep=inputs["t"], class_labels=inputs["labels"]).sample

    elif method == "impa":
        gen = models["generator"]
        mapping = models["mapping_network"]
        style_encoder = models.get("style_encoder")
        discriminator = models["discriminator"]

        z_emb = inputs["z_emb"]
        y = inputs["y"]
        x_real = inputs["x_real"]
        x_ctrl = inputs["x_ctrl"]
        z = inputs["z"]

        style = mapping(z_emb, y, None)
        if style_encoder is not None:
            style = style_encoder(style)
        _, x_fake = gen(x_ctrl, style)
        d_real = discriminator(x_real, y)
        d_fake = discriminator(x_fake.detach(), y)
        return {"d_real": d_real, "d_fake": d_fake, "x_fake": x_fake}

    elif method == "stargan":
        gen = models["generator"]
        disc = models["discriminator"]
        x = inputs["x"]
        c = inputs["c"]
        c_trg = inputs["c_trg"]

        x_fake = gen(x, c_trg)
        d_src = disc(x, c)
        d_fake = disc(x_fake.detach(), c_trg)
        return {"d_src": d_src, "d_fake": d_fake, "x_fake": x_fake}

    elif method == "morphodiff":
        vae = models["vae"]
        unet = models["unet"]
        pert_enc = models["pert_encoder"]

        x = inputs["x"]
        labels = inputs["labels"]
        with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
        noisy_latents = models["scheduler"].add_noise(latents, noise, timesteps)
        encoder_hidden_states = pert_enc(labels)
        return unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────
# Method-specific model factories
# ──────────────────────────────────────────────

def phendiff_factory(bs: int) -> dict:
    from diffusers import DDIMScheduler
    sys.path.insert(0, str(repo_path("baselines/external/phendiff/src")))
    from cond_unet_2d import CustomCondUNet2DModel

    model = CustomCondUNet2DModel(
        sample_size=128, in_channels=3, out_channels=3,
        layers_per_block=2, block_out_channels=[64, 128, 256],
        down_block_types=["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
        up_block_types=["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
        num_class_embeds=3,
    )
    return {"unet": model, "_method": "phendiff"}


def phendiff_input_factory(bs: int, device: torch.device) -> dict:
    return {
        "x": torch.randn(bs, 3, 128, 128, device=device),
        "t": torch.randint(0, 1000, (bs,), device=device).long(),
        "labels": torch.randint(0, 3, (bs,), device=device).long(),
    }


def impa_factory(bs: int) -> dict:
    sys.path.insert(0, str(repo_path("baselines/external/impa")))
    from IMPA.model import build_model
    from argparse import Namespace

    args = Namespace(
        img_size=128, latent_dim=512, hidden_dim=256, style_dim=64,
        stochastic=True, z_dimension=8, dim_in=64, n_channels=3,
        multimodal=False, batch_correction=False,
        modality_list=["Gene"], encode_rdkit=True,
        num_layers_mapping_net=1,
    )
    nets = build_model(args, 3, "cuda" if torch.cuda.is_available() else "cpu", multimodal=False, batch_correction=False, modality_list=["Gene"], latent_dim=512)
    nets["_method"] = "impa"
    return nets


def impa_input_factory(bs: int, device: torch.device) -> dict:
    latent_dim = 512
    return {
        "x_real": torch.randn(bs, 3, 128, 128, device=device),
        "x_ctrl": torch.randn(bs, 3, 128, 128, device=device),
        "y": torch.randint(0, 3, (bs,), device=device).long(),
        "z_emb": torch.randn(bs, latent_dim, device=device),
        "z": torch.randn(bs, 8, device=device),
    }


def stargan_factory(bs: int) -> dict:
    sys.path.insert(0, str(repo_path("baselines/external/stargan")))
    from model import Generator, Discriminator

    gen = Generator(g_conv_dim=64, c_dim=3, repeat_num=6)
    disc = Discriminator(image_size=128, c_dim=3, repeat_num=6)
    return {"generator": gen, "discriminator": disc, "_method": "stargan"}


def stargan_input_factory(bs: int, device: torch.device) -> dict:
    return {
        "x": torch.randn(bs, 3, 128, 128, device=device),
        "c": torch.randint(0, 3, (bs,), device=device).long(),
        "c_trg": torch.randint(0, 3, (bs,), device=device).long(),
    }


def morphodiff_factory(bs: int) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    sys.path.insert(0, str(repo_path("baselines")))
    from morphodiff_train import DietPerturbationEncoder

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=2, block_out_channels=[128, 256, 512, 512],
        attention_head_dim=[4, 8, 8, 8], norm_num_groups=32,
        cross_attention_dim=768,
        down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    )
    pert_enc = DietPerturbationEncoder(num_conditions=3)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon")
    return {"vae": vae, "unet": unet, "pert_encoder": pert_enc, "scheduler": scheduler, "_method": "morphodiff"}


def morphodiff_input_factory(bs: int, device: torch.device) -> dict:
    return {
        "x": torch.randn(bs, 3, 128, 128, device=device),
        "labels": torch.randint(0, 3, (bs,), device=device).long(),
    }


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

METHODS = {
    "phendiff": (phendiff_factory, phendiff_input_factory, lambda m: torch.optim.Adam(m["unet"].parameters(), lr=1e-4)),
    "impa": (impa_factory, impa_input_factory, lambda m: torch.optim.Adam(
        list(m["generator"].parameters()) + list(m["discriminator"].parameters()) + list(m["mapping_network"].parameters()), lr=1e-4)),
    "stargan": (stargan_factory, stargan_input_factory, lambda m: torch.optim.Adam(
        list(m["generator"].parameters()) + list(m["discriminator"].parameters()), lr=1e-4)),
    "morphodiff": (morphodiff_factory, morphodiff_input_factory, lambda m: torch.optim.Adam(
        list(m["unet"].parameters()) + list(m["pert_encoder"].parameters()), lr=1e-4)),
}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=list(METHODS))
    parser.add_argument("--target-vram-gb", type=float, default=22.0)
    parser.add_argument("--min-bs", type=int, default=4)
    parser.add_argument("--max-bs", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    target = min(args.target_vram_gb, total_gb - 2.0)  # leave 2GB for framework overhead

    if not args.json:
        print(f"=== Calibrating {args.method} ===")
        print(f"  GPU: {torch.cuda.get_device_name(device)} ({total_gb:.0f} GB)")
        print(f"  Target VRAM: {target:.1f} GB")
        print(f"  Free VRAM: {free_vram_gb():.1f} GB")

    model_factory, input_factory, opt_factory = METHODS[args.method]
    best_bs = calibrate_model(
        model_factory, input_factory, opt_factory,
        device, target, args.min_bs, args.max_bs, steps=5,
    )

    if args.json:
        print(json.dumps({"method": args.method, "batch_size": best_bs, "target_vram_gb": target, "total_vram_gb": total_gb}))
    else:
        print(f"\n>>> Recommended batch_size for {args.method}: {best_bs} (target VRAM: {target:.1f} GB / {total_gb:.0f} GB)")


if __name__ == "__main__":
    main()
