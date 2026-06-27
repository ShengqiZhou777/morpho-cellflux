"""Train MorphoDiff on exported Morpho-CellFlux benchmark data.

MorphoDiff = Stable Diffusion-style conditional diffusion in VAE latent space.
- VAE: stabilityai/sd-vae-ft-mse (frozen)
- UNet: UNet2DConditionModel with cross-attention on perturbation embeddings
- PerturbationEncoder: learned embedding per condition → (B, 77, 768)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from adapter_common import load_config, repo_path

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Perturbation Encoder: maps diet condition labels to SD cross-attention embeddings
# ---------------------------------------------------------------------------
class DietPerturbationEncoder(nn.Module):
    """Learned embedding for diet conditions → (batch, 77, 768) cross-attn tokens."""

    def __init__(self, num_conditions: int, embed_dim: int = 512, cross_attn_dim: int = 768, num_tokens: int = 77):
        super().__init__()
        self.condition_embed = nn.Embedding(num_conditions, embed_dim)
        self.proj = nn.Linear(embed_dim, cross_attn_dim)
        self.num_tokens = num_tokens

    def forward(self, class_labels: torch.LongTensor) -> torch.Tensor:
        # class_labels: (B,) ints
        emb = self.condition_embed(class_labels)          # (B, 512)
        emb = self.proj(emb)                               # (B, 768)
        emb = emb.unsqueeze(1).repeat(1, self.num_tokens, 1)  # (B, 77, 768)
        return emb


# ---------------------------------------------------------------------------
# Dataset: imagefolder → (image, condition_label)
# ---------------------------------------------------------------------------
def build_morphodiff_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 128,
    condition_name_to_idx: dict | None = None,
) -> tuple[DataLoader, dict]:
    """Load HuggingFace imagefolder dataset and return DataLoader + condition map."""
    from datasets import load_dataset

    ds = load_dataset("imagefolder", data_files={"train": os.path.join(data_dir, "imagefolder", split, "**")}, split="train")
    # imagefolder metadata.jsonl has "additional_feature" column from our export

    if condition_name_to_idx is None:
        # Build condition map from phendiff_class_to_idx.json (reused from export)
        c2i_path = data_dir / "phendiff_class_to_idx.json"
        if c2i_path.exists():
            condition_name_to_idx = json.loads(c2i_path.read_text())
        else:
            raise FileNotFoundError(f"Missing condition map: {c2i_path}")

    idx_to_name = {v: k for k, v in condition_name_to_idx.items()}

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = [train_transforms(img) for img in images]
        # Map condition string → int label
        conditions = examples.get("additional_feature", examples.get("label", []))
        labels = []
        for c in conditions:
            c_str = str(c).strip()
            if c_str in condition_name_to_idx:
                labels.append(condition_name_to_idx[c_str])
            else:
                labels.append(0)  # fallback
        examples["class_labels"] = labels
        return examples

    ds = ds.with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples]).contiguous().float()
        class_labels = torch.tensor([e["class_labels"] for e in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "class_labels": class_labels}

    loader = DataLoader(
        ds,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, condition_name_to_idx


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train_morphodiff(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    benchmark: str,
    epochs: int,
    batch_size: int,
    lr: float,
    mixed_precision: str,
    gradient_accumulation_steps: int,
    num_workers: int,
    seed: int,
    use_ema: bool,
    proba_uncond: float,
    vae_model_id: str,
) -> None:
    config = load_config(config_path)
    run_dir = output_dir / "external_checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # ---------- accelerator ----------
    accel_config = ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_config=accel_config,
    )

    # ---------- dataset ----------
    train_loader, condition_map = build_morphodiff_dataloader(
        data_dir, "train", batch_size, num_workers, image_size=128
    )
    num_conditions = len(condition_map)

    if accelerator.is_main_process:
        logger.info(f"MorphoDiff diet: {num_conditions} conditions, {len(train_loader.dataset)} train samples")
        logger.info(f"Condition map: {condition_map}")

    # ---------- VAE (frozen) ----------
    vae = AutoencoderKL.from_pretrained(vae_model_id)
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # ---------- UNet ----------
    # Use a compact UNet config suitable for 128×128 images (latent 16×16)
    unet = UNet2DConditionModel(
        sample_size=16,               # 128 / 8 (VAE downsample)
        in_channels=4,                # VAE latent channels
        out_channels=4,
        layers_per_block=2,
        block_out_channels=[128, 256, 512, 512],
        attention_head_dim=[4, 8, 8, 8],
        norm_num_groups=32,
        cross_attention_dim=768,      # SD-standard cross-attn dim
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
    )

    # ---------- Perturbation Encoder ----------
    pert_encoder = DietPerturbationEncoder(num_conditions=num_conditions, embed_dim=512, cross_attn_dim=768, num_tokens=77)

    # ---------- Noise Scheduler ----------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # ---------- Optimizer ----------
    trainable_params = list(unet.parameters()) + list(pert_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # ---------- EMA ----------
    if use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # ---------- LR Scheduler ----------
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    max_train_steps = epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_train_steps,
    )

    # ---------- Accelerator prepare ----------
    unet, pert_encoder, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, pert_encoder, optimizer, train_loader, lr_scheduler
    )
    if use_ema:
        ema_unet.to(accelerator.device)

    # Mixed precision weight dtype
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)

    # ---------- Trackers ----------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            f"morphodiff_{benchmark}",
            config={
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "num_conditions": num_conditions, "condition_map": condition_map,
                "vae": vae_model_id,
            },
        )

    # ---------- Training ----------
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** MorphoDiff Training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Batch size (per device) = {batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(epochs):
        unet.train()
        pert_encoder.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                clean_images = batch["pixel_values"].to(weight_dtype)
                class_labels = batch["class_labels"]

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(clean_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Perturbation embedding
                encoder_hidden_states = pert_encoder(class_labels).float()

                # CFG: random unconditional pass
                if proba_uncond > 0 and torch.rand(1).item() < proba_uncond:
                    encoder_hidden_states = torch.ones_like(encoder_hidden_states)  # naive embedding

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=train_loss)
                train_loss = 0.0

                # Checkpoint every 2000 steps
                if global_step % 2000 == 0 and accelerator.is_main_process:
                    ckpt_dir = run_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    _save_checkpoint(accelerator, unet, pert_encoder, vae, noise_scheduler,
                                     condition_map, ckpt_dir, use_ema, ema_unet if use_ema else None)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

        logger.info(f"Epoch {epoch} completed, global_step={global_step}")

    # ---------- Final save ----------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(accelerator, unet, pert_encoder, vae, noise_scheduler,
                         condition_map, final_dir, use_ema, ema_unet if use_ema else None)
        logger.info(f"Final model saved to {final_dir}")

    accelerator.end_training()


def _save_checkpoint(accelerator, unet, pert_encoder, vae, noise_scheduler,
                     condition_map, save_dir, use_ema, ema_unet):
    """Save full pipeline components for later inference."""
    unwrapped_unet = accelerator.unwrap_model(unet)
    if use_ema and ema_unet is not None:
        ema_unet.copy_to(unwrapped_unet.parameters())

    unwrapped_unet.save_pretrained(save_dir / "unet")
    accelerator.unwrap_model(pert_encoder).state_dict()
    torch.save(
        {"pert_encoder": accelerator.unwrap_model(pert_encoder).state_dict(),
         "condition_map": condition_map},
        save_dir / "pert_encoder.pt",
    )
    noise_scheduler.save_pretrained(save_dir / "scheduler")
    # VAE is already frozen, just save config reference
    (save_dir / "vae_model_id.txt").write_text("stabilityai/sd-vae-ft-mse")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--proba-uncond", type=float, default=0.1)
    p.add_argument("--vae-model-id", default="stabilityai/sd-vae-ft-mse")
    return p.parse_args()


def main():
    args = parse_args()
    train_morphodiff(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        seed=args.seed,
        use_ema=args.use_ema,
        proba_uncond=args.proba_uncond,
        vae_model_id=args.vae_model_id,
    )


if __name__ == "__main__":
    main()
