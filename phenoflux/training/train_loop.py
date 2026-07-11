# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
import random
import time
from typing import Iterable
from phenoflux.training.dataloader import CellDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from phenoflux.models.ema import EMA
from phenoflux.training.data_utils import centered_noise
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from phenoflux.training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50

# --- PatchGAN Discriminator (used when --gan_weight > 0) ---
_DISCRIMINATOR: nn.Module | None = None
_DISC_OPTIMIZER: torch.optim.Optimizer | None = None


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN with spectral norm for stable adversarial training."""
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(base, base*2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(base*2, base*4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(base*4, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.net(x)


def skewed_timestep_sample(
    num_samples: int,
    device: torch.device,
    p_mean_shifts: torch.Tensor | None = None,
    cond_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    P_mean_base = -1.2
    P_std = 1.2
    if p_mean_shifts is not None and cond_indices is not None:
        P_mean = P_mean_base + p_mean_shifts[cond_indices]
    else:
        P_mean = P_mean_base
    # Split deterministic shift (gradient path) from random noise (no grad).
    # sigma = exp(randn*P_std) * exp(P_mean)  preserves LogNormal distribution
    # but gradients only flow through exp(P_mean), removing noise from the signal.
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma_random = (rnd_normal * P_std).exp().detach()
    sigma_deterministic = torch.as_tensor(P_mean, device=device).exp()
    sigma = sigma_random * sigma_deterministic
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def foreground_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    threshold: float,
    foreground_weight: float,
    background_weight: float,
) -> torch.Tensor:
    """MSE weighted toward marker-positive cell foreground.

    Perturb-Multi crops are sparse false-color marker readouts. The tensors are
    in [-1,1], so the dynamic mask is computed after mapping back to [0,1].
    """
    err = torch.pow(pred - target, 2)
    x0_raw = (x_0.detach() + 1.0) * 0.5
    x1_raw = (x_1.detach() + 1.0) * 0.5
    mask = ((x0_raw.amax(dim=1, keepdim=True) > threshold) |
            (x1_raw.amax(dim=1, keepdim=True) > threshold))
    weights = torch.where(
        mask,
        torch.as_tensor(foreground_weight, device=err.device, dtype=err.dtype),
        torch.as_tensor(background_weight, device=err.device, dtype=err.dtype),
    )
    weights = weights.expand_as(err)
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def my_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    datamodule: CellDataLoader,
    use_initial: int,
    wandb_run=None,
):
    gc.collect()
    model.train(True)
    _model = model.module if hasattr(model, 'module') else model
    _model = getattr(_model, 'model', _model)  # unwrap EMA if present
    _n_cond = getattr(_model, 'base_condition_dim', 0)
    # Per-condition loss tracking (only when p_mean_shifts is active)
    _p_mean_shifts = None  # microalgae: no per-condition p_mean shifts
    cond_loss_sums = None
    cond_loss_counts = None
    if _p_mean_shifts is not None:
        cond_loss_sums = torch.zeros(_n_cond, device=device)
        cond_loss_counts = torch.zeros(_n_cond, device=device)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    for data_iter_step, batch in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break
        
        x_real, y_trg, y_mod = batch['X'], batch['mols'], batch['y_id']
        x_real_ctrl, x_real_trt = x_real
        x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)
        y_trg = y_trg.long().to(device)            
        y_org = None 
        z_emb_trg = datamodule.embedding_matrix(y_trg).to(device)
        samples = None
        labels = None
        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"concat_conditioning": z_emb_trg}


        if args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.torch.rand(samples.shape[0]).to(device)
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            if args.skewed_timesteps:
                t = skewed_timestep_sample(
                    x_real_ctrl.shape[0],
                    device=device,
                )
            else:
                t = torch.torch.rand(x_real_ctrl.shape[0]).to(device)
            if use_initial == 1:
                x_0 = x_real_ctrl
            elif use_initial == 2:
                p_r = random.random()
                if p_r > args.noise_prob:
                    x_0 = x_real_ctrl
                else:
                    x_0 = x_real_ctrl + torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device) * args.noise_level
            else:
                x_0 = centered_noise(x_real_ctrl.shape, getattr(args, "center_noise_sigma", 0.0), device=device)
            
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_real_trt)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.amp.autocast("cuda"):
                pred = model(x_t, t, extra=conditioning)
                if getattr(args, "foreground_loss", False):
                    loss = foreground_weighted_mse(
                        pred,
                        u_t,
                        x_0,
                        x_real_trt,
                        threshold=args.foreground_threshold,
                        foreground_weight=args.foreground_weight,
                        background_weight=args.background_weight,
                    )
                else:
                    loss = torch.pow(pred - u_t, 2).mean()

        # --- GAN adversarial loss (breaks near-identity) ---
        gan_weight = getattr(args, 'gan_weight', 0.0)
        if gan_weight > 0 and not args.discrete_flow_matching:
            global _DISCRIMINATOR, _DISC_OPTIMIZER
            # Reset discriminator each epoch to prevent it from getting too strong
            if data_iter_step == 0 or _DISCRIMINATOR is None:
                _DISCRIMINATOR = PatchDiscriminator().to(device)
                _DISC_OPTIMIZER = torch.optim.AdamW(
                    _DISCRIMINATOR.parameters(), lr=args.lr * 0.1, betas=(0.5, 0.9)
                )

            # Predict target image from velocity field
            with torch.amp.autocast("cuda"):
                x_pred_target = x_t + (1.0 - t.view(-1, 1, 1, 1)) * pred

            # Update D every 8 steps, skip first 500 steps, only after epoch 0
            update_d = (epoch > 0 or data_iter_step >= 500) and (data_iter_step % 8 == 0)
            if update_d:
                with torch.amp.autocast("cuda"):
                    real_out = _DISCRIMINATOR(x_real_trt)
                    fake_out = _DISCRIMINATOR(x_pred_target.detach())
                    d_loss = F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()
                    _DISC_OPTIMIZER.zero_grad()
                torch.nn.utils.clip_grad_norm_(_DISCRIMINATOR.parameters(), 0.1)
                loss_scaler(
                    d_loss, _DISC_OPTIMIZER, parameters=_DISCRIMINATOR.parameters(), update_grad=True
                )

            # Generator (flow model) adversarial loss
            with torch.amp.autocast("cuda"):
                fake_out_g = _DISCRIMINATOR(x_pred_target)
                g_loss = -fake_out_g.mean()
            loss = loss + gan_weight * g_loss

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        # Track per-condition loss (Feature B monitoring)
        if cond_loss_sums is not None and "concat_conditioning" in conditioning:
            per_sample = torch.pow(pred.detach() - u_t, 2).mean(dim=[1,2,3])  # [B]
            cond_loss_sums.index_add_(0, y_trg, per_sample)
            cond_loss_counts.index_add_(0, y_trg, torch.ones_like(per_sample))

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )
        # Live monitoring: save feature values to JSON files
        # (written at log frequency regardless of wandb)
        if data_iter_step % args.wandb_log_freq == 0 and args.output_dir:
            if _p_mean_shifts is not None:
                import json, os
                shifts_path = os.path.join(args.output_dir, "p_mean_shifts_live.json")
                with open(shifts_path, "w") as f:
                    json.dump({
                        "epoch": epoch, "step": data_iter_step,
                        "p_mean_shifts": _p_mean_shifts.detach().cpu().tolist(),
                    }, f)
            _vel_bias_proj = getattr(_model, 'velocity_bias_proj', None)
            if _vel_bias_proj is not None:
                import json, os
                vb_path = os.path.join(args.output_dir, "velocity_bias_live.json")
                with open(vb_path, "w") as f:
                    json.dump({
                        "epoch": epoch, "step": data_iter_step,
                        "velocity_bias_weight_norm": _vel_bias_proj[-1].weight.norm().item(),
                        "velocity_bias_bias": _vel_bias_proj[-1].bias.detach().cpu().tolist(),
                    }, f)

        # wandb per-step logging
        if wandb_run is not None and data_iter_step % args.wandb_log_freq == 0:
            log_dict = {
                "train/batch_loss": batch_loss.compute().item(),
                "train/lr": lr,
                "train/epoch": epoch,
            }
            if _p_mean_shifts is not None:
                for ci, shift in enumerate(_p_mean_shifts.detach().cpu().tolist()):
                    log_dict[f"train/p_mean_shift_c{ci}"] = shift
            _vel_bias_proj = getattr(_model, 'velocity_bias_proj', None)
            if _vel_bias_proj is not None:
                log_dict["train/velocity_bias_norm"] = _vel_bias_proj[-1].weight.norm().item()
            if cond_loss_sums is not None:
                cond_avg = cond_loss_sums / cond_loss_counts.clamp_min(1)
                for ci in range(_n_cond):
                    if cond_loss_counts[ci] > 0:
                        log_dict[f"train/cond_loss_c{ci}"] = cond_avg[ci].item()
            wandb_run.log(log_dict)

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
