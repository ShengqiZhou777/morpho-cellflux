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
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from phenoflux.models.ema import EMA
from phenoflux.data.data_utils import centered_noise
from phenoflux.data.dataloader import CellDataLoader
from phenoflux.training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50

# --- PatchGAN discriminator (used only when --gan_weight > 0) ---
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


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """EDM-style log-normal time sampling (P_mean=-1.2, P_std=1.2)."""
    P_mean = -0.5
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma_random = (rnd_normal * P_std).exp().detach()
    sigma = sigma_random * torch.as_tensor(P_mean, device=device).exp()
    time = 1 / (1 + sigma)
    return torch.clip(time, min=0.0001, max=1.0)


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Maximum Mean Discrepancy with Gaussian RBF kernel (scDFM-style, Yu et al. 2026).

    Compares two minibatches at the distribution level — unlike per-sample MSE this
    penalises variance collapse. The kernel bandwidth defaults to the median
    pairwise distance (median heuristic, Gretton et al. 2012)."""
    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2
    if sigma is None:
        # median heuristic on the joint set
        with torch.no_grad():
            d = torch.cat([x, y], dim=0)
            pd = torch.cdist(d, d, p=2) ** 2
            sigma2 = pd.median()
    else:
        sigma2 = sigma ** 2
    gamma = 1.0 / (2.0 * sigma2 + 1e-8)
    k_xx = torch.exp(-gamma * xx).mean()
    k_yy = torch.exp(-gamma * yy).mean()
    k_xy = torch.exp(-gamma * xy).mean()
    return k_xx + k_yy - 2.0 * k_xy


def my_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
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
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    accum_iter = args.accum_iter
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
        z_emb_trg = datamodule.embedding_matrix(y_trg).to(device)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"concat_conditioning": z_emb_trg}

        if args.skewed_timesteps:
            t = skewed_timestep_sample(x_real_ctrl.shape[0], device=device)
        else:
            t = torch.rand(x_real_ctrl.shape[0], device=device)

        # Flow start point: 1 = control image, 2 = control + noise, 0 = (centered) noise.
        if use_initial == 1:
            x_0 = x_real_ctrl
        elif use_initial == 2:
            if random.random() > args.noise_prob:
                x_0 = x_real_ctrl
            else:
                x_0 = x_real_ctrl + torch.randn_like(x_real_ctrl) * args.noise_level
        else:
            x_0 = centered_noise(x_real_ctrl.shape, getattr(args, "center_noise_sigma", 0.0), device=device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_real_trt)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        with torch.amp.autocast("cuda"):
            pred = model(x_t, t, extra=conditioning)
            loss = torch.pow(pred - u_t, 2).mean()

        # --- GAN adversarial loss (breaks near-identity), enabled by --gan_weight > 0 ---
        gan_weight = getattr(args, 'gan_weight', 0.0)
        if gan_weight > 0:
            global _DISCRIMINATOR, _DISC_OPTIMIZER
            if data_iter_step == 0 or _DISCRIMINATOR is None:
                _DISCRIMINATOR = PatchDiscriminator().to(device)
                _DISC_OPTIMIZER = torch.optim.AdamW(
                    _DISCRIMINATOR.parameters(), lr=args.lr * 0.1, betas=(0.5, 0.9)
                )
            with torch.amp.autocast("cuda"):
                x_pred_target = x_t + (1.0 - t.view(-1, 1, 1, 1)) * pred
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
            with torch.amp.autocast("cuda"):
                fake_out_g = _DISCRIMINATOR(x_pred_target)
                g_loss = -fake_out_g.mean()
            loss = loss + gan_weight * g_loss

        # --- MMD distribution-matching loss (scDFM, Yu et al. 2026) ---
        mmd_weight = getattr(args, 'mmd_weight', 0.0)
        if mmd_weight > 0 and use_initial == 0:
            # Compare generated batch vs real target at the distribution level.
            # x_pred_target was computed above; downsample to 16×16 for efficiency.
            x_pred_target = x_t + (1.0 - t.view(-1, 1, 1, 1)) * pred
            gen_ds = torch.nn.functional.interpolate(
                x_pred_target, size=(16, 16), mode='bilinear', align_corners=False
            ).flatten(1)
            real_ds = torch.nn.functional.interpolate(
                x_real_trt, size=(16, 16), mode='bilinear', align_corners=False
            ).flatten(1)
            loss = loss + mmd_weight * _mmd_rbf(gen_ds, real_ds)

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)
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
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss = {batch_loss.compute()}, lr = {lr}"
            )
        if wandb_run is not None and data_iter_step % args.wandb_log_freq == 0:
            wandb_run.log({
                "train/batch_loss": batch_loss.compute().item(),
                "train/lr": lr,
                "train/epoch": epoch,
            })

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
