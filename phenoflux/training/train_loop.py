# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from phenoflux.models.ema import EMA
from phenoflux.training.dataloader import CellDataLoader
from phenoflux.training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """EDM-style log-normal time sampling (P_mean=-1.2, P_std=1.2)."""
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma_random = (rnd_normal * P_std).exp().detach()
    sigma = sigma_random * torch.as_tensor(P_mean, device=device).exp()
    time = 1 / (1 + sigma)
    return torch.clip(time, min=0.0001, max=1.0)


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

        # Control-initialized flow (use_initial=1 is the only supported mode).
        x_0 = x_real_ctrl
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_real_trt)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        with torch.amp.autocast("cuda"):
            pred = model(x_t, t, extra=conditioning)
            loss = torch.pow(pred - u_t, 2).mean()

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
