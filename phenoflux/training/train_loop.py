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
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from phenoflux.models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from phenoflux.training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
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
    _use_msa = getattr(_model, 'use_msa', False)
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

        # Marker profile: only add when conditioning is active (not dropped for CFG).
        # Otherwise the unconditional path sees the molecular profile and can cheat,
        # weakening classifier-free guidance.
        if "marker_profile" in batch and "concat_conditioning" in conditioning:
            conditioning["marker_profile"] = batch["marker_profile"].to(device)
        
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
                t = skewed_timestep_sample(x_real_ctrl.shape[0], device=device)
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
                x_0 = torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device)
            
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
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )
        # wandb per-step logging
        if wandb_run is not None and data_iter_step % args.wandb_log_freq == 0:
            wandb_run.log({
                "train/batch_loss": batch_loss.compute().item(),
                "train/lr": lr,
                "train/epoch": epoch,
            })

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
