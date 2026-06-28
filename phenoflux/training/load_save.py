# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
from phenoflux.training.distributed import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, epoch, model, model_without_ddp, optimizer, lr_schedule, loss_scaler,
    datamodule=None,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % epoch_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            # Save inner model weights (unwrap EMA if present).
            # EMA.state_dict() includes shadow_params + model.* prefix;
            # saving inner model directly produces clean portable checkpoints.
            inner = model_without_ddp
            if hasattr(model_without_ddp, 'model'):
                inner = model_without_ddp.model
            to_save = {
                "model": inner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def load_model(
    args, model_without_ddp, optimizer, loss_scaler, lr_schedule,
    datamodule=None,
):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Load model weights with EMA awareness.
        # Old checkpoints saved EMA.state_dict() which wraps model params
        # with a 'model.' prefix and includes shadow_params.* entries.
        # New checkpoints save inner model state directly.
        # We handle both formats and always load into the inner (unwrapped) model.
        raw_state = checkpoint["model"]
        state_dict = {}
        has_model_prefix = any(k.startswith("model.") for k in raw_state)
        has_shadow = any(k.startswith("shadow_params.") for k in raw_state)

        for k, v in raw_state.items():
            # Skip EMA bookkeeping keys (shadow params, update counter)
            if k.startswith("shadow_params.") or k == "num_updates":
                continue
            # Strip 'model.' prefix from old EMA-wrapped checkpoints
            if k.startswith("model."):
                k = k[len("model."):]

            state_dict[k] = v

        # Always load into the inner (unwrapped) model.
        # model_without_ddp may be EMA(UNetModel) or raw UNetModel.
        target = model_without_ddp
        if hasattr(model_without_ddp, 'model'):
            target = model_without_ddp.model

        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Note: {len(missing)} keys in model not found in checkpoint"
                  f" (new layers, OK for backward compat)")
        if unexpected:
            print(f"  Note: {len(unexpected)} keys in checkpoint not in model"
                  f" (removed layers, OK for backward compat)")
        if has_model_prefix:
            print(f"  Loaded EMA-wrapped checkpoint (stripped model. prefix)")
        print(f"  Model weights: {len(state_dict)} parameters loaded")
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (getattr(args, "eval_only", False) or getattr(args, "eval", False))
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

            print("With optim & sched!")
