#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphoflux.data.torch_dataset import CellFluxPairDataset
from morphoflux.models import ConditionalUNet2D
from morphoflux.training import (
    build_flow_batch,
    combine_velocity_heads,
    euler_sample,
    flow_matching_loss,
    image_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CellFlux-style conditional flow matching model."
    )
    parser.add_argument(
        "--config",
        default="configs/train_cellflux.yaml",
        help="Training YAML path relative to the project root.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help="Use only the first N training rows for a smoke run.",
    )
    parser.add_argument(
        "--limit-val-rows",
        type=int,
        default=None,
        help="Use only the first N validation rows for a smoke run.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(device_name: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if distributed:
        if device_name.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("DDP requested CUDA but CUDA is not available")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device(device_name)
            backend = "gloo"
        dist.init_process_group(backend=backend)
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", 0)

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def reduce_loss_sum_count(
    loss_sum: float,
    count: int,
    device: torch.device,
    distributed: bool,
) -> float:
    values = torch.tensor([float(loss_sum), float(count)], device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_count = float(values[1].item())
    return float(values[0].item() / total_count) if total_count > 0 else float("nan")


def reduce_mean_scalar(
    value: torch.Tensor,
    distributed: bool,
) -> float:
    reduced = value.detach().float()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
    return float(reduced.item())


def build_loader(
    pairs_path: Path,
    cfg: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    row_limit: int | None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = CellFluxPairDataset(
        pairs_path=pairs_path,
        project_root=PROJECT_ROOT,
        image_key=cfg["data"].get("image_key", "x"),
        channel_indices=cfg["data"].get("channel_indices"),
        return_onehot=False,
    )
    if row_limit is not None:
        dataset = Subset(dataset, range(min(int(row_limit), len(dataset))))

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        if distributed
        else None
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": int(cfg["training"].get("num_workers", 0)),
        "pin_memory": bool(cfg["training"].get("pin_memory", False)),
        "persistent_workers": bool(cfg["training"].get("persistent_workers", False))
        and int(cfg["training"].get("num_workers", 0)) > 0,
        "drop_last": False,
    }
    if int(cfg["training"].get("num_workers", 0)) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 2))
    return DataLoader(
        dataset,
        **loader_kwargs,
    ), sampler


def infer_num_conditions(vocab_path: Path) -> int:
    vocab = json.loads(vocab_path.read_text())
    if not vocab:
        raise ValueError(f"condition vocab is empty: {vocab_path}")
    return max(int(v) for v in vocab.values()) + 1


def move_batch(batch: tuple[Any, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    source, target, condition, _meta = batch
    return (
        source.to(device=device, dtype=torch.float32, non_blocking=True),
        target.to(device=device, dtype=torch.float32, non_blocking=True),
        condition.to(device=device, dtype=torch.long, non_blocking=True),
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    flow_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    distributed: bool,
) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            source, target, condition = move_batch(batch, device)
            x_t, t, condition, velocity = build_flow_batch(
                source,
                target,
                condition,
                target_scaffold=bool(flow_cfg.get("target_scaffold", False)),
                mask_threshold=float(flow_cfg.get("mask_threshold", 1e-4)),
                start_mode=str(flow_cfg.get("start_mode", "source")),
                start_noise_scale=float(flow_cfg.get("start_noise_scale", 0.2)),
                start_noise_prob=float(flow_cfg.get("start_noise_prob", 0.5)),
            )
            pred_raw = model(x_t, t, condition)
            pred, pred_residual = combine_velocity_heads(
                pred_raw,
                image_channels=target.shape[1],
                residual_scale=float(loss_cfg.get("residual_scale", 1.0)),
            )
            target_mask = image_mask(target, float(flow_cfg.get("mask_threshold", 1e-4)))
            loss = flow_matching_loss(
                pred,
                velocity,
                pred_residual_velocity=pred_residual,
                target_image=target,
                mask=target_mask,
                channel_weights=loss_cfg.get("channel_weights"),
                foreground_weight=float(loss_cfg.get("foreground_weight", 0.0)),
                image_weight=float(loss_cfg.get("image_weight", 0.0)),
                highpass_weight=float(loss_cfg.get("highpass_weight", 0.0)),
                highpass_channels=loss_cfg.get("highpass_channels"),
                highpass_kernel=int(loss_cfg.get("highpass_kernel", 9)),
                highpass_sigma=float(loss_cfg.get("highpass_sigma", 1.5)),
                puncta_weight=float(loss_cfg.get("puncta_weight", 0.0)),
                puncta_channels=loss_cfg.get("puncta_channels"),
                puncta_fraction=float(loss_cfg.get("puncta_fraction", 0.03)),
                puncta_kernel=int(loss_cfg.get("puncta_kernel", 9)),
                puncta_sigma=float(loss_cfg.get("puncta_sigma", 1.5)),
                puncta_temperature=float(loss_cfg.get("puncta_temperature", 0.05)),
                residual_weight=float(loss_cfg.get("residual_weight", 0.0)),
                residual_channels=loss_cfg.get("residual_channels"),
                residual_kernel=int(loss_cfg.get("residual_kernel", 9)),
                residual_sigma=float(loss_cfg.get("residual_sigma", 1.5)),
            )
            batch_size = int(source.shape[0])
            loss_sum += float(loss.item()) * batch_size
            count += batch_size
    if was_training:
        model.train()
    return reduce_loss_sum_count(loss_sum, count, device, distributed)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: dict[str, Any],
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "config": cfg,
            "val_loss": float(val_loss),
        },
        path,
    )


def save_preview(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    steps: int,
    guidance_scale: float,
    start_mode: str,
    start_noise_scale: float,
    flow_cfg: dict[str, Any],
    residual_scale: float,
    channel_names: list[str],
) -> None:
    model = unwrap_model(model)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        source, target, condition = move_batch(batch, device)
        scaffold_mask = (
            image_mask(target[: min(4, target.shape[0])], float(flow_cfg.get("mask_threshold", 1e-4)))
            if bool(flow_cfg.get("target_scaffold", False))
            else None
        )
        generated = euler_sample(
            model,
            source[: min(4, source.shape[0])],
            condition[: min(4, condition.shape[0])],
            steps=steps,
            guidance_scale=guidance_scale,
            scaffold_mask=scaffold_mask,
            start_mode=start_mode,
            start_noise_scale=start_noise_scale,
            residual_scale=residual_scale,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source=source[: generated.shape[0]].cpu().numpy(),
        target=target[: generated.shape[0]].cpu().numpy(),
        generated=generated.cpu().numpy(),
        condition=condition[: generated.shape[0]].cpu().numpy(),
        channel_names=np.asarray(channel_names, dtype="U"),
    )
    if was_training:
        model.train()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    seed = int(cfg["training"].get("seed", 17))
    device_name = args.device or cfg["training"].get("device", "auto")
    distributed, rank, local_rank, world_size, device = setup_distributed(str(device_name))
    set_seed(seed + rank)

    try:
        global_batch_size = args.batch_size or int(cfg["training"]["batch_size"])
        if distributed and global_batch_size % world_size != 0:
            raise ValueError(
                f"global batch_size={global_batch_size} must be divisible by "
                f"DDP world_size={world_size}"
            )
        per_device_batch_size = (
            global_batch_size // world_size if distributed else global_batch_size
        )
        max_steps = args.max_steps or int(cfg["training"]["max_steps"])
        output_dir = resolve(args.output_dir or cfg["training"]["output_dir"])
        checkpoint_dir = output_dir / "checkpoints"
        metrics_path = output_dir / "metrics.jsonl"

        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(distributed)

        train_loader, train_sampler = build_loader(
            resolve(cfg["data"]["train_pairs"]),
            cfg,
            batch_size=per_device_batch_size,
            shuffle=True,
            row_limit=args.limit_train_rows,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        val_loader, val_sampler = build_loader(
            resolve(cfg["data"]["val_pairs"]),
            cfg,
            batch_size=per_device_batch_size,
            shuffle=False,
            row_limit=args.limit_val_rows,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        preview_loader = None
        if is_main_process(rank):
            preview_loader, _ = build_loader(
                resolve(cfg["data"]["val_pairs"]),
                cfg,
                batch_size=min(global_batch_size, 4),
                shuffle=False,
                row_limit=args.limit_val_rows,
                distributed=False,
            )

        image_channels = int(cfg["data"]["image_shape"][0])
        channel_names = [
            str(name)
            for name in cfg["data"].get(
                "channel_names",
                [f"ch{idx:02d}" for idx in range(image_channels)],
            )
        ]
        if len(channel_names) != image_channels:
            raise ValueError(
                f"data.channel_names has {len(channel_names)} names, "
                f"expected {image_channels}"
            )
        flow_cfg = cfg.get("flow_matching", {})
        loss_cfg = cfg.get("loss", {})
        target_scaffold = bool(flow_cfg.get("target_scaffold", False))
        num_conditions = int(
            cfg["data"].get("num_conditions")
            or infer_num_conditions(resolve(cfg["data"]["condition_vocab"]))
        )
        model_cfg = cfg["model"]
        high_frequency_residual = bool(model_cfg.get("high_frequency_residual", False))
        model = ConditionalUNet2D(
            in_channels=image_channels + (1 if target_scaffold else 0),
            out_channels=image_channels * (2 if high_frequency_residual else 1),
            num_conditions=num_conditions,
            hidden_channels=int(model_cfg["hidden_channels"]),
            channel_mults=tuple(int(v) for v in model_cfg["channel_mults"]),
            embedding_dim=int(model_cfg["embedding_dim"]),
        ).to(device)

        if distributed:
            if device.type == "cuda":
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
            else:
                model = DistributedDataParallel(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        )

        log_every = int(cfg["training"].get("log_every", 20))
        save_every = int(cfg["training"].get("save_every", 500))
        val_batches = int(cfg["training"].get("val_batches", 8))
        val_every_cfg = cfg["training"].get("val_every", "epoch")
        val_every_mode = str(val_every_cfg).lower()
        val_every_steps = None
        if val_every_mode not in {"epoch", "never"}:
            val_every_steps = int(val_every_cfg)
            if val_every_steps < 1:
                raise ValueError("val_every must be 'epoch', 'never', or a positive integer")
        preview_every = int(cfg["training"].get("preview_every", 0))
        grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))

        step = 0
        best_val = float("inf")
        started = time.time()
        model.train()
        if is_main_process(rank):
            print(
                json.dumps(
                    {
                        "event": "start",
                        "device": str(device),
                        "distributed": distributed,
                        "rank": rank,
                        "world_size": world_size,
                        "global_batch_size": global_batch_size,
                        "per_device_batch_size": per_device_batch_size,
                        "max_steps": max_steps,
                        "num_conditions": num_conditions,
                        "channel_names": channel_names,
                        "high_frequency_residual": high_frequency_residual,
                        "loss": loss_cfg,
                        "val_every": val_every_cfg,
                        "train_batches_per_rank": len(train_loader),
                        "val_batches_per_rank": len(val_loader),
                    },
                    sort_keys=True,
                )
            )

        epoch = 0
        while step < max_steps:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_loader):
                source, target, condition = move_batch(batch, device)
                x_t, t, train_condition, velocity = build_flow_batch(
                    source,
                    target,
                    condition,
                    condition_drop_prob=float(flow_cfg.get("condition_drop_prob", 0.0)),
                    source_noise_prob=float(flow_cfg.get("source_noise_prob", 0.0)),
                    source_noise_scale=float(flow_cfg.get("source_noise_scale", 0.0)),
                    null_condition_id=int(flow_cfg.get("null_condition_id", 0)),
                    target_scaffold=target_scaffold,
                    mask_threshold=float(flow_cfg.get("mask_threshold", 1e-4)),
                    start_mode=str(flow_cfg.get("start_mode", "source")),
                    start_noise_scale=float(flow_cfg.get("start_noise_scale", 0.2)),
                    start_noise_prob=float(flow_cfg.get("start_noise_prob", 0.5)),
                )

                pred_raw = model(x_t, t, train_condition)
                pred, pred_residual = combine_velocity_heads(
                    pred_raw,
                    image_channels=target.shape[1],
                    residual_scale=float(loss_cfg.get("residual_scale", 1.0)),
                )
                target_mask = image_mask(target, float(flow_cfg.get("mask_threshold", 1e-4)))
                loss = flow_matching_loss(
                    pred,
                    velocity,
                    pred_residual_velocity=pred_residual,
                    target_image=target,
                    mask=target_mask,
                    channel_weights=loss_cfg.get("channel_weights"),
                    foreground_weight=float(loss_cfg.get("foreground_weight", 0.0)),
                    image_weight=float(loss_cfg.get("image_weight", 0.0)),
                    highpass_weight=float(loss_cfg.get("highpass_weight", 0.0)),
                    highpass_channels=loss_cfg.get("highpass_channels"),
                    highpass_kernel=int(loss_cfg.get("highpass_kernel", 9)),
                    highpass_sigma=float(loss_cfg.get("highpass_sigma", 1.5)),
                    puncta_weight=float(loss_cfg.get("puncta_weight", 0.0)),
                    puncta_channels=loss_cfg.get("puncta_channels"),
                    puncta_fraction=float(loss_cfg.get("puncta_fraction", 0.03)),
                    puncta_kernel=int(loss_cfg.get("puncta_kernel", 9)),
                    puncta_sigma=float(loss_cfg.get("puncta_sigma", 1.5)),
                    puncta_temperature=float(loss_cfg.get("puncta_temperature", 0.05)),
                    residual_weight=float(loss_cfg.get("residual_weight", 0.0)),
                    residual_channels=loss_cfg.get("residual_channels"),
                    residual_kernel=int(loss_cfg.get("residual_kernel", 9)),
                    residual_sigma=float(loss_cfg.get("residual_sigma", 1.5)),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                step += 1

                should_log = step % log_every == 0 or step == 1 or step == max_steps
                should_save = step % save_every == 0 or step == max_steps
                is_epoch_end = batch_idx + 1 == len(train_loader)
                should_validate = (
                    should_save
                    or (val_every_mode == "epoch" and is_epoch_end)
                    or (
                        val_every_steps is not None
                        and step % val_every_steps == 0
                    )
                )
                should_preview = preview_every > 0 and (
                    step % preview_every == 0 or step == max_steps
                )

                val_loss = None
                if should_validate:
                    val_loss = evaluate(
                        model,
                        val_loader,
                        device,
                        val_batches,
                        flow_cfg,
                        loss_cfg,
                        distributed,
                    )

                if should_log or should_validate:
                    train_loss = reduce_mean_scalar(loss, distributed)
                    event = {
                        "event": "step",
                        "step": step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                    if val_loss is not None:
                        event["val_loss"] = float(val_loss)
                    if is_main_process(rank):
                        print(json.dumps(event, sort_keys=True))
                        with metrics_path.open("a") as f:
                            f.write(json.dumps(event, sort_keys=True) + "\n")
                if (
                    val_loss is not None
                    and is_main_process(rank)
                    and float(val_loss) < best_val
                ):
                    best_val = float(val_loss)
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        step,
                        epoch,
                        cfg,
                        float(val_loss),
                    )

                if should_save and is_main_process(rank):
                    assert val_loss is not None
                    save_checkpoint(
                        checkpoint_dir / f"step_{step:07d}.pt",
                        unwrap_model(model),
                        optimizer,
                        step,
                        epoch,
                        cfg,
                        float(val_loss),
                    )

                if should_preview and is_main_process(rank):
                    assert preview_loader is not None
                    save_preview(
                        model,
                        preview_loader,
                        device,
                        output_dir / "previews" / f"step_{step:07d}.npz",
                        steps=int(cfg["sampling"].get("steps", 16)),
                        guidance_scale=float(cfg["sampling"].get("guidance_scale", 1.0)),
                        start_mode=str(cfg["sampling"].get("start_mode", "source")),
                        start_noise_scale=float(
                            cfg["sampling"].get("start_noise_scale", 0.2)
                        ),
                        flow_cfg=flow_cfg,
                        residual_scale=float(loss_cfg.get("residual_scale", 1.0)),
                        channel_names=channel_names,
                    )

                if should_save or should_preview:
                    barrier(distributed)

                if step >= max_steps:
                    break

        if is_main_process(rank):
            print(
                json.dumps(
                    {
                        "event": "done",
                        "step": step,
                        "best_val_loss": best_val,
                        "output_dir": str(output_dir),
                    },
                    sort_keys=True,
                )
            )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
