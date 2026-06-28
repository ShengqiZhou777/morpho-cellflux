# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from tqdm import tqdm
from phenoflux.models.configs import instantiate_model
from phenoflux.args import get_args_parser

from phenoflux.training.distributed import init_distributed_mode, is_main_process, get_rank, get_world_size
from phenoflux.training.data_transform import get_train_transform
from phenoflux.training.eval_loop import eval_model
from phenoflux.training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from phenoflux.training.load_save import load_model, save_model
from phenoflux.training.train_loop import my_train_one_epoch
from phenoflux.training.dataloader import CellDataLoader
logger = logging.getLogger(__name__)


def main(args):
    # Prevent CUDA memory allocator fragmentation on single-GPU setups.
    # expandable_segments avoids fixed-size block pooling that causes
    # cudaMalloc deadlocks when the allocator runs out of contiguous blocks
    # despite having free memory. Must be set BEFORE any CUDA operation.
    # Multi-GPU (2x32GB) does not hit this; single-GPU (24GB) benefits.
    if 'PYTORCH_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    # Force stdout/stderr to be line-buffered so training progress is visible
    # in real-time even when piped to a file or background task manager.
    # Without this, Python uses block buffering for non-TTY outputs, causing
    # multi-hour gaps between visible log entries.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # override any stale handler config from imports
    )
    init_distributed_mode(args)

    # wandb initialization (rank 0 only, only when --wandb_project is set)
    # Tries online mode first; falls back to offline mode if no API key is configured.
    # Offline logs are saved to {output_dir}/wandb/ and can be uploaded later via `wandb sync`.
    wandb_run = None
    if args.wandb_project and is_main_process():
        import wandb
        run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))
        for mode in ("online", "offline"):
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    tags=args.wandb_tags,
                    config=vars(args),
                    dir=args.output_dir,
                    resume="allow",
                    mode=mode,
                )
                logger.info(f"wandb ({mode}): {wandb_run.get_url() if mode == 'online' else wandb_run.dir}")
                break
            except Exception as e:
                if mode == "online":
                    logger.warning(f"wandb online failed ({e}), trying offline mode...")
                else:
                    logger.warning(f"wandb offline also failed ({e}), continuing without wandb")
                    wandb_run = None

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    valid_datasets = ['phenoflux']
    if args.dataset in valid_datasets:
        args.num_tasks = get_world_size()
        num_tasks = args.num_tasks
        args.global_rank = get_rank()
        global_rank = args.global_rank
        logger.info("Intializing DataLoader")
        datamodule = CellDataLoader(args)
        data_loader_train = datamodule.train_dataloader()
        data_loader_test = datamodule.test_dataloader()
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")


    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
        overrides=vars(args),  # pass YAML molecular prior flags
    )
    model.to(device)

    model_without_ddp = model

    eff_batch_size = (
        args.batch_size * args.accum_iter * get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
        datamodule=datamodule,
    )
    if args.use_initial in [1, 2]:
        logger.info("Generating From Control Image!!!!!")
    else:
        logger.info("Generating From Random Noise Image!!!!!")

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    best_loss = float("inf")
    best_fid = float("inf")
    early_stop_counter = 0
    import shutil
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = my_train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
                wandb_run=wandb_run,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        # Save checkpoint every epoch so training can be paused/resumed at any point
        if args.output_dir and not args.eval_only:
            save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=loss_scaler,
                epoch=epoch,
                datamodule=datamodule,
            )
            # Track best-loss checkpoint + early stopping (all ranks check for DDP safety)
            current_loss = train_stats.get("loss", float("inf"))
            improved = current_loss < best_loss - args.early_stop_min_delta
            if improved:
                best_loss = current_loss
                early_stop_counter = 0
                if is_main_process():
                    best_path = Path(args.output_dir) / "checkpoint-best_loss.pth"
                    shutil.copy2(Path(args.output_dir) / "checkpoint.pth", best_path)
                    logger.info(f"New best loss: {best_loss:.6f} @ epoch {epoch}")
            elif args.early_stop_patience > 0:
                early_stop_counter += 1
                if is_main_process():
                    logger.info(f"Early stop: {early_stop_counter}/{args.early_stop_patience} epochs without loss improvement")
                if early_stop_counter >= args.early_stop_patience:
                    if is_main_process():
                        logger.info(f"Early stopping triggered at epoch {epoch} (loss plateaued for {early_stop_counter} epochs).")
                    break

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks

            eval_stats = eval_model(
                model,
                data_loader_test,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
                interpolate=args.interpolate,
            )
            try:
                log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
                logger.info(log_stats)
                # Track best-FID checkpoint
                if is_main_process() and "fid" in eval_stats and eval_stats["fid"] is not None:
                    current_fid = eval_stats["fid"]
                    if current_fid < best_fid:
                        best_fid = current_fid
                        best_path = Path(args.output_dir) / "checkpoint-best_fid.pth"
                        shutil.copy2(Path(args.output_dir) / "checkpoint.pth", best_path)
                        logger.info(f"New best FID: {best_fid:.4f} @ epoch {epoch}")
            except:
                pass
        if args.output_dir and is_main_process():
            # wandb epoch-level logging
            if wandb_run is not None:
                wandb_run.log(
                    {f"epoch/{k}": v for k, v in log_stats.items()},
                    step=epoch,
                )
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

    if wandb_run is not None:
        wandb_run.finish()


def load_yaml_config(yaml_path):
    # A path ending in .yaml or an absolute path is used as given.
    # A bare config name is resolved inside the config directory, which defaults
    # to the repository's configs/ and can be overridden with the
    # MORPHOFLUX_CONFIG_DIR environment variable.
    if yaml_path.endswith(".yaml") or os.path.isabs(yaml_path):
        path = yaml_path
    else:
        repo_root = os.path.dirname(os.path.dirname(__file__))  # phenoflux/ → repo root
        default_config_dir = os.path.join(repo_root, "configs")
        config_dir = os.environ.get("MORPHOFLUX_CONFIG_DIR", default_config_dir)
        path = os.path.join(config_dir, yaml_path + ".yaml")
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    yaml_config = load_yaml_config(args.config)
    args_dict = vars(args)
    args_dict.update(yaml_config)
    # CLI --data_index override: allows switching data size without duplicating configs
    if args.data_index:
        args_dict["data_index_path"] = args.data_index
    args = SimpleNamespace(**args_dict)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
