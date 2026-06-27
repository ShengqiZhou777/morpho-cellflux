"""Train IMPA on exported Morpho-CellFlux benchmark data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from adapter_common import load_config, repo_path
from impa_export_fid import default_impa_args


IMPA_ROOT = repo_path("baselines/external/impa")
sys.path.insert(0, str(IMPA_ROOT))

from IMPA.dataset.data_loader import CellDataLoader  # noqa: E402
from IMPA.solver import IMPAmodule  # noqa: E402


def train_impa(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    benchmark: str,
    epochs: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    devices: int,
) -> None:
    config = load_config(config_path)
    run_dir = output_dir / "external_checkpoints" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    args = default_impa_args(config, data_dir, output_dir / "external_checkpoints")
    args.task_name = f"impa_{benchmark}"
    args.total_epochs = epochs
    args.batch_size = batch_size
    args.val_batch_size = val_batch_size
    args.num_workers = num_workers
    args.project = f"impa_{benchmark}"
    args.offline = True
    args.experiment_directory = str(output_dir / "external_checkpoints")

    # vendor compat: IMPA's solver writes per-epoch debug samples (utils.debug_image) and
    # checkpoints to these subdirs of dest_dir but never creates them -> FileNotFoundError
    # in on_train_epoch_end. Pre-create them.
    for _sub in (args.sample_dir, args.checkpoint_dir):
        (run_dir / _sub).mkdir(parents=True, exist_ok=True)

    datamodule = CellDataLoader(args)
    args.latent_dim = datamodule.latent_dim  # must match embedding file dimension
    solver = IMPAmodule(args, str(run_dir), datamodule)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "hydra_checkpoints",
        filename=args.filename,
        monitor=args.monitor,
        mode=args.mode,
        save_last=args.save_last,
    )
    logger = WandbLogger(save_dir=str(run_dir), offline=True, project=args.project, log_model=False)
    trainer_kwargs = {
        "callbacks": [checkpoint, EarlyStopping(monitor="fid_transformations", patience=3, min_delta=0.001, mode="min")],
        "default_root_dir": str(run_dir),
        "logger": logger,
        "max_epochs": epochs,
        "accelerator": "gpu" if devices > 0 else "cpu",
        "log_every_n_steps": args.log_every_n_steps,
    }
    # Fast smoke iteration: IMPA_LIMIT_TRAIN_BATCHES=N caps train batches/epoch so the
    # epoch-end hooks (eval, debug image, checkpoint) + export fire in minutes, not ~36 min.
    _lim = os.environ.get("IMPA_LIMIT_TRAIN_BATCHES")
    if _lim:
        trainer_kwargs["limit_train_batches"] = int(_lim)
    if devices == 1:
        trainer_kwargs["devices"] = 1
    elif devices > 1:
        raise ValueError(
            "IMPA modules are already wrapped in nn.DataParallel. Use --devices 1 "
            "with CUDA_VISIBLE_DEVICES=0,1 so IMPA's internal DataParallel uses both GPUs."
        )

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(
        model=solver,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_impa(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
