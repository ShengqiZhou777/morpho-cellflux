#!/usr/bin/env python3
"""Run a CPU smoke check for the field-level microalgae lane."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morpho-cellflux-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from phenoflux.args import get_args_parser
from phenoflux.models.configs import instantiate_model
from phenoflux.training.dataloader import CellDataLoader


def main() -> int:
    parser = get_args_parser()
    parsed = parser.parse_args([])
    with open("configs/microalgae_field.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args_dict = vars(parsed)
    args_dict.update(config)
    args_dict.update(
        {
            "dataset": "phenoflux_small",
            "batch_size": 2,
            "num_workers": 0,
            "pin_mem": False,
            "num_tasks": 1,
            "global_rank": 0,
            "device": "cpu",
            "test_run": True,
            "use_ema": False,
        }
    )
    args = SimpleNamespace(**args_dict)

    for path_key in ["image_path", "data_index_path", "embedding_path"]:
        path = Path(getattr(args, path_key))
        if not path.exists():
            raise FileNotFoundError(f"{path_key} does not exist: {path}")

    datamodule = CellDataLoader(args)
    batch = next(iter(datamodule.train_dataloader()))
    x_ctrl, x_trt = batch["X"]
    y_trg = batch["mols"].long()
    z_emb = datamodule.embedding_matrix(y_trg)

    model = instantiate_model(
        architechture=args.dataset,
        use_ema=False,
        overrides=vars(args),
    )
    model.eval()

    t = torch.full((x_ctrl.shape[0],), 0.5)
    with torch.no_grad():
        pred = model(x_ctrl, t, extra={"concat_conditioning": z_emb})
        loss = torch.pow(pred - (x_trt - x_ctrl), 2).mean().item()

    assert pred.shape == x_ctrl.shape, f"unexpected prediction shape: {pred.shape}"
    assert torch.isfinite(torch.tensor(loss)), f"non-finite field smoke loss: {loss}"
    print(
        "field smoke ok: "
        f"train_batches={len(datamodule.train_dataloader())} "
        f"test_batches={len(datamodule.test_dataloader())} "
        f"batch_shape={tuple(x_ctrl.shape)} "
        f"embedding_dim={z_emb.shape[1]} "
        f"loss={loss:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
