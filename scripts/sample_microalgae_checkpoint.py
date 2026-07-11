#!/usr/bin/env python3
"""Generate visual control/generated/target grids from a microalgae checkpoint."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from flow_matching.solver.ode_solver import ODESolver

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenoflux.args import get_args_parser
from phenoflux.models.configs import instantiate_model
from phenoflux.training.data_utils import _load_microalgae_rgb
from phenoflux.training.eval_loop import CFGScaledModel


def _load_args(config_name: str) -> SimpleNamespace:
    parser = get_args_parser()
    args = parser.parse_args([])
    config_path = ROOT / "configs" / f"{config_name}.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    args_dict = vars(args)
    args_dict.update(config)
    args_dict.update(
        {
            "config": config_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "use_ema": False,
        }
    )
    return SimpleNamespace(**args_dict)


def _pick_labels(labels: list[str], n_per_condition: int) -> list[str]:
    picked: list[str] = []
    for prefix in ("dark_", "light_"):
        cond_labels = sorted([label for label in labels if label.startswith(prefix)])
        if not cond_labels:
            continue
        if len(cond_labels) <= n_per_condition:
            picked.extend(cond_labels)
            continue
        positions = np.linspace(0, len(cond_labels) - 1, n_per_condition)
        picked.extend([cond_labels[int(round(pos))] for pos in positions])
    return picked


def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x.detach().cpu() * 0.5 + 0.5, 0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _load_model(args: SimpleNamespace, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = instantiate_model(
        architechture=args.dataset,
        use_ema=False,
        overrides=vars(args),
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: {len(missing)} missing model keys")
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected checkpoint keys")
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {checkpoint}")
    return model


def _draw_grid(records: list[dict[str, object]], out_path: Path) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, max(2.0, 2.0 * n)))
    if n == 1:
        axes = np.asarray([axes])
    titles = ("control 0h", "generated", "target")
    for row, rec in enumerate(records):
        for col, key in enumerate(("control", "generated", "target")):
            ax = axes[row, col]
            ax.imshow(rec[key])
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=9)
        axes[row, 0].set_ylabel(str(rec["label"]), fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="microalgae_timepoint_quick")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels-per-condition", type=int, default=3)
    parser.add_argument("--examples-per-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cfg-scale", type=float, default=0.2)
    parser.add_argument("--ode-method", default="midpoint")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument(
        "--init-mode",
        choices=("control", "noise"),
        default="control",
        help="Initial state for ODE sampling. Use noise for models trained with USE_INITIAL=0.",
    )
    parser.add_argument("--init-seed", type=int, default=1234)
    args_cli = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/morpho-cellflux-matplotlib")
    rng = np.random.default_rng(args_cli.seed)
    args = _load_args(args_cli.config)
    device = torch.device(args.device)
    print(f"device={device}")

    index = pd.read_csv(ROOT / args.data_index_path, index_col=0)
    embedding = pd.read_csv(ROOT / args.embedding_path, index_col=0)
    labels = _pick_labels(list(embedding.index), args_cli.labels_per_condition)

    selected: list[tuple[pd.Series, pd.Series]] = []
    for label in labels:
        split = index[(index["SPLIT"] == "test") & (index["CPD_NAME"] == label)]
        treated = split[split["ANNOT"] == "treated"]
        controls = split[split["ANNOT"] == "negative_control"]
        if treated.empty or controls.empty:
            continue
        take = min(args_cli.examples_per_label, len(treated))
        chosen = treated.sample(n=take, random_state=args_cli.seed)
        for _, target in chosen.iterrows():
            ctrl = controls.iloc[int(rng.integers(0, len(controls)))]
            selected.append((ctrl, target))

    if not selected:
        raise ValueError("No examples selected for sampling")

    model = _load_model(args, args_cli.checkpoint, device=device)
    solver = ODESolver(velocity_model=CFGScaledModel(model))

    ctrl_tensors, target_tensors, z_rows, names = [], [], [], []
    for ctrl, target in selected:
        ctrl_tensors.append(_load_microalgae_rgb(ROOT / args.image_path, ctrl["SAMPLE_KEY"]))
        target_tensors.append(_load_microalgae_rgb(ROOT / args.image_path, target["SAMPLE_KEY"]))
        z_rows.append(embedding.loc[target["CPD_NAME"]].to_numpy(dtype=np.float32))
        names.append((target["CPD_NAME"], Path(target["SAMPLE_KEY"]).stem))

    x_ctrl = torch.stack(ctrl_tensors).to(device)
    x_target = torch.stack(target_tensors).to(device)
    z = torch.from_numpy(np.stack(z_rows)).to(device)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    if args_cli.init_mode == "control":
        x_init = x_ctrl
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(args_cli.init_seed)
        x_init = torch.randn(
            x_ctrl.shape,
            dtype=x_ctrl.dtype,
            device=device,
            generator=generator,
        )

    with torch.no_grad():
        generated = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=args_cli.ode_method,
            step_size=args_cli.step_size if args_cli.ode_method != "dopri5" else None,
            atol=1e-5,
            rtol=1e-5,
            cfg_scale=args_cli.cfg_scale,
            extra={"concat_conditioning": z},
        )

    records = []
    for i, (label, stem) in enumerate(names):
        rec = {
            "label": label,
            "control": _to_numpy_image(x_ctrl[i]),
            "generated": _to_numpy_image(generated[i]),
            "target": _to_numpy_image(x_target[i]),
        }
        records.append(rec)
        label_dir = args_cli.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for key in ("control", "generated", "target"):
            plt.imsave(label_dir / f"{stem}_{key}.png", rec[key])

    grid_path = args_cli.output_dir / "grid_control_generated_target.png"
    _draw_grid(records, grid_path)
    print(f"Saved {len(records)} examples to {args_cli.output_dir}")
    print(f"Grid: {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
