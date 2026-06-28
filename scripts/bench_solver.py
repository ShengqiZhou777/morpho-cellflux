#!/usr/bin/env python3
"""Quick A/B comparison of ODE solvers for flow matching image generation.

Compares midpoint (current default) vs dopri5 vs midpoint with larger step_size,
measuring NFE, wall-clock time, and saving sample images for visual comparison.

Usage:
  python scripts/bench_solver.py \
    --ckpt outputs/runs/crispr/baseline/checkpoint-9.pth \
    --args outputs/runs/crispr/baseline/args.json \
    --n_samples 20
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils


def bench_solver(model, x_0, extra, solver_configs, n_samples):
    """Benchmark multiple solver configs on the same input."""
    results = {}
    for name, cfg in solver_configs.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking: {name}")
        print(f"  Config: {cfg}")

        model.velocity_model.reset_nfe_counter()
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                if cfg.get("method") == "dopri5":
                    samples = model.sample(
                        time_grid=torch.tensor([0.0, 1.0], device=x_0.device),
                        x_init=x_0,
                        step_size=cfg.get("step_size", 0.01),  # initial step hint
                        method="dopri5",
                        atol=cfg.get("atol", 1e-5),
                        rtol=cfg.get("rtol", 1e-5),
                        cfg_scale=cfg.get("cfg_scale", 0.2),
                        extra=extra,
                    )
                else:
                    samples = model.sample(
                        time_grid=torch.tensor([0.0, 1.0], device=x_0.device),
                        x_init=x_0,
                        method=cfg.get("method", "midpoint"),
                        step_size=cfg.get("step_size", 0.01),
                        cfg_scale=cfg.get("cfg_scale", 0.2),
                        extra=extra,
                    )

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        nfe = model.velocity_model.get_nfe()

        results[name] = {
            "nfe": nfe,
            "time_s": round(elapsed, 1),
            "samples": samples.cpu(),
            "time_per_image_s": round(elapsed / (n_samples or 1), 2),
        }
        print(f"  NFE: {nfe}, Time: {elapsed:.1f}s ({elapsed/n_samples:.1f}s/image)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--args", required=True, help="args.json from training")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs/solver_bench")
    args = parser.parse_args()

    # Load training args
    with open(args.args) as f:
        train_args = json.load(f)
    train_args = argparse.Namespace(**train_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from phenoflux.models.configs import instantiate_model
    from phenoflux.training.eval_loop import ODESolver, CFGScaledModel
    from phenoflux.training.dataloader import CellDataLoader_Eval

    print("Loading model...")
    model = instantiate_model(
        architechture="phenoflux",
        is_discrete=False,
        use_ema=getattr(train_args, "use_ema", True),
        overrides=vars(train_args),
    )
    model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Handle EMA wrapping
    state_dict = ckpt["model"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[6:]] = v
            else:
                new_sd[k] = v
        state_dict = new_sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded from epoch {ckpt['epoch']}")

    cfg_model = CFGScaledModel(model=model)
    solver = ODESolver(velocity_model=cfg_model)

    # Load a small batch of test data
    print("Loading test data...")
    data_module = CellDataLoader_Eval(train_args)
    test_loader = torch.utils.data.DataLoader(
        data_module.test_set, batch_size=args.n_samples, shuffle=False,
        num_workers=4, drop_last=False,
    )
    batch = next(iter(test_loader))
    x_real, y_trg, _ = batch["X"], batch["mols"], batch["y_id"]
    x_real_ctrl = x_real[0].to(device)
    y_trg = y_trg.long().to(device)
    z_emb = data_module.embedding_matrix(y_trg).to(device)

    extra = {"concat_conditioning": z_emb}
    if "marker_profile" in batch:
        extra["marker_profile"] = batch["marker_profile"].to(device)

    n_actual = len(x_real_ctrl)
    print(f"Test batch: {n_actual} images")

    # Define solver configs to compare
    solver_configs = {
        "midpoint_step0.01 (current)": {"method": "midpoint", "step_size": 0.01, "cfg_scale": 0.2},
        "dopri5_atol1e-5": {"method": "dopri5", "atol": 1e-5, "rtol": 1e-5, "cfg_scale": 0.2},
        "dopri5_atol1e-4": {"method": "dopri5", "atol": 1e-4, "rtol": 1e-4, "cfg_scale": 0.2},
        "midpoint_step0.02": {"method": "midpoint", "step_size": 0.02, "cfg_scale": 0.2},
        "midpoint_step0.05": {"method": "midpoint", "step_size": 0.05, "cfg_scale": 0.2},
    }

    results = bench_solver(solver, x_real_ctrl, extra, solver_configs, n_actual)

    # Save comparison grid
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison grid: [method × n_samples]
    all_images = []
    labels = []
    for name, r in results.items():
        all_images.append(r["samples"])
        labels.append(f"{name} | {r['nfe']}NFE {r['time_s']}s")

    # Save individual grids
    for name, r in results.items():
        grid = vutils.make_grid(r["samples"], nrow=5, normalize=True, value_range=(-1, 1))
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        vutils.save_image(grid, out_dir / f"{safe_name}.png")
        print(f"Saved: {out_dir / f'{safe_name}.png'}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Solver':<30} {'NFE':>6} {'Time':>8} {'s/img':>8} {'vs baseline':>12}")
    print("-" * 60)
    baseline_nfe = results["midpoint_step0.01 (current)"]["nfe"]
    baseline_time = results["midpoint_step0.01 (current)"]["time_s"]
    for name, r in results.items():
        speedup = baseline_time / r["time_s"]
        nfe_ratio = r["nfe"] / baseline_nfe
        print(f"{name:<30} {r['nfe']:>6} {r['time_s']:>7.1f}s {r['time_per_image_s']:>7.2f}s {speedup:>8.1f}x")

    print(f"\n5,120 images projection:")
    print("-" * 60)
    for name, r in results.items():
        est_time = r["time_per_image_s"] * 5120 / 60
        print(f"  {name:<30} ~{est_time:.0f} min")


if __name__ == "__main__":
    main()
