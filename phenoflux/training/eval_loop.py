# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import random
import PIL.Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from phenoflux.training.dataloader import CellDataLoader
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from phenoflux.models.discrete_unet import DiscreteUNetModel
from phenoflux.models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from phenoflux.training.distributed import is_dist_avail_and_initialized, get_world_size, is_main_process
from phenoflux.training.edm_time import get_time_discretization
from phenoflux.training.train_loop import MASK_TOKEN
from phenoflux.training.data_utils import convert_6ch_to_3ch, convert_5ch_to_3ch, centered_noise
logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def _gather_trt2ctrl_idx(local_mapping: dict) -> dict:
    """Merge per-rank treated->control mappings before writing eval metadata."""
    if not is_dist_avail_and_initialized():
        return dict(local_mapping)

    gathered = [None for _ in range(get_world_size())]
    torch.distributed.all_gather_object(gathered, local_mapping)

    merged = {}
    for rank, rank_mapping in enumerate(gathered):
        if not rank_mapping:
            continue
        overlap = set(merged).intersection(rank_mapping)
        if overlap:
            logger.warning(
                "Overwriting %d duplicate treated->control mapping keys from rank %d",
                len(overlap),
                rank,
            )
        merged.update(rank_mapping)
    return merged


def _gather_int_sum(value: int, device: torch.device) -> int:
    """Sum an integer across distributed ranks."""
    if not is_dist_avail_and_initialized():
        return int(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, extra: dict
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.amp.autocast("cuda"), torch.no_grad():
                conditional = self.model(x, t, extra=extra)
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.amp.autocast("cuda"), torch.no_grad():
                result = self.model(x, t, extra=extra)

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
    datamodule: CellDataLoader,
    use_initial: int = 0,
    interpolate: bool = False,
):
    gc.collect()
    _model = model.module if hasattr(model, 'module') else model
    _model = getattr(_model, 'model', _model)  # unwrap EMA if present

    # Eval uses less GPU memory (no gradients/optimizer), so we can use a larger
    # batch size than training to speed up FID image generation.
    train_bs = getattr(args, 'batch_size', 32)
    eval_bs = getattr(args, 'eval_batch_size', 0) or train_bs
    if eval_bs != train_bs:
        logger.info(f"Eval batch_size: {eval_bs} (train was {train_bs})")
        data_loader = torch.utils.data.DataLoader(
            datamodule.test_set,
            batch_size=eval_bs,
            shuffle=True,  # shuffle so fid_samples span all conditions, not just the first alphabetically
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    trt2ctrl_idx = {}
    id2mol = {v: k for k, v in datamodule.mol2id.items()}
    for data_iter_step, batch in tqdm(enumerate(data_loader)):

        x_real, y_trg, y_mod = batch['X'], batch['mols'], batch['y_id']
        x_real_ctrl, x_real_trt = x_real
        x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)
        y_trg = y_trg.long().to(device)
        y_org = None
        z_emb_trg = datamodule.embedding_matrix(y_trg).to(device)
        samples = None
        labels = None

        # Build extra conditioning dict
        eval_extra = {"concat_conditioning": z_emb_trg}

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
            else:
                # Continuous sampling
                if use_initial == 1:
                    x_0 = x_real_ctrl
                elif use_initial == 2:
                    x_0 = x_real_ctrl + torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device) * args.noise_level
                else:
                    x_0 = centered_noise(x_real_ctrl.shape, getattr(args, "center_noise_sigma", 0.0), device=device)


                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0, # x_real_ctrl
                    method=args.ode_method,
                    return_intermediates=interpolate,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"]
                    if "step_size" in ode_opts
                    else None,
                    cfg_scale=args.cfg_scale,
                    extra=eval_extra,
                )
                if interpolate:
                    # Save the intermediate images
                    save_interpolation_grid(
                        synthetic_samples,
                        y_trg,
                        x_real_ctrl,
                        x_real_trt,
                        time_grid,
                        save_dir=Path(args.output_dir) / "interpolation",
                        title="Interpolation Visualization",
                    )
                    return {}
                if args.dataset_name == 'rxrx1':
                    x_real_trt = convert_6ch_to_3ch(x_real_trt)
                    synthetic_samples = convert_6ch_to_3ch(synthetic_samples)
                elif args.dataset_name == 'cpg0000':
                    x_real_trt = convert_5ch_to_3ch(x_real_trt)
                    synthetic_samples = convert_5ch_to_3ch(synthetic_samples)
                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)



            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            logger.info(
                f"{synthetic_samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )

            real_samples = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0)
            real_samples = torch.floor(real_samples * 255)
            real_samples = real_samples.to(torch.float32) / 255.0
            real_samples = real_samples[: synthetic_samples.shape[0]]


            fid_metric.update(real_samples, real=True)
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                snapshot_path = (
                    Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png"
                )
                save_image(
                    synthetic_samples,
                    fp=snapshot_path,
                )
                snapshots_saved = True
                # wandb snapshot logging
                if getattr(args, 'wandb_project', None):
                    if is_main_process():
                        import wandb
                        wandb.log({
                            "eval/snapshots": wandb.Image(str(snapshot_path)),
                            "eval/epoch": epoch,
                        })
            target_class_labels = y_trg.cpu().numpy()
            img_file_ctrl, img_file_trt = batch['file_names']
            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples" / f"epoch-{epoch}"
                    target_class_name = id2mol[target_class_labels[batch_index]]
                    save_dir = image_dir / target_class_name
                    os.makedirs(save_dir, exist_ok=True)
                    fname = Path(img_file_trt[batch_index]).name
                    if fname.endswith('.png'):
                        fname = fname[:-4]
                    image_path = save_dir / f"{fname}.png"
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)
                    trt2ctrl_idx[img_file_trt[batch_index]] = img_file_ctrl[batch_index]

        if not args.compute_fid:
            return {}

        if num_synthetic >= fid_samples:
            break

        if args.test_run:
            break
    image_dir = Path(args.output_dir) / "fid_samples"
    os.makedirs(image_dir, exist_ok=True)
    logger.info(f"Saving generated images to {image_dir}")
    # Global copy (latest eval) + per-epoch copy so each epoch's generated PNGs keep
    # their OWN treated->control pairing. The control pool is resampled every eval, so a
    # global-only file makes old epochs' images get re-paired against the newest controls,
    # adding spurious noise to per-epoch Delta-direction. Aggregate eval prefers the
    # per-epoch file (epoch-<e>/trt2ctrl_idx.json) when present.
    global_num_synthetic = _gather_int_sum(num_synthetic, device)
    merged_trt2ctrl_idx = _gather_trt2ctrl_idx(trt2ctrl_idx)
    if is_main_process():
        if global_num_synthetic < args.fid_samples:
            logger.warning(
                "Requested %d FID samples, but generated %d. Eval target split is smaller than the requested budget.",
                args.fid_samples,
                global_num_synthetic,
            )
        for jpath in (f'{image_dir}/trt2ctrl_idx.json',
                      f'{image_dir}/epoch-{epoch}/trt2ctrl_idx.json'):
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
            with open(jpath, 'w') as f:
                json.dump(merged_trt2ctrl_idx, f, indent=4)
                f.flush()
        logger.info("Saved %d treated->control mappings", len(merged_trt2ctrl_idx))
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
    return {
        "fid": float(fid_metric.compute().detach().cpu()),
        "fid_samples_requested": int(args.fid_samples),
        "fid_samples_generated": int(global_num_synthetic),
        "saved_mappings": int(len(merged_trt2ctrl_idx)),
    }


def save_interpolation_grid(
    intermediate_images: torch.Tensor,
    y_trg: torch.Tensor,
    real_ctrl: torch.Tensor,
    real_trt: torch.Tensor,
    time_grid: torch.Tensor,
    save_dir: Path,
    title: str = "Interpolation Visualization",
):
    """
    Save grids of images for a batch, showing intermediate steps, along with real ctrl and trt images.

    Args:
        intermediate_images (torch.Tensor): Tensor of intermediate images with shape (T, B, C, H, W).
        real_ctrl (torch.Tensor): Tensor of the real control images with shape (B, C, H, W).
        real_trt (torch.Tensor): Tensor of the real treatment images with shape (B, C, H, W).
        time_grid (torch.Tensor): Tensor of time steps corresponding to intermediate images.
        save_dir (Path): Directory to save the output images.
        title (str): Title for the grid of images.
    """
    # Ensure save_dir exists
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Interpolation Image saved to: ", save_dir)
    # Convert images from [C, H, W] to [H, W, C] for visualization
    def to_numpy(image_tensor):
        img = torch.clamp(image_tensor * 0.5 + 0.5, min=0.0, max=1.0)
        img = torch.floor(img * 255)
        return (img.permute(1, 2, 0).cpu().numpy()).astype("uint8")

    # Loop over batch
    batch_size = real_ctrl.shape[0]
    for b in range(batch_size):
        # Prepare images for the current sample in the batch
        real_ctrl_img = to_numpy(real_ctrl[b])
        real_trt_img = to_numpy(real_trt[b])
        intermediate_imgs = [to_numpy(intermediate_images[t, b]) for t in range(intermediate_images.shape[0])]

        # Add real_ctrl and real_trt to the image list
        images = [real_ctrl_img] + intermediate_imgs + [real_trt_img]
        labels = ["Real Ctrl"] + [f"t={t:.2f}" for t in time_grid] + ["Real Trt"]

        # Create a grid for visualization

        # Determine grid size
        num_images = len(images)
        num_cols = (num_images + 4) // 5  # Auto-calculated columns

        # Create a grid for visualization
        fig, axes = plt.subplots(5, num_cols, figsize=(2 * num_cols, 10))
        fig.suptitle(f"{title} - Sample {b} - Target {y_trg[b].cpu().numpy()}", fontsize=16)

        # Fill the grid with images and labels
        for i in range(5 * num_cols):
            row, col = divmod(i, num_cols)
            if i < num_images:
                axes[row, col].imshow(images[i])
                axes[row, col].set_title(labels[i], fontsize=8)
            axes[row, col].axis("off")  # Turn off axis for empty cells

        # Save the resulting grid image for the current batch sample
        sample_save_path = save_dir / f"sample_{b}_interpolation_grid.png"
        plt.tight_layout()
        plt.savefig(sample_save_path, dpi=300)
        plt.close()
        # Also dump raw per-cell trajectory arrays (in [-1,1]) for the polished figure
        # renderer (scripts/plot_flow_figure.py). Only runs in --interpolate mode.
        np.savez(
            save_dir / f"sample_{b}_traj.npz",
            control=real_ctrl[b].cpu().numpy(),
            trajectory=intermediate_images[:, b].cpu().numpy(),
            target=real_trt[b].cpu().numpy(),
            gene=int(y_trg[b].cpu().numpy()),
            t_grid=time_grid.detach().cpu().numpy(),
        )
