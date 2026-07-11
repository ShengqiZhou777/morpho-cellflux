# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import logging

from phenoflux.models.configs import MODEL_CONFIGS
from torchdiffeq._impl.odeint import SOLVERS

logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=921, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=0.2,
        help="Probability to drop conditioning during training",
    )
    parser.add_argument(
        "--skewed_timesteps",
        action="store_true",
        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        default=list(MODEL_CONFIGS.keys())[0],
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_path",
        default="./data/image_generation",
        type=str,
        help="imagenet root folder with train, val and test subfolders",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    # wandb logging (optional — only enabled when --wandb_project is set)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project name; enables wandb logging when set",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity/team name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run name; defaults to output_dir basename",
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="+",
        type=str,
        default=None,
        help="wandb tags for the run",
    )
    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=50,
        help="Frequency (in steps) for wandb logging within an epoch",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="Batch size for eval image generation (0 = use training batch_size). "
             "Eval uses less GPU memory (no gradients), so a larger value speeds up FID generation.",
    )
    parser.add_argument(
        "--ode_method",
        default="dopri5",
        choices=list(SOLVERS.keys()) + ["edm_heun"],
        help="ODE solver used to generate samples. dopri5 is adaptive 5th-order (recommended, ~44 NFE).",
    )
    parser.add_argument(
        "--ode_options",
        default='{"atol": 1e-5, "rtol": 1e-5}',
        type=json.loads,
        help="ODE solver options. dopri5 uses atol/rtol; midpoint uses step_size.",
    )
    parser.add_argument(
        "--sym",
        default=0.0,
        type=float,
        help="Symmetric term for sampling the discrete flow.",
    )
    parser.add_argument(
        "--temp",
        default=1.0,
        type=float,
        help="Temperature for sampling the discrete flow.",
    )
    parser.add_argument(
        "--sym_func",
        action="store_true",
        help="Use a fixed function for the symmetric term in the discrete flow.",
    )
    parser.add_argument(
        "--sampling_dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Solver dtype for sampling the discrete flow.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.2,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--fid_samples",
        default=1000,
        type=int,
        help="number of synthetic samples for FID evaluations during training. "
             "For final paper metrics, use --fid_samples 5000 --eval_only with best checkpoint.",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )
    parser.add_argument(
        "--eval_frequency",
        default=50,
        type=int,
        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.",
    )
    parser.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save all samples generated for FID computation.",
    )
    parser.add_argument(
        "--marker_aux_weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary marker expression loss (0 = disabled). "
        "Adds MSE between approximated target channel means and ground-truth "
        "target channel means, encouraging the model to respect marker-level expression.",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=0,
        type=int,
        help="Stop training if loss does not improve for N consecutive epochs. 0 = disabled.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        default=1e-6,
        type=float,
        help="Minimum absolute loss decrease to count as improvement for early stopping.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )
    parser.add_argument(
        "--discrete_flow_matching",
        action="store_true",
        help="Train discrete flow matching model.",
    )
    parser.add_argument(
        "--discrete_fm_steps",
        default=1024,
        type=int,
        help="Number of sampling steps for discrete FM.",
    )
    parser.add_argument(
        "--config",
        default="bbbc021_all",
        type=str,
        help="Path to a configuration file with additional arguments.",
    )
    parser.add_argument(
        "--data_index",
        default="",
        type=str,
        help="Override data_index_path from config YAML. Use e.g. 'data/processed/diet/index_diet_5k.csv' for 5k subset.",
    )
    parser.add_argument(
        "--use_initial",
        default=0,
        type=int,
        help="Use the initial state as input to the ODE (0 no use, 1 use, 2 use w/ noise).",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Save the interpolate image between initial and target",
    )
    parser.add_argument(
        "--iter_ctrl",
        action="store_true",
        help="Iterate over control samples or target image",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.2,
        help="Noise level for the control image.",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--center_noise_sigma",
        type=float,
        default=0.0,
        help="Standard deviation for center-weighted Gaussian noise envelope "
             "(in normalized [-1,1] image coordinates). 0 = uniform noise (default). "
             ">0 enables centered noise for use_initial=0 mode to bias generation "
             "toward a single centered cell. Suggested: 0.3–0.5.",
    )
    parser.add_argument(
        "--foreground_loss",
        action="store_true",
        help="Use foreground-weighted flow-matching MSE for sparse marker crops.",
    )
    parser.add_argument(
        "--foreground_threshold",
        type=float,
        default=0.05,
        help="Raw [0,1] marker-intensity threshold for dynamic foreground masks.",
    )
    parser.add_argument(
        "--foreground_weight",
        type=float,
        default=5.0,
        help="Loss weight for foreground pixels when --foreground_loss is enabled.",
    )
    parser.add_argument(
        "--background_weight",
        type=float,
        default=0.1,
        help="Loss weight for background pixels when --foreground_loss is enabled.",
    )

    parser.add_argument(
        "--morph_loss_weight",
        type=float,
        default=0.0,
        help="Weight for image-feature reconstruction loss on predicted target (0=disabled).",
    )

    parser.add_argument(
        "--gan_weight",
        type=float,
        default=0.0,
        help="Weight for GAN adversarial loss (PatchGAN discriminator). 0=disabled.",
    )

    # --- Pairing strategy (ADR-002 data-quality improvement) ---
    parser.add_argument(
        "--pairing_mode",
        type=str,
        default="batch_random",
        choices=["batch_random", "merfish_nn", "cluster_match"],
        help="Ctrl-trt pairing: batch_random (default), merfish_nn (MERFISH NN), cluster_match (same cluster)",
    )
    parser.add_argument(
        "--pairing_path",
        type=str,
        default=None,
        help="Path to precomputed pairing JSON (merfish_nn) or index CSV (cluster_match)",
    )
    parser.add_argument(
        "--augment_strength",
        type=str,
        default="default",
        choices=["default", "strong", "none"],
        help="Augmentation strength: default, strong (jitter+intensity+noise), none",
    )

    return parser
