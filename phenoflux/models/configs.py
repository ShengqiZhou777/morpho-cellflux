# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""PhenoFlux model registry.

Simplified UNet for RGB microalgae data (no molecular priors).
"""

from typing import Union

from phenoflux.models.discrete_unet import DiscreteUNetModel
from phenoflux.models.ema import EMA
from phenoflux.models.unet import UNetModel

# ── Shared UNet body ──────────────────────────────────────────────
_SHARED_UNET = {
    "in_channels": 3,
    "model_channels": 128,
    "out_channels": 3,
    "num_res_blocks": 4,
    "attention_resolutions": [4],
    "dropout": 0.3,
    "channel_mult": [2, 2, 2],
    "conv_resample": False,
    "dims": 2,
    "num_classes": None,
    "use_checkpoint": False,
    "num_heads": 1,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "resblock_updown": False,
    "use_new_attention_order": True,
    "with_fourier_features": False,
}

_SMALL_UNET = {**_SHARED_UNET, "model_channels": 64, "num_res_blocks": 2, "channel_mult": [1, 2, 2]}
_MEDIUM_UNET = {**_SHARED_UNET, "model_channels": 96, "num_res_blocks": 3, "channel_mult": [1, 2, 2]}

MODEL_CONFIGS = {
    "phenoflux": {
        **_SHARED_UNET,
        "base_condition_dim": 0,
        "condition_dim": 0,
    },
    "phenoflux_medium": {
        **_MEDIUM_UNET,
        "base_condition_dim": 0,
        "condition_dim": 0,
    },
    "phenoflux_small": {
        **_SMALL_UNET,
        "base_condition_dim": 0,
        "condition_dim": 0,
    },
}


def instantiate_model(
    architechture: str,
    is_discrete: bool,
    use_ema: bool,
    overrides: dict | None = None,
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    config = dict(MODEL_CONFIGS[architechture])  # copy — don't mutate the shared dict

    # ── Apply YAML overrides ──
    if overrides:
        for key in {"base_condition_dim"}:
            if key in overrides:
                config[key] = overrides[key]

    # ── Compute condition_dim ──
    config["condition_dim"] = config["base_condition_dim"]

    if is_discrete:
        model = DiscreteUNetModel(vocab_size=257, **config)
    else:
        model = UNetModel(**config)

    if use_ema:
        return EMA(model=model)
    return model
