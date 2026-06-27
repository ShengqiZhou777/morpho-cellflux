# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""PhenoFlux model registry.

One UNet body, configurable molecular prior modules:
  Diet:   MSA (Marker Self-Attention) + PCD (Per-Channel Decoder)
  CRISPR: PCGE (Program-Conditioned Gene Embedding)

Architecture:      phenoflux  (the only MODEL_CONFIGS entry)
Molecular priors:  set via YAML config flags (use_msa, use_pcd, use_pcge, use_marker_profile)
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

MODEL_CONFIGS = {
    "phenoflux": {
        **_SHARED_UNET,
        # ── Molecular prior flags (overridden by YAML config) ──
        "base_condition_dim": 0,   # one-hot dim: 3 (diet) or 204 (crispr)
        # Diet molecular priors (MSA + PCD operate on 18ch MERFISH profiles)
        "use_msa": False,          # Marker Self-Attention → +64 dims
        "use_pcd": False,          # Per-Channel Decoder (requires MSA)
        "use_marker_profile": False,  # naive 18ch concat → +18 dims (info control)
        "msa_output_dim": 64,
        # CRISPR molecular prior
        "use_pcge": False,         # Program-Conditioned Gene Embedding
        # condition_dim is computed automatically below
    },
}


# Molecular prior keys that can be overridden by YAML config
_MOLECULAR_PRIOR_KEYS = {
    "base_condition_dim", "use_msa", "use_pcd", "use_marker_profile",
    "use_pcge", "msa_output_dim",
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

    # ── Apply YAML overrides for molecular prior flags ──
    if overrides:
        for key in _MOLECULAR_PRIOR_KEYS:
            if key in overrides:
                config[key] = overrides[key]

    # ── Compute condition_dim from molecular prior flags ──
    condition_dim = config["base_condition_dim"]
    if config["use_msa"]:
        condition_dim += config["msa_output_dim"]
    elif config["use_marker_profile"]:
        condition_dim += 18  # naive concat of 18ch marker means
    config["condition_dim"] = condition_dim

    if is_discrete:
        model = DiscreteUNetModel(vocab_size=257, **config)
    else:
        # Filter out keys that are NOT UNetModel dataclass fields.
        # Stored as model attributes for train/eval loop detection.
        _NON_UNET_KEYS = {"use_pcge", "use_marker_profile"}
        unet_config = {k: v for k, v in config.items() if k not in _NON_UNET_KEYS}
        model = UNetModel(**unet_config)
        for key in _NON_UNET_KEYS:
            if config.get(key):
                setattr(model, key, True)

    if use_ema:
        return EMA(model=model)
    return model
