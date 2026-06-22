# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from morphoflux.engine.models.discrete_unet import DiscreteUNetModel
from morphoflux.engine.models.ema import EMA
# from flow_matching.examples.image.models.unet_old import UNetModel
from morphoflux.engine.models.unet import UNetModel

MODEL_CONFIGS = {
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
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
    },
    "bbbc021": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
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
        "condition_dim": 1024,
    },
    "rxrx1": {
        "in_channels": 6,
        "model_channels": 128,
        "out_channels": 6,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
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
        "condition_dim": 200,
    },
    "cpg0000": {
        "in_channels": 5,
        "model_channels": 128,
        "out_channels": 5,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
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
        "condition_dim": 1224,
    },
    "perturbmulti_id": {
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
        "condition_dim": 204,
    },
    "perturbmulti_idsig": {
        # Same UNet as perturbmulti_id, but the condition is the RNA-signature CONCAT:
        # one-hot gene identity (204) + scale-balanced 209-dim MERFISH RNA signature = 413.
        # Kept as a SEPARATE arch so perturbmulti_id (one-hot, condition_dim 204) is unaffected.
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
        "condition_dim": 413,
    },
    "diet_id": {
        # Same UNet as perturbmulti_id; condition is the diet state (adlib/fasted/hfd),
        # a 3-dim one-hot, so condition_dim = 3.
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
        "condition_dim": 3,
    },
    "phenoflux_diet": {
        # PhenoFlux: marker-aware flow matching variant of diet_id.
        # Same UNet body, but adds a Marker-Aware Conditioning (MAC) module:
        # MarkerProfileEncoder (18-ch → tokens) + CrossAttention at the bottleneck,
        # enabling the model to condition on the full 18-channel MERFISH molecular
        # profile rather than just a global one-hot condition vector.
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
        "condition_dim": 3,
        # --- PhenoFlux MAC module ---
        "use_marker_cross_attn": True,
        "marker_profile_dim": 18,
        "marker_cross_attn_resolutions": [8],
    },
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    architechture: str, is_discrete: bool, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    if is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        else:
            config = MODEL_CONFIGS[architechture]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        model = UNetModel(**MODEL_CONFIGS[architechture])
    if use_ema:
        return EMA(model=model)
    else:
        return model
