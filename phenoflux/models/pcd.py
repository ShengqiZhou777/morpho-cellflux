"""
Per-Channel Condition Decoder (PCD) — lightweight per-channel modulation.

Data motivation:
- HFD changes Calreticulin spatial pattern more than TOMM20 (d=0.44 vs 0.26).
- Different perturbation conditions affect different channels differently.
- MSA learns "what changes", but the 64-dim output is blindly concatenated to
  the condition vector.  PCD decomposes this into per-channel modulation.

Design:
- Maps MSA output → 3 × (scale, bias) pairs — one per output channel.
- Applied as per-channel residual modulation on UNet output velocity.
- ~5K parameters, no spatial dimensions, purely per-channel.
"""

import torch
import torch.nn as nn


class PerChannelDecoder(nn.Module):
    """Decode MSA context vector into per-channel modulation.

    Produces 3 independent (scale, bias) pairs — one for each output channel
    (Calreticulin, Perilipin, TOMM20).  This allows the model to learn
    condition-specific per-channel adjustments.
    """

    def __init__(
        self,
        msa_dim: int = 64,
        cond_dim: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels
        input_dim = msa_dim + cond_dim

        # Lightweight: one shared MLP → per-channel scale+bias
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2),
        )

        # Learnable initial scale — starts near 0 for stable training
        self.scale_factor = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        msa_out: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            msa_out: [B, msa_dim] MSA context vector
            cond: [B, cond_dim] perturbation one-hot

        Returns:
            scale: [B, out_channels, 1, 1] per-channel multiplicative factor
            bias:  [B, out_channels, 1, 1] per-channel additive bias
        """
        x = torch.cat([msa_out, cond], dim=-1)  # [B, msa_dim + cond_dim]
        x = self.proj(x)  # [B, out_channels * 2]
        scale, bias = x.chunk(2, dim=-1)  # each [B, out_channels]

        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        bias = bias.unsqueeze(-1).unsqueeze(-1)     # [B, 3, 1, 1]

        return scale * self.scale_factor, bias * self.scale_factor
