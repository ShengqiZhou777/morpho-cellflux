"""
Per-Channel Condition Decoder (PCD) — lightweight condition-aware modulation.

Data motivation:
- HFD changes Calreticulin spatial pattern more than other markers (d=0.44 vs 0.26).
- Different perturbation conditions affect different channels at different magnitudes.
- MSA learns "what changes", but the 64-dim output is blindly concatenated to
  the condition vector.  PCD decomposes this into per-channel modulation.

Design (v2 — condition-aware, bounded):
- Maps MSA output → 3 × (scale, bias) pairs — one per output channel.
- Applied as per-channel residual modulation on UNet output velocity.
- Modulation magnitude is BOUNDED via tanh (no unbounded growth) and GATED
  per-condition via a learned sigmoid gate g(cond) — so a weak perturbation
  (e.g. fasted) can independently suppress its own modulation instead of being
  dragged along by the dominant HFD gradient through a shared global scalar.
- ~5K parameters, no spatial dimensions, purely per-channel.

Why v2: the v1 single global `scale_factor` scalar grew unbounded during training
and, combined with the shared MLP being dominated by strong HFD gradients, caused
systematic OVERSHOOT on weak perturbations (fasted Perilipin/TOMM20 PGC collapse
around epoch 9; see docs/ABLATION_RESULTS.md).  The per-condition gate + tanh
bound replace that single scalar.
"""

import torch
import torch.nn as nn


class PerChannelDecoder(nn.Module):
    """Decode MSA context vector into bounded, condition-gated per-channel modulation.

    Produces ``out_channels`` independent (scale, bias) pairs — one per output
    channel — whose magnitude is bounded by tanh and scaled by a learned
    per-condition gate, allowing condition-specific per-channel adjustments
    without runaway growth on weak perturbations.
    """

    def __init__(
        self,
        msa_dim: int = 64,
        cond_dim: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        max_scale: float = 0.5,
        max_bias: float = 0.5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.max_scale = max_scale
        self.max_bias = max_bias
        input_dim = msa_dim + cond_dim

        # Lightweight: one shared MLP → per-channel raw scale+bias
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2),
        )

        # Per-condition gate g(cond) in (0,1) — replaces the v1 global scalar.
        # Negative bias init makes the gate start near 0 → modulation ≈ 0 early
        # (stable training), and each condition learns its own magnitude.
        self.gate = nn.Linear(cond_dim, out_channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)  # sigmoid(-4) ≈ 0.018

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
        raw_scale, raw_bias = x.chunk(2, dim=-1)  # each [B, out_channels]

        # Bounded magnitude — no unbounded growth.
        scale = self.max_scale * torch.tanh(raw_scale)
        bias = self.max_bias * torch.tanh(raw_bias)

        # Per-condition gate (0,1): weak perturbations can suppress modulation
        # independently of strong ones.
        g = torch.sigmoid(self.gate(cond))  # [B, out_channels]
        scale = scale * g
        bias = bias * g

        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        bias = bias.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]

        return scale, bias
