"""
Marker Self-Attention (MSA) — lightweight module for learning per-channel
marker co-regulation relationships.

Replaces naive 18ch concat in the info_control baseline with learned
marker interactions: 18 marker descriptors → self-attention → context-aware
condition vector.

Key design decisions (from data analysis):
- No spatial tokens — cell内部的空间变异是表达量的5倍，但r=0.92表示表达量已
  是空间组织的强代理。MSA只建模marker间的共调控，不浪费计算在空间结构上。
- Condition-gated attention — 不同扰动关注不同marker子集。
- 18 tokens × small d → 极轻量，生成速度与info_control完全相同。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkerDescriptor(nn.Module):
    """Extract per-channel descriptors from 18ch marker profile.

    For each of 18 channels: mean (expression level), std (spatial dispersion),
    puncta (fraction of pixels > 2*mean, capturing aggregation).
    """

    def __init__(self, n_channels: int = 18):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, mp: torch.Tensor) -> torch.Tensor:
        # mp: [B, 18, H, W]
        B, C, H, W = mp.shape

        # Mean expression per channel
        ch_mean = mp.mean(dim=[2, 3])  # [B, 18]

        # Spatial std per channel (normalized by mean for scale invariance)
        ch_std = mp.std(dim=[2, 3])  # [B, 18]
        ch_cv = ch_std / (ch_mean + 1e-6)  # coefficient of variation

        # Puncta score: fraction of high-intensity pixels
        threshold = ch_mean.unsqueeze(-1).unsqueeze(-1) * 2.0  # [B, 18, 1, 1]
        puncta = (mp > threshold).float().mean(dim=[2, 3])  # [B, 18]

        # Stack descriptors: [B, 18, 3]
        descriptors = torch.stack([ch_mean, ch_cv, puncta], dim=-1)
        return descriptors


class MarkerSelfAttention(nn.Module):
    """Multi-head self-attention over 18 marker tokens.

    Args:
        n_markers: number of marker channels (18)
        d_model: token embedding dimension
        n_heads: number of attention heads
        n_layers: number of self-attention layers
        condition_dim: dimension of perturbation condition (3 for diet)
        output_dim: dimension of output condition vector (concatenated to UNet condition)
    """

    def __init__(
        self,
        n_markers: int = 18,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        condition_dim: int = 3,
        output_dim: int = 64,
    ):
        super().__init__()
        self.n_markers = n_markers
        self.d_model = d_model

        # Per-channel descriptor extractor
        self.descriptor = MarkerDescriptor(n_markers)

        # Project 3 descriptors → d_model
        self.token_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for marker identity (learnable)
        self.marker_pos = nn.Parameter(torch.randn(1, n_markers, d_model) * 0.02)

        # Transformer encoder over markers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Condition gate: perturbation condition → per-marker attention bias
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Output: pool marker tokens → condition vector
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        marker_profile: torch.Tensor,
        cond: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        B = marker_profile.shape[0]

        # 1. Extract per-channel descriptors
        desc = self.descriptor(marker_profile)  # [B, 18, 3]

        # 2. Project to tokens
        tokens = self.token_proj(desc)  # [B, 18, d_model]
        tokens = tokens + self.marker_pos  # add learnable marker identity

        # 3. Self-attention over markers
        attn_output = self.transformer(tokens)  # [B, 18, d_model]

        # 4. Condition-gated pooling: condition determines which markers to attend to
        cond_gate = self.cond_proj(cond)  # [B, d_model]
        cond_gate = cond_gate.unsqueeze(1)  # [B, 1, d_model]
        weights = torch.sigmoid((attn_output * cond_gate).sum(dim=-1))  # [B, 18]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        pooled = (attn_output * weights.unsqueeze(-1)).sum(dim=1)  # [B, d_model]

        # 5. Output projection
        marker_context = self.output_proj(pooled)  # [B, output_dim]

        if return_attention:
            return marker_context, weights
        return marker_context


def create_msa_module(
    n_markers: int = 18,
    d_model: int = 64,
    output_dim: int = 64,
    condition_dim: int = 3,
) -> MarkerSelfAttention:
    """Factory function for creating MSA module."""
    return MarkerSelfAttention(
        n_markers=n_markers,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        condition_dim=condition_dim,
        output_dim=output_dim,
    )
