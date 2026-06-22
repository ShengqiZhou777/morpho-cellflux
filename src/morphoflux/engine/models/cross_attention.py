"""
PhenoFlux Marker-Aware Conditioning (MAC) module.

MarkerProfileEncoder: compresses the full 18-channel MERFISH marker profile into
spatially-structured tokens that encode per-channel molecular state at each
spatial location.

CrossAttentionBlock: lets UNet bottleneck features query these molecular state
tokens via multi-head cross-attention, enabling the model to condition generation
on spatially-resolved marker-specific perturbation responses.

Reference:
  PhenoFlux: Marker-Aware Flow Matching for Molecular Phenotype Transport
"""

import torch
import torch.nn as nn
import numpy as np

from morphoflux.engine.models.unet import normalization, zero_module, conv_nd, checkpoint


# ---------------------------------------------------------------------------
# MarkerProfileEncoder — compresses (18, H, W) marker profiles into tokens
# ---------------------------------------------------------------------------

class MarkerProfileEncoder(nn.Module):
    """Lightweight CNN that encodes an 18-channel marker profile into a set of
    spatially-structured feature tokens for cross-attention.

    Architecture (3 conv stages):
      Stage 1: 18 -> 64  channels, stride 2  (H/2, W/2)
      Stage 2: 64 -> 128 channels, stride 2  (H/4, W/4)
      Stage 3: 128 -> 256 channels, stride 2 (H/8, W/8)
      Output:  256-d tokens, flattened to (H/8 * W/8, 256)

    The output spatial grid matches the UNet bottleneck resolution (128/8 = 16),
    giving 256 tokens, each a 256-dim embedding of a local region's marker profile.
    """

    def __init__(self, in_channels: int = 18, hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            # Stage 1: 128 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            # Stage 2: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # Stage 3: 32 -> 16
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
        )

        # Learnable positional encoding for the 16x16 spatial grid
        self.pos_embed = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode marker profile into token sequence.

        Args:
            x: (B, 18, H, W) full marker profile in [0, 1]

        Returns:
            tokens: (B, N_tokens, hidden_dim) where N_tokens = H/8 * W/8
        """
        B = x.shape[0]
        feats = self.encoder(x)           # (B, hidden_dim, H/8, W/8)
        _, C, h, w = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, h*w, hidden_dim)
        tokens = tokens + self.pos_embed[:, : h * w, :]
        return tokens


# ---------------------------------------------------------------------------
# CrossAttentionBlock — Q from UNet features, K/V from condition tokens
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Cross-attention: UNet feature maps (Q) attend to marker profile tokens (K, V).

    Follows the same pattern as the existing AttentionBlock in unet.py:
    GroupNorm -> projection -> attention -> zero-init output projection -> residual add.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int = 256,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        assert channels % num_heads == 0, (
            f"channels {channels} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = channels // num_heads

        self.norm_q = normalization(channels)
        self.norm_kv = nn.LayerNorm(context_dim)

        # Q projection from UNet features
        self.to_q = nn.Linear(channels, channels, bias=False)
        # K, V projections from condition tokens
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)

        # Zero-initialized output projection (stable training)
        self.proj_out = zero_module(nn.Linear(channels, channels))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward, (x, context), self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention forward.

        Args:
            x: (B, C, H, W) UNet feature map
            context: (B, N_ctx, context_dim) marker profile tokens

        Returns:
            (B, C, H, W) feature map with cross-attended residual
        """
        B, C, H, W = x.shape
        N_ctx = context.shape[1]

        # Q: spatial features — norm on (B, C, -1), then transpose to (B, HW, C) for linear proj
        x_flat = x.reshape(B, C, -1)
        q = self.to_q(self.norm_q(x_flat).transpose(1, 2))
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, H*W, head_dim)

        # K, V: context tokens -> (B, num_heads, N_ctx, head_dim)
        ctx_norm = self.norm_kv(context)
        k = self.to_k(ctx_norm).reshape(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx_norm).reshape(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale      # (B, heads, HW, N_ctx)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                               # (B, heads, HW, head_dim)

        # Merge heads -> project -> residual
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return x + out
