"""
PhenoFlux: two complementary conditioning modules for molecular phenotype transport.

MAC (Marker-Aware Conditioning): compresses full 18-channel MERFISH profiles into
spatially-structured tokens, then lets UNet bottleneck features query them via
cross-attention — solving WHERE the molecular state is distributed.

CCM (Channel-wise Condition Modulation): derives per-channel FiLM parameters from
the same marker tokens, modulating decoder output channels independently —
solving HOW different markers respond differently to the same perturbation.

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


# ---------------------------------------------------------------------------
# CCM — Channel-wise Condition Modulation
# ---------------------------------------------------------------------------

class ChannelConditionModulation(nn.Module):
    """Per-channel FiLM modulation from pooled marker profile tokens.

    Different marker channels respond to the same perturbation through different
    biological mechanisms (e.g. HFD: Calreticulin via ER stress, Perilipin via
    lipid accumulation, TOMM20 via mitochondrial adaptation).  CCM enables the
    decoder to apply channel-specific modulation, so each output channel can
    respond with the correct magnitude and direction.

    Design:
      - Pool marker tokens spatially (mean over N_tokens)
      - Learn per-channel scale and shift from the pooled representation
      - Apply as residual FiLM: out = x * (1 + scale) + shift
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_channels = num_channels

        self.pool_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Per-channel scale initialized near zero (identity at start)
        self.scale_proj = zero_module(nn.Linear(hidden_dim, num_channels))
        # Per-channel shift initialized near zero
        self.shift_proj = zero_module(nn.Linear(hidden_dim, num_channels))

    def forward(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel FiLM modulation.

        Args:
            tokens: (B, N_tokens, token_dim) from MarkerProfileEncoder
            x:      (B, num_channels, H, W) decoder features

        Returns:
            (B, num_channels, H, W) modulated features
        """
        # Pool spatially: (B, N_tokens, token_dim) -> (B, token_dim)
        pooled = tokens.mean(dim=1)
        h = self.pool_proj(pooled)                              # (B, hidden_dim)
        scale = self.scale_proj(h).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = self.shift_proj(h).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * (1.0 + scale) + shift
