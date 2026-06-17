from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-style embedding for scalar flow time values in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dim must be at least 2")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionedResBlock(nn.Module):
    """Residual block modulated by a time-plus-condition embedding."""

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels * 2),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.embedding(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class ConditionalUNet2D(nn.Module):
    """Small conditional UNet that predicts CellFlux flow velocities."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conditions: int,
        hidden_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        embedding_dim: int = 128,
    ):
        super().__init__()
        if num_conditions < 1:
            raise ValueError("num_conditions must be positive")
        if not channel_mults:
            raise ValueError("channel_mults cannot be empty")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_conditions = int(num_conditions)
        self.hidden_channels = int(hidden_channels)
        self.channel_mults = tuple(int(v) for v in channel_mults)
        self.embedding_dim = int(embedding_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition_embedding = nn.Embedding(num_conditions, embedding_dim)
        self.input = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        channels = [hidden_channels * mult for mult in self.channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = hidden_channels
        for idx, channels_out in enumerate(channels):
            self.down_blocks.append(ConditionedResBlock(current, channels_out, embedding_dim))
            current = channels_out
            if idx < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(current, current, kernel_size=4, stride=2, padding=1)
                )

        self.middle = ConditionedResBlock(current, current, embedding_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        skip_channels = list(reversed(channels))
        for idx, channels_skip in enumerate(skip_channels):
            if idx > 0:
                self.upsamples.append(
                    nn.ConvTranspose2d(current, current, kernel_size=4, stride=2, padding=1)
                )
            self.up_blocks.append(
                ConditionedResBlock(current + channels_skip, channels_skip, embedding_dim)
            )
            current = channels_skip

        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(current), current),
            nn.SiLU(),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim > 1:
            condition = condition.squeeze(-1)
        condition = condition.long().clamp(min=0, max=self.num_conditions - 1)
        emb = self.time_embedding(t) + self.condition_embedding(condition)

        h = self.input(x)
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.down_blocks):
            h = block(h, emb)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.middle(h, emb)

        for idx, block in enumerate(self.up_blocks):
            if idx > 0:
                h = self.upsamples[idx - 1](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)

        return self.output(h)
