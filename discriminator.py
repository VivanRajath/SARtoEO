"""
discriminator.py — PatchGAN Discriminator (Isola et al., 2017).

Architecture:
  Input: SAR (1ch) concatenated with EO (3ch) → 4-channel input
  Four DiscriminatorBlocks of increasing filters
  Final Conv2d → patch-wise real/fake score map

  The PatchGAN classifies overlapping 70×70 patches as real/fake, not the
  whole image. This encourages high-frequency texture realism.

Args:
    in_channels: Total input channels = SAR channels + EO channels (default 4 = 1+3).
    ndf:         Base number of discriminator filters (default 64).
"""

import torch
import torch.nn as nn


# ── Building Block ─────────────────────────────────────────────────────────────

class DiscriminatorBlock(nn.Module):
    """
    Conv(stride) → [BatchNorm] → LeakyReLU(0.2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, batch_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Discriminator ──────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator.

    Concatenates the SAR conditioning image with the EO image (real or fake)
    along the channel dimension, then produces a spatial map of real/fake scores.

    Args:
        in_channels: Total input channels (SAR + EO). Default 4 (1 + 3).
        ndf:         Base number of discriminator filters. Default 64.
    """

    def __init__(self, in_channels: int = 4, ndf: int = 64):
        super().__init__()

        f = ndf

        self.model = nn.Sequential(
            # Block 1: no BatchNorm on first layer (paper convention)
            DiscriminatorBlock(in_channels, f,     stride=2, batch_norm=False),   # 256→128
            DiscriminatorBlock(f,           f * 2, stride=2),                      # 128→64
            DiscriminatorBlock(f * 2,       f * 4, stride=2),                      # 64→32
            # stride=1 on last conv block (PatchGAN design)
            DiscriminatorBlock(f * 4,       f * 8, stride=1),                      # 32→31
            # Final layer: 1-channel patch score, no activation (BCEWithLogitsLoss)
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),              # 31→30
        )

    def forward(self, sar: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: [B, 1, H, W]  — SAR conditioning image in [0, 1].
            eo:  [B, 3, H, W]  — EO image (real or fake) in [-1, 1].

        Returns:
            [B, 1, H', W']  — Patch-wise logits (no sigmoid; use BCEWithLogitsLoss).
        """
        x = torch.cat([sar, eo], dim=1)   # [B, 4, H, W]
        return self.model(x)