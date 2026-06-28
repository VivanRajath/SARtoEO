

import torch
import torch.nn as nn


# Building Blocks

class EncoderBlock(nn.Module):
    """
    Downsampling block: Conv(stride=2) → [BatchNorm] → LeakyReLU(0.2)
    """

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsampling block: ConvTranspose(stride=2) → BatchNorm → ReLU → [Dropout(0.5)]
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Generator

class Generator(nn.Module):
    """
    Pix2Pix U-Net Generator with 8 encoder/decoder levels and skip connections.

    Args:
        in_channels:  Input channels  (1 for grayscale SAR VV).
        out_channels: Output channels (3 for RGB EO).
        ngf:          Base number of generator filters (default 64, paper default).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, ngf: int = 64):
        super().__init__()

        f = ngf  # shorthand

        # Encoder
        self.e1 = EncoderBlock(in_channels, f,     batch_norm=False)  # 256→128
        self.e2 = EncoderBlock(f,           f * 2)                     # 128→64
        self.e3 = EncoderBlock(f * 2,       f * 4)                     # 64→32
        self.e4 = EncoderBlock(f * 4,       f * 8)                     # 32→16
        self.e5 = EncoderBlock(f * 8,       f * 8)                     # 16→8
        self.e6 = EncoderBlock(f * 8,       f * 8)                     # 8→4
        self.e7 = EncoderBlock(f * 8,       f * 8)                     # 4→2

        # Bottleneck
        # Compresses to 1×1, capturing global structure
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # Each DecoderBlock input = its own output channels (after cat = doubled)
        self.d7 = DecoderBlock(f * 8,      f * 8, dropout=True)   # 1→2
        self.d6 = DecoderBlock(f * 8 * 2,  f * 8, dropout=True)   # 2→4
        self.d5 = DecoderBlock(f * 8 * 2,  f * 8, dropout=True)   # 4→8
        self.d4 = DecoderBlock(f * 8 * 2,  f * 8)                  # 8→16
        self.d3 = DecoderBlock(f * 8 * 2,  f * 4)                  # 16→32
        self.d2 = DecoderBlock(f * 4 * 2,  f * 2)                  # 32→64
        self.d1 = DecoderBlock(f * 2 * 2,  f)                      # 64→128

        # Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),          # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_channels, H, W]  — SAR image tensor in [0, 1].

        Returns:
            [B, out_channels, H, W]    — Generated EO image in [-1, 1].
        """
        # Encode
        e1 = self.e1(x)           # [B,   64, 128, 128]
        e2 = self.e2(e1)          # [B,  128,  64,  64]
        e3 = self.e3(e2)          # [B,  256,  32,  32]
        e4 = self.e4(e3)          # [B,  512,  16,  16]
        e5 = self.e5(e4)          # [B,  512,   8,   8]
        e6 = self.e6(e5)          # [B,  512,   4,   4]
        e7 = self.e7(e6)          # [B,  512,   2,   2]

        # Bottleneck
        b = self.bottleneck(e7)   # [B,  512,   1,   1]

        # Decode with skip connections
        d7 = self.d7(b)                          # [B,  512,  2,  2]
        d7 = torch.cat([d7, e7], dim=1)          # [B, 1024,  2,  2]

        d6 = self.d6(d7)                         # [B,  512,  4,  4]
        d6 = torch.cat([d6, e6], dim=1)          # [B, 1024,  4,  4]

        d5 = self.d5(d6)                         # [B,  512,  8,  8]
        d5 = torch.cat([d5, e5], dim=1)          # [B, 1024,  8,  8]

        d4 = self.d4(d5)                         # [B,  512, 16, 16]
        d4 = torch.cat([d4, e4], dim=1)          # [B, 1024, 16, 16]

        d3 = self.d3(d4)                         # [B,  256, 32, 32]
        d3 = torch.cat([d3, e3], dim=1)          # [B,  512, 32, 32]

        d2 = self.d2(d3)                         # [B,  128, 64, 64]
        d2 = torch.cat([d2, e2], dim=1)          # [B,  256, 64, 64]

        d1 = self.d1(d2)                         # [B,   64,128,128]
        d1 = torch.cat([d1, e1], dim=1)          # [B,  128,128,128]

        return self.final(d1)                    # [B,    3,256,256]