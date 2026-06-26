"""
test_disciminator.py — Quick shape test for the PatchGAN Discriminator.
"""
import torch
from discriminator import Discriminator

model = Discriminator(in_channels=4, ndf=64)

sar = torch.randn(2, 1, 256, 256)   # SAR input  [0, 1]
eo  = torch.randn(2, 3, 256, 256)   # EO input   [-1, 1]

output = model(sar, eo)

print("SAR shape   :", sar.shape)
print("EO shape    :", eo.shape)
print("Output shape:", output.shape)   # [2, 1, 30, 30] patch map
assert output.ndim == 4 and output.shape[1] == 1, "Output should be [B, 1, H', W']!"
print("[PASS] Discriminator test passed.")