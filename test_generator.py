"""
test_generator.py — Quick shape test for the 8-level U-Net Generator.
"""
import torch
from generator import Generator

model = Generator(in_channels=1, out_channels=3, ngf=64)

x = torch.randn(2, 1, 256, 256)
y = model(x)

print("Input  shape:", x.shape)    # [2, 1, 256, 256]
print("Output shape:", y.shape)    # [2, 3, 256, 256]
assert y.shape == (2, 3, 256, 256), "Output shape mismatch!"
assert y.min() >= -1.0 and y.max() <= 1.0, "Output out of Tanh range!"
print("[PASS] Generator test passed.")