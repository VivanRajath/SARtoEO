"""
test_dataset.py — Quick smoke test for the SAREODataset loader.
"""
from dataset import SAREODataset

dataset = SAREODataset("data/agri", image_size=256, augment=False)

print(f"Dataset size : {len(dataset)}")

sar, eo = dataset[0]

print(f"SAR shape    : {sar.shape}   range [{sar.min():.3f}, {sar.max():.3f}]")
print(f"EO  shape    : {eo.shape}   range [{eo.min():.3f}, {eo.max():.3f}]")

assert sar.shape == (1, 256, 256), "SAR should be [1, 256, 256]"
assert eo.shape  == (3, 256, 256), "EO should be [3, 256, 256]"
assert sar.min() >= 0.0 and sar.max() <= 1.0,  "SAR should be in [0, 1]"
assert eo.min()  >= -1.0 and eo.max() <= 1.0,  "EO should be in [-1, 1]"

print("[PASS] Dataset test passed.")