

import os
import random
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

from utils import find_subdir


# Supported image file extensions
_VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Candidate folder names (tried in order, case-insensitive)
_SAR_CANDIDATES = ["S1", "s1", "SAR", "sar", "sentinel1", "Sentinel1"]
_EO_CANDIDATES  = ["S2", "s2", "EO",  "eo",  "sentinel2", "Sentinel2", "RGB", "rgb"]


class SAREODataset(Dataset):
    """
    PyTorch Dataset for paired SAR (Sentinel-1) and EO (Sentinel-2) image patches.

    Expected directory layout:
        root_dir/
            S1/   (or s1/, sar/, …)   ← Sentinel-1 VV SAR patches
            S2/   (or s2/, eo/,  …)   ← Sentinel-2 RGB EO patches

    Filename pairing strategies (tried in order):
        1. Replace '_s1_' with '_s2_'  — Kaggle terrain dataset naming
        2. Replace 's1' with 's2'      — generic naming
        3. Identical filename          — SEN1-2 / SEN12MS style

    Args:
        root_dir:   Path to dataset root containing SAR and EO subfolders.
        image_size: All images are resized to image_size × image_size (default 256).
        augment:    If True, random h-flip and v-flip are applied (train only).
    """

    def __init__(self, root_dir: str, image_size: int = 256, augment: bool = False):
        self.image_size = image_size
        self.augment    = augment

        # Discover SAR and EO directories
        self.sar_dir = find_subdir(root_dir, _SAR_CANDIDATES)
        self.eo_dir  = find_subdir(root_dir, _EO_CANDIDATES)

        # Collect all SAR files
        self.sar_files = sorted([
            f for f in os.listdir(self.sar_dir)
            if os.path.splitext(f)[1].lower() in _VALID_EXTS
        ])

        if not self.sar_files:
            raise FileNotFoundError(
                f"No valid image files (.png/.jpg/.tif) found in SAR directory:\n"
                f"  {self.sar_dir}"
            )

        # Build EO filename lookup set
        self._eo_lookup = {
            f for f in os.listdir(self.eo_dir)
            if os.path.splitext(f)[1].lower() in _VALID_EXTS
        }

        # Shared resize transform
        self._resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    # Internal helpers

    def _get_eo_filename(self, sar_filename: str) -> str:
        """
        Derive the paired EO filename from a SAR filename.
        Tries three strategies and raises if none match.
        """
        # Strategy 1: Kaggle _s1_ / _s2_ naming convention
        candidate = sar_filename.replace("_s1_", "_s2_")
        if candidate in self._eo_lookup:
            return candidate

        # Strategy 2: generic s1 → s2 swap
        candidate = sar_filename.replace("s1", "s2")
        if candidate in self._eo_lookup:
            return candidate

        # Strategy 3: identical filename (SEN1-2 style)
        if sar_filename in self._eo_lookup:
            return sar_filename

        raise FileNotFoundError(
            f"No matching EO file found for SAR file: '{sar_filename}'\n"
            f"  Tried: '{sar_filename.replace('_s1_', '_s2_')}', "
            f"'{sar_filename.replace('s1', 's2')}', '{sar_filename}'\n"
            f"  EO directory: {self.eo_dir}"
        )

    # Dataset interface

    def __len__(self) -> int:
        return len(self.sar_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sar_filename = self.sar_files[idx]
        eo_filename  = self._get_eo_filename(sar_filename)

        sar_path = os.path.join(self.sar_dir, sar_filename)
        eo_path  = os.path.join(self.eo_dir,  eo_filename)

        # Load — SAR as greyscale, EO as RGB (handles any source bit-depth)
        sar_img = Image.open(sar_path).convert("L")
        eo_img  = Image.open(eo_path).convert("RGB")

        # Resize to target size (gracefully handles any input resolution)
        sar_img = self._resize(sar_img)
        eo_img  = self._resize(eo_img)

        # Augmentation
        # CRITICAL: same random state applied to both images to preserve alignment
        if self.augment:
            if random.random() > 0.5:
                sar_img = TF.hflip(sar_img)
                eo_img  = TF.hflip(eo_img)
            if random.random() > 0.5:
                sar_img = TF.vflip(sar_img)
                eo_img  = TF.vflip(eo_img)

        # Normalisation
        # SAR:  PIL [0,255] → Tensor [0.0, 1.0]
        sar_tensor = TF.to_tensor(sar_img)

        # EO:   PIL [0,255] → Tensor [0.0, 1.0] → [-1.0, 1.0]
        # The [-1,1] range matches the generator's Tanh output, so MSE/L1 is
        # computed in the same space as the network's output.
        eo_tensor = TF.to_tensor(eo_img)
        eo_tensor = TF.normalize(eo_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return sar_tensor, eo_tensor