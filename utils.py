

import os
import platform
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


# Folder Discovery

def find_subdir(root: str, candidates: list) -> str:
    """
    Case-agnostic subfolder lookup.

    Searches `root` for any direct child folder whose name matches one of the
    `candidates` strings (case-insensitive). Returns the full path of the first
    match found.

    Args:
        root:       Parent directory to search.
        candidates: Ordered list of acceptable folder names (e.g. ["S1","s1","sar"]).

    Returns:
        Absolute path to the matched subfolder.

    Raises:
        FileNotFoundError: If `root` doesn't exist or no candidate folder is found.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root directory not found: '{root}'")

    entries = os.listdir(root)
    candidates_lower = {c.lower(): c for c in candidates}

    for entry in entries:
        if entry.lower() in candidates_lower and os.path.isdir(os.path.join(root, entry)):
            return os.path.join(root, entry)

    available = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    raise FileNotFoundError(
        f"Could not find any of {candidates} inside '{root}'.\n"
        f"  Available subdirectories: {available}\n"
        f"  Tip: Your SAR folder should be named one of {candidates}."
    )


# Reproducibility

def set_seed(seed: int):
    """
    Set random seed globally for torch, numpy, and Python random.
    Call once at the start of training for reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Tensor Utilities

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from [-1, 1] range to [0, 1] range.

    The generator outputs values in [-1, 1] (Tanh activation). This function
    converts them to [0, 1] before saving or displaying images.

    Args:
        tensor: Any-shape float tensor in [-1, 1].

    Returns:
        Float tensor in [0, 1], same shape.
    """
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0


# Qualitative Visualisation

def save_triplet_grid(
    sar_tensor:  torch.Tensor,
    pred_tensor: torch.Tensor,
    gt_tensor:   torch.Tensor,
    out_path:    str,
):
    """
    Save a side-by-side comparison grid:
        [ SAR Input (grey) | Generated EO | Ground Truth EO ]

    Args:
        sar_tensor:  Shape [1, 1, H, W] or [1, 3, H, W].  Range [0, 1].
        pred_tensor: Shape [1, 3, H, W].                   Range [-1, 1] (raw generator output).
        gt_tensor:   Shape [1, 3, H, W].                   Range [-1, 1].
        out_path:    Destination file path (PNG).
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Ensure SAR is 3-channel for visual alignment in grid
    sar = sar_tensor.float()
    if sar.dim() == 3:
        sar = sar.unsqueeze(0)
    if sar.shape[1] == 1:
        sar = sar.repeat(1, 3, 1, 1)   # grayscale → RGB grey

    pred_disp = denormalize(pred_tensor.float())
    gt_disp   = denormalize(gt_tensor.float())

    # Concatenate panels horizontally (along width dimension)
    grid = torch.cat([sar, pred_disp, gt_disp], dim=3)   # [1, 3, H, 3W]
    save_image(grid, out_path, normalize=False)


# DataLoader Workers

def get_num_workers() -> int:
    """
    Return the recommended number of DataLoader worker processes.

    - Windows: 0  (multiprocessing DataLoader is unstable on Windows with PyTorch)
    - Linux / macOS / Colab: 2
    """
    return 0 if platform.system() == "Windows" else 2
