"""
train.py — Pix2Pix SAR-to-EO training script.

Features:
  - 3-way reproducible train / val / test split (indices saved to CSV)
  - Ablation logging: L1-only loss logged alongside GAN+L1 every epoch
  - Auto num_workers: 0 on Windows, 2 on Linux/Colab
  - Checkpoint save and resume
  - Per-epoch CSV log + loss curve PNG
  - Qualitative triplet images (SAR | Pred EO | GT EO) saved periodically
  - Automatic test-set L1 evaluation + 5 triplet examples after training
  - [Colab-resilient] Google Drive checkpoint mirroring — survives runtime restarts
  - [Colab-resilient] SIGTERM + atexit emergency save on disconnect / timeout
  - [Colab-resilient] In-epoch periodic save every N batches

Usage:
    python train.py [--config config.yaml]

Colab usage (point checkpoints at Google Drive so they survive restarts):
    !python train.py --config config.yaml \\
        --checkpoint_dir /content/drive/MyDrive/SAR2EO/checkpoints \\
        --gdrive_checkpoint_dir /content/drive/MyDrive/SAR2EO/checkpoints

    # Or just pass gdrive_checkpoint_dir as a backup mirror:
    !python train.py --config config.yaml \\
        --gdrive_checkpoint_dir /content/drive/MyDrive/SAR2EO/checkpoints

After training, download checkpoints/checkpoint_latest.pth (or a numbered
checkpoint) and use infer.py locally.
"""

import atexit
import csv
import json
import os
import shutil
import signal
import sys
import argparse
import platform

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Colab and headless servers)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import yaml

from dataset      import SAREODataset
from generator    import Generator
from discriminator import Discriminator
from utils        import set_seed, denormalize, save_triplet_grid, get_num_workers


# ══════════════════════════════════════════════════════════════════════════════
# Configuration defaults (overridden by config.yaml via load_config())
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH      = "data/agri"
IMAGE_SIZE        = 256
BATCH_SIZE        = 2
NUM_EPOCHS        = 100
LEARNING_RATE     = 0.0002
BETAS             = (0.5, 0.999)
LAMBDA_L1         = 100
AUGMENT           = True
ABLATION_LOG      = True
NGF               = 64
NDF               = 64

TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15
SEED              = 42

CHECKPOINT_EVERY  = 5
SAMPLE_EVERY      = 5
CHECKPOINT_DIR    = "checkpoints"
OUTPUT_DIR        = "outputs"

# ── Colab-resilience settings ─────────────────────────────────────────────────
# Google Drive path to mirror checkpoints so they survive Colab runtime restarts.
# Set via CLI: --gdrive_checkpoint_dir /content/drive/MyDrive/SAR2EO/checkpoints
# or in config.yaml:  gdrive_checkpoint_dir: "/content/drive/MyDrive/SAR2EO/checkpoints"
# Leave as None (or omit from config) when NOT running on Colab.
GDRIVE_CHECKPOINT_DIR   = None

# Save an emergency in-epoch checkpoint every N batches.
# Useful for very long epochs — 0 disables it.
EMERGENCY_SAVE_EVERY_N  = 0

NUM_WORKERS       = get_num_workers()
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Derived paths (recomputed after loading config)
LOG_CSV            = os.path.join(OUTPUT_DIR, "training_log.csv")
LOSS_CURVE         = os.path.join(OUTPUT_DIR, "loss_curve.png")
SPLIT_CSV          = os.path.join(OUTPUT_DIR, "data_split.csv")
TEST_METRICS_JSON  = os.path.join(OUTPUT_DIR, "test_metrics.json")

# ── Runtime state used by the emergency-save handler ─────────────────────────
# These are populated during training so the signal handler can flush them.
_emergency_state: dict = {}   # filled in train() with live model/optimizer refs


# ══════════════════════════════════════════════════════════════════════════════
# Config Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    """Parse CLI args, load config.yaml, and overwrite global constants."""
    global DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    global BETAS, LAMBDA_L1, AUGMENT, ABLATION_LOG, NGF, NDF
    global TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED
    global CHECKPOINT_EVERY, SAMPLE_EVERY, CHECKPOINT_DIR, OUTPUT_DIR
    global NUM_WORKERS, DEVICE
    global LOG_CSV, LOSS_CURVE, SPLIT_CSV, TEST_METRICS_JSON
    global GDRIVE_CHECKPOINT_DIR, EMERGENCY_SAVE_EVERY_N

    parser = argparse.ArgumentParser(description="Train Pix2Pix SAR-to-EO model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    # ── CLI overrides: these always win over config.yaml ─────────────────────
    # Useful for Colab where the dataset lives in Google Drive:
    #   !python train.py --dataset_path /content/drive/MyDrive/SAR2EO/data/agri
    parser.add_argument("--dataset_path",    type=str, default=None,
                        help="Override dataset_path from config (e.g. Google Drive path in Colab).")
    parser.add_argument("--output_dir",      type=str, default=None,
                        help="Override output_dir from config.")
    parser.add_argument("--checkpoint_dir",  type=str, default=None,
                        help="Override checkpoint_dir from config.")
    parser.add_argument("--gdrive_checkpoint_dir", type=str, default=None,
                        help="Google Drive path to mirror checkpoints for Colab resilience. "
                             "Example: /content/drive/MyDrive/SAR2EO/checkpoints")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.config):
        print(f"[Config] '{args.config}' not found — using built-in defaults.")
    else:
        print(f"[Config] Loading from: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

        DATASET_PATH      = cfg.get("dataset_path",     DATASET_PATH)
        IMAGE_SIZE        = cfg.get("image_size",        IMAGE_SIZE)
        BATCH_SIZE        = cfg.get("batch_size",        BATCH_SIZE)
        NUM_EPOCHS        = cfg.get("num_epochs",        NUM_EPOCHS)
        LEARNING_RATE     = cfg.get("learning_rate",     LEARNING_RATE)
        BETAS             = tuple(cfg.get("betas",       list(BETAS)))
        LAMBDA_L1         = cfg.get("lambda_l1",         LAMBDA_L1)
        AUGMENT           = cfg.get("augment",           AUGMENT)
        ABLATION_LOG      = cfg.get("ablation_log",      ABLATION_LOG)
        NGF               = cfg.get("ngf",               NGF)
        NDF               = cfg.get("ndf",               NDF)
        TRAIN_SPLIT       = cfg.get("train_split",       TRAIN_SPLIT)
        VAL_SPLIT         = cfg.get("val_split",         VAL_SPLIT)
        TEST_SPLIT        = cfg.get("test_split",        TEST_SPLIT)
        SEED              = cfg.get("seed",              SEED)
        CHECKPOINT_EVERY  = cfg.get("checkpoint_every",  CHECKPOINT_EVERY)
        SAMPLE_EVERY      = cfg.get("sample_every",      SAMPLE_EVERY)
        CHECKPOINT_DIR    = cfg.get("checkpoint_dir",    CHECKPOINT_DIR)
        OUTPUT_DIR        = cfg.get("output_dir",        OUTPUT_DIR)
        GDRIVE_CHECKPOINT_DIR = cfg.get("gdrive_checkpoint_dir", GDRIVE_CHECKPOINT_DIR)
        EMERGENCY_SAVE_EVERY_N = int(cfg.get("emergency_save_every_n_batches", EMERGENCY_SAVE_EVERY_N))

        # num_workers: "auto" or integer
        nw = cfg.get("num_workers", "auto")
        NUM_WORKERS = get_num_workers() if str(nw).lower() == "auto" else int(nw)

        # device: "auto", "cuda", or "cpu"
        dev = cfg.get("device", "auto")
        DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  if dev == "auto" else torch.device(dev))

    # ── CLI flags override anything set by config.yaml ────────────────────────
    if args.dataset_path   is not None:
        DATASET_PATH   = args.dataset_path
        print(f"[Config] dataset_path  overridden via CLI -> {DATASET_PATH}")
    if args.output_dir     is not None:
        OUTPUT_DIR     = args.output_dir
        print(f"[Config] output_dir    overridden via CLI -> {OUTPUT_DIR}")
    if args.checkpoint_dir is not None:
        CHECKPOINT_DIR = args.checkpoint_dir
        print(f"[Config] checkpoint_dir overridden via CLI -> {CHECKPOINT_DIR}")
    if args.gdrive_checkpoint_dir is not None:
        GDRIVE_CHECKPOINT_DIR = args.gdrive_checkpoint_dir
        print(f"[Config] gdrive_checkpoint_dir overridden via CLI -> {GDRIVE_CHECKPOINT_DIR}")

    # Recompute derived paths now that OUTPUT_DIR is finalised
    LOG_CSV           = os.path.join(OUTPUT_DIR, "training_log.csv")
    LOSS_CURVE        = os.path.join(OUTPUT_DIR, "loss_curve.png")
    SPLIT_CSV         = os.path.join(OUTPUT_DIR, "data_split.csv")
    TEST_METRICS_JSON = os.path.join(OUTPUT_DIR, "test_metrics.json")


load_config()


# ══════════════════════════════════════════════════════════════════════════════
# Data Split
# ══════════════════════════════════════════════════════════════════════════════

def build_splits(full_dataset: SAREODataset):
    """
    Create a reproducible 70/15/15 (or configured) train/val/test split.
    Saves indices to SPLIT_CSV so the exact split can be reconstructed later.

    Returns:
        train_indices, val_indices, test_indices  (lists of int)
    """
    n        = len(full_dataset)
    n_test   = int(n * TEST_SPLIT)
    n_val    = int(n * VAL_SPLIT)
    n_train  = n - n_val - n_test

    gen      = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=gen
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SPLIT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "filename"])
        for i in train_ds.indices:
            writer.writerow(["train", i, full_dataset.sar_files[i]])
        for i in val_ds.indices:
            writer.writerow(["val", i, full_dataset.sar_files[i]])
        for i in test_ds.indices:
            writer.writerow(["test", i, full_dataset.sar_files[i]])

    print(f"  [Split] Indices and filenames saved -> {SPLIT_CSV}")
    return train_ds.indices, val_ds.indices, test_ds.indices


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mirror_to_gdrive(src_path: str, label: str = "") -> None:
    """
    Copy *src_path* to GDRIVE_CHECKPOINT_DIR (if configured and reachable).
    Silently skips if the Drive is not mounted or the copy fails.
    """
    if not GDRIVE_CHECKPOINT_DIR:
        return
    try:
        os.makedirs(GDRIVE_CHECKPOINT_DIR, exist_ok=True)
        dst = os.path.join(GDRIVE_CHECKPOINT_DIR, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        tag = f" [{label}]" if label else ""
        print(f"  [GDrive{tag}] Mirrored -> {dst}")
    except Exception as exc:
        # Never crash training because Drive is slow / disconnected
        print(f"  [GDrive] Warning: could not mirror {src_path}: {exc}")


def save_checkpoint(gen, disc, opt_G, opt_D, epoch: int,
                    extra_label: str = "") -> str:
    """
    Save numbered + latest checkpoints and mirror both to Google Drive.

    Args:
        extra_label: Optional suffix added to the filename (e.g. "_emergency").

    Returns:
        Path of the latest-checkpoint file that was written.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    state = {
        "epoch":         epoch,
        "generator":     gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_G":   opt_G.state_dict(),
        "optimizer_D":   opt_D.state_dict(),
    }

    latest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    torch.save(state, latest_path)
    _mirror_to_gdrive(latest_path, "latest")

    if (epoch + 1) % CHECKPOINT_EVERY == 0 or extra_label:
        suffix = extra_label if extra_label else ""
        path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_epoch_{epoch + 1:03d}{suffix}.pth"
        )
        torch.save(state, path)
        print(f"  [Checkpoint] Saved {path}")
        _mirror_to_gdrive(path)

    return latest_path


def emergency_save(signum=None, frame=None) -> None:
    """
    Called on SIGTERM (Colab disconnect / timeout) or via atexit.
    Flushes the current model state to disk + Google Drive so training
    can resume from the latest completed batch.
    """
    if not _emergency_state:
        return   # train() hasn't started yet — nothing to save

    gen    = _emergency_state.get("gen")
    disc   = _emergency_state.get("disc")
    opt_G  = _emergency_state.get("opt_G")
    opt_D  = _emergency_state.get("opt_D")
    epoch  = _emergency_state.get("epoch", 0)

    if gen is None:
        return

    print("\n[Emergency] Disconnect/signal detected — saving checkpoint NOW...")
    try:
        save_checkpoint(gen, disc, opt_G, opt_D, epoch, extra_label="_emergency")
        print("[Emergency] Checkpoint saved successfully.")
    except Exception as exc:
        print(f"[Emergency] Save failed: {exc}")

    if signum is not None:
        # Re-raise so the process exits cleanly
        sys.exit(1)


def _register_emergency_handlers() -> None:
    """Register SIGTERM handler + atexit so emergency_save fires on disconnect."""
    atexit.register(emergency_save)
    try:
        signal.signal(signal.SIGTERM, emergency_save)
    except (OSError, ValueError):
        pass   # SIGTERM not available on Windows — harmless


def load_checkpoint(gen, disc, opt_G, opt_D) -> int:
    """
    Load the latest checkpoint if present.

    Search order:
      1. CHECKPOINT_DIR/checkpoint_latest.pth  (local — fast)
      2. GDRIVE_CHECKPOINT_DIR/checkpoint_latest.pth  (Drive fallback after
         a Colab runtime restart that wiped /content/)

    Returns:
        start epoch (0 if no checkpoint found).
    """
    # Prefer local checkpoint
    local_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")

    # Fall back to Google Drive if local is missing
    if not os.path.exists(local_path) and GDRIVE_CHECKPOINT_DIR:
        drive_path = os.path.join(GDRIVE_CHECKPOINT_DIR, "checkpoint_latest.pth")
        if os.path.exists(drive_path):
            print(f"[Resume] Local checkpoint missing — restoring from Drive: {drive_path}")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            shutil.copy2(drive_path, local_path)
            print(f"[Resume] Copied Drive checkpoint -> {local_path}")

    if not os.path.exists(local_path):
        print("[Resume] No checkpoint found — starting from scratch.")
        return 0

    print(f"[Resume] Loading checkpoint: {local_path}")
    state = torch.load(local_path, map_location=DEVICE, weights_only=False)
    gen.load_state_dict(state["generator"])
    disc.load_state_dict(state["discriminator"])
    opt_G.load_state_dict(state["optimizer_G"])
    opt_D.load_state_dict(state["optimizer_D"])
    start = state["epoch"] + 1
    print(f"[Resume] Resuming from epoch {start + 1} / {NUM_EPOCHS}")
    return start


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    """Create CSV with header row if it doesn't already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_g_total", "train_g_l1_only", "train_d_loss",
                "val_g_total",   "val_g_l1_only",   "val_d_loss",
            ])


def log_csv(epoch, tr_g, tr_g_l1, tr_d, v_g, v_g_l1, v_d):
    """Append one row to the training log CSV."""
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch + 1,
            f"{tr_g:.6f}", f"{tr_g_l1:.6f}", f"{tr_d:.6f}",
            f"{v_g:.6f}",  f"{v_g_l1:.6f}",  f"{v_d:.6f}",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Sample Image
# ══════════════════════════════════════════════════════════════════════════════

def save_sample(gen, sar_batch: torch.Tensor, gt_batch: torch.Tensor, epoch: int):
    """
    Generate one sample and save a SAR | Pred EO | GT EO triplet grid.
    Generator is temporarily set to eval mode.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gen.eval()
    with torch.no_grad():
        sar  = sar_batch[:1].to(DEVICE)
        gt   = gt_batch[:1].to(DEVICE)
        pred = gen(sar)
        out  = os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch + 1:03d}.png")
        save_triplet_grid(sar.cpu(), pred.cpu(), gt.cpu(), out)
    gen.train()
    print(f"  [Sample] Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Loss Curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curve(history: list):
    """
    Save a 3-panel loss curve PNG:
      Panel 1 — Generator: GAN+L1 vs L1-only (ablation comparison)
      Panel 2 — Discriminator
      Panel 3 — Loss decomposition: GAN component vs L1 component
    """
    epochs    = [h["epoch"]      for h in history]
    tr_g      = [h["train_g"]    for h in history]
    tr_g_l1   = [h["train_g_l1"] for h in history]
    tr_d      = [h["train_d"]    for h in history]
    v_g       = [h["val_g"]      for h in history]
    v_g_l1    = [h["val_g_l1"]   for h in history]
    v_d       = [h["val_d"]      for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Generator ablation
    axes[0].plot(epochs, tr_g,    color="royalblue",  label="Train G (GAN+L1)")
    axes[0].plot(epochs, v_g,     color="royalblue",  linestyle="--", label="Val G (GAN+L1)")
    axes[0].plot(epochs, tr_g_l1, color="darkorange", label="Train G (L1 only)")
    axes[0].plot(epochs, v_g_l1,  color="darkorange", linestyle="--", label="Val G (L1 only)")
    axes[0].set_title("Generator Loss (Ablation: GAN+L1 vs L1-only)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: Discriminator
    axes[1].plot(epochs, tr_d, color="tomato", label="Train D")
    axes[1].plot(epochs, v_d,  color="tomato", linestyle="--", label="Val D")
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Panel 3: Decompose GAN component from total generator loss
    tr_g_gan = [g - l for g, l in zip(tr_g, tr_g_l1)]
    axes[2].plot(epochs, tr_g_gan, color="purple", label="GAN component")
    axes[2].plot(epochs, tr_g_l1,  color="green",  label="L1 component (λ×L1)")
    axes[2].set_title("Generator Loss Decomposition")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Pix2Pix SAR→EO  |  Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(LOSS_CURVE, dpi=150)
    plt.close()
    print(f"[Plot] Loss curve -> {LOSS_CURVE}")


# ══════════════════════════════════════════════════════════════════════════════
# Test Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_test_set(gen, test_loader: DataLoader):
    """
    Run the generator over the full test split.
    Saves:
      - outputs/test_metrics.json  (L1 loss)
      - outputs/test_example_01..05.png  (up to 5 qualitative triplets)

    Note: For LPIPS/FID/SSIM/PSNR, run eval.py with the generated images.
    """
    gen.eval()
    l1_fn    = nn.L1Loss()
    total_l1 = 0.0
    n_saved  = 0

    with torch.no_grad():
        for sar_batch, real_eo in test_loader:
            sar_batch = sar_batch.to(DEVICE)
            real_eo   = real_eo.to(DEVICE)
            fake_eo   = gen(sar_batch)
            total_l1 += l1_fn(fake_eo, real_eo).item()

            # Save up to 5 triplet examples
            for j in range(sar_batch.size(0)):
                if n_saved >= 5:
                    break
                n_saved += 1
                path = os.path.join(OUTPUT_DIR, f"test_example_{n_saved:02d}.png")
                save_triplet_grid(
                    sar_batch[j:j+1].cpu(),
                    fake_eo[j:j+1].cpu(),
                    real_eo[j:j+1].cpu(),
                    path,
                )

    avg_l1  = total_l1 / max(len(test_loader), 1)
    metrics = {
        "test_g_l1_loss": round(avg_l1, 6),
        "note": (
            "Run eval.py --pred_dir <dir> --gt_dir <dir> "
            "for LPIPS / FID / SSIM / PSNR on generated images."
        ),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TEST_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Test] L1 Loss  : {avg_l1:.6f}")
    print(f"[Test] Metrics  -> {TEST_METRICS_JSON}")
    print(f"[Test] Triplets -> {OUTPUT_DIR}/test_example_*.png  ({n_saved} saved)")


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train():
    set_seed(SEED)
    _register_emergency_handlers()

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Load without augmentation first (used for val/test subsets and split computation)
    full_dataset = SAREODataset(DATASET_PATH, image_size=IMAGE_SIZE, augment=False)

    train_idx, val_idx, test_idx = build_splits(full_dataset)

    # For the training subset, load a second instance WITH augmentation enabled
    aug_dataset  = SAREODataset(DATASET_PATH, image_size=IMAGE_SIZE, augment=AUGMENT)
    train_subset = Subset(aug_dataset,  train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Print run summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  Pix2Pix SAR-to-EO  |  Training")
    print("=" * 65)
    print(f"  Dataset      : {DATASET_PATH}  ({len(full_dataset):,} total pairs)")
    print(f"  Train / Val / Test : {len(train_subset):,} / {len(val_subset):,} / {len(test_subset):,}")
    print(f"  Device       : {DEVICE}" +
          (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  Epochs       : {NUM_EPOCHS}   Batch: {BATCH_SIZE}   LR: {LEARNING_RATE}")
    print(f"  lambda_L1    : {LAMBDA_L1}   Augment: {AUGMENT}   Ablation log: {ABLATION_LOG}")
    print(f"  ngf / ndf    : {NGF} / {NDF}")
    print()

    # ── Models ────────────────────────────────────────────────────────────────
    gen  = Generator(in_channels=1, out_channels=3, ngf=NGF).to(DEVICE)
    disc = Discriminator(in_channels=4, ndf=NDF).to(DEVICE)

    # ── Loss functions ────────────────────────────────────────────────────────
    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss  = nn.L1Loss()

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(gen.parameters(),  lr=LEARNING_RATE, betas=BETAS)
    opt_D = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(gen, disc, opt_G, opt_D)
    init_csv()

    # Populate the emergency-save state now that models exist
    _emergency_state.update({
        "gen":   gen,
        "disc":  disc,
        "opt_G": opt_G,
        "opt_D": opt_D,
        "epoch": start_epoch,
    })

    if GDRIVE_CHECKPOINT_DIR:
        print(f"[Colab] Google Drive mirror: {GDRIVE_CHECKPOINT_DIR}")
    if EMERGENCY_SAVE_EVERY_N > 0:
        print(f"[Colab] In-epoch emergency save every {EMERGENCY_SAVE_EVERY_N} batches")

    history = []

    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Keep emergency state current so a mid-epoch signal saves this epoch
        _emergency_state["epoch"] = epoch

        # ── TRAIN ─────────────────────────────────────────────────────────────
        gen.train(); disc.train()
        tr_g_total = tr_g_l1 = tr_d = 0.0
        first_sar = first_gt = None

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:>3}/{NUM_EPOCHS} [Train]",
                    leave=False, ncols=110)

        for batch_idx, (sar_batch, real_eo) in enumerate(pbar):
            if first_sar is None:
                first_sar = sar_batch.clone()
                first_gt  = real_eo.clone()

            sar_batch = sar_batch.to(DEVICE)
            real_eo   = real_eo.to(DEVICE)
            fake_eo   = gen(sar_batch)

            # Discriminator step
            real_pred   = disc(sar_batch, real_eo)
            fake_pred_d = disc(sar_batch, fake_eo.detach())
            d_loss = (
                gan_loss(real_pred,   torch.ones_like(real_pred))  +
                gan_loss(fake_pred_d, torch.zeros_like(fake_pred_d))
            ) / 2
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Generator step
            fake_pred = disc(sar_batch, fake_eo)
            g_l1      = l1_loss(fake_eo, real_eo)
            g_gan     = gan_loss(fake_pred, torch.ones_like(fake_pred))
            g_total   = g_gan + LAMBDA_L1 * g_l1
            opt_G.zero_grad(); g_total.backward(); opt_G.step()

            tr_d      += d_loss.item()
            tr_g_total += g_total.item()
            tr_g_l1   += (LAMBDA_L1 * g_l1).item()   # L1-only for ablation log

            pbar.set_postfix(
                d=f"{d_loss.item():.4f}",
                g=f"{g_total.item():.4f}",
            )

            # ── In-epoch emergency save ────────────────────────────────────────
            # Fires every EMERGENCY_SAVE_EVERY_N batches (0 = disabled).
            # Saves a "_inepoch" checkpoint so a mid-epoch crash can be recovered.
            if (EMERGENCY_SAVE_EVERY_N > 0
                    and (batch_idx + 1) % EMERGENCY_SAVE_EVERY_N == 0):
                save_checkpoint(gen, disc, opt_G, opt_D, epoch,
                                extra_label="_inepoch")
                tqdm.write(
                    f"  [InEpoch] Emergency save at epoch {epoch+1}, "
                    f"batch {batch_idx+1}/{len(train_loader)}"
                )

        n = len(train_loader)
        avg_tr_g, avg_tr_l1, avg_tr_d = tr_g_total/n, tr_g_l1/n, tr_d/n

        # ── VALIDATE ──────────────────────────────────────────────────────────
        gen.eval(); disc.eval()
        v_g_total = v_g_l1 = v_d = 0.0

        with torch.no_grad():
            for sar_batch, real_eo in val_loader:
                sar_batch = sar_batch.to(DEVICE)
                real_eo   = real_eo.to(DEVICE)
                fake_eo   = gen(sar_batch)

                real_pred = disc(sar_batch, real_eo)
                fake_pred = disc(sar_batch, fake_eo)
                d_val = (
                    gan_loss(real_pred, torch.ones_like(real_pred)) +
                    gan_loss(fake_pred, torch.zeros_like(fake_pred))
                ) / 2
                g_l1_v  = l1_loss(fake_eo, real_eo)
                g_gan_v = gan_loss(fake_pred, torch.ones_like(fake_pred))
                g_tot_v = g_gan_v + LAMBDA_L1 * g_l1_v

                v_d      += d_val.item()
                v_g_total += g_tot_v.item()
                v_g_l1   += (LAMBDA_L1 * g_l1_v).item()

        m = len(val_loader)
        avg_v_g, avg_v_l1, avg_v_d = v_g_total/m, v_g_l1/m, v_d/m

        # ── Epoch summary ──────────────────────────────────────────────────────
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS}  "
            f"| Train — G(total): {avg_tr_g:.4f}  G(L1): {avg_tr_l1:.4f}  D: {avg_tr_d:.4f}"
            f"  | Val — G(total): {avg_v_g:.4f}  G(L1): {avg_v_l1:.4f}  D: {avg_v_d:.4f}"
        )

        history.append({
            "epoch":      epoch + 1,
            "train_g":    avg_tr_g,  "train_g_l1": avg_tr_l1, "train_d": avg_tr_d,
            "val_g":      avg_v_g,   "val_g_l1":   avg_v_l1,  "val_d":   avg_v_d,
        })

        log_csv(epoch, avg_tr_g, avg_tr_l1, avg_tr_d, avg_v_g, avg_v_l1, avg_v_d)

        if (epoch + 1) % SAMPLE_EVERY == 0 and first_sar is not None:
            save_sample(gen, first_sar, first_gt, epoch)

        save_checkpoint(gen, disc, opt_G, opt_D, epoch)

    # ── Post-training ──────────────────────────────────────────────────────────
    if history:
        plot_loss_curve(history)

    print("\n" + "=" * 65)
    print("  Test-set evaluation")
    print("=" * 65)
    evaluate_test_set(gen, test_loader)

    print("\n[PASS] Training complete.")
    print(f"  Checkpoints  -> {CHECKPOINT_DIR}/")
    print(f"  CSV log      -> {LOG_CSV}")
    print(f"  Loss curve   -> {LOSS_CURVE}")
    print(f"  Split CSV    -> {SPLIT_CSV}")
    print(f"  Test metrics -> {TEST_METRICS_JSON}")
    print(f"  Samples      -> {OUTPUT_DIR}/sample_epoch_*.png")
    print(f"  Test triplets -> {OUTPUT_DIR}/test_example_*.png")


if __name__ == "__main__":
    train()
