"""
eval.py — Evaluation script for SAR-to-EO image translation.

Computes all four metrics required by the GalaxEye assessment:
    Primary   (ranking): LPIPS ↓, FID ↓
    Secondary (quality): SSIM ↑, PSNR ↑

Usage:
    python eval.py --pred_dir <generated_eo_dir> --gt_dir <ground_truth_dir>

Optional flags:
    --output_csv   Path for per-image CSV results (default: outputs/eval_results.csv)
    --output_json  Path for aggregate JSON results (default: outputs/eval_results.json)
    --split_csv    outputs/data_split.csv — if provided, evaluates only 'test' split files
    --device       cuda | cpu | auto

Dependencies (all in requirements.txt):
    lpips, pytorch-fid, scikit-image
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Graceful optional imports ──────────────────────────────────────────────────

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[Warning] lpips not installed — run: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Warning] scikit-image not installed — run: pip install scikit-image")

# Patch scipy.linalg.sqrtm for pytorch-fid compatibility with newer SciPy
try:
    import scipy.linalg as _la
    _orig_sqrtm = _la.sqrtm
    def _patched_sqrtm(A, *args, **kwargs):
        res = _orig_sqrtm(A)
        # Newer SciPy returns ndarray instead of (ndarray, errest) tuple
        if isinstance(res, tuple):
            return res
        return (res, 0.0)
    _la.sqrtm = _patched_sqrtm
except Exception:
    pass

try:
    from pytorch_fid import fid_score as fid_module
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("[Warning] pytorch-fid not installed — run: pip install pytorch-fid")


# ── Argument Parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAR-to-EO translation: LPIPS, FID, SSIM, PSNR"
    )
    parser.add_argument("--pred_dir",    type=str, required=True,
                        help="Directory of generated (predicted) EO images.")
    parser.add_argument("--gt_dir",      type=str, required=True,
                        help="Directory of ground-truth EO images.")
    parser.add_argument("--output_csv",  type=str,
                        default=os.path.join("outputs", "eval_results.csv"),
                        help="Where to save per-image metric results (CSV).")
    parser.add_argument("--output_json", type=str,
                        default=os.path.join("outputs", "eval_results.json"),
                        help="Where to save aggregate metric results (JSON).")
    parser.add_argument("--split_csv",   type=str, default=None,
                        help="outputs/data_split.csv — if given, only evaluate 'test' files.")
    parser.add_argument("--device",      type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'.")
    return parser.parse_args()


# ── Image Utilities ────────────────────────────────────────────────────────────

def load_image_np(path: str) -> np.ndarray:
    """Load image as uint8 NumPy array [H, W, 3]."""
    return np.array(Image.open(path).convert("RGB"))


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load image as normalised float tensor [1, 3, H, W] in [-1, 1] for LPIPS."""
    img = Image.open(path).convert("RGB")
    t   = transforms.ToTensor()(img)          # [3, H, W]  in [0, 1]
    t   = (t * 2.0) - 1.0                     # → [-1, 1]  (LPIPS input convention)
    return t.unsqueeze(0).to(device)


# ── File Matching ──────────────────────────────────────────────────────────────

def match_files(pred_dir: str, gt_dir: str, split_csv: str = None):
    """
    Return sorted list of (pred_path, gt_path) pairs whose filenames match.

    If split_csv is provided, only includes files whose basenames correspond to
    entries in the 'test' split (for no-leakage evaluation).
    """
    valid_exts = {".png", ".jpg", ".jpeg"}

    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if Path(f).suffix.lower() in valid_exts
    ])

    gt_lookup = {
        f for f in os.listdir(gt_dir)
        if Path(f).suffix.lower() in valid_exts
    }

    # Optional: filter to test-split filenames only
    test_filter = None
    if split_csv and os.path.exists(split_csv):
        test_filter = set()
        with open(split_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split") == "test":
                    fn = row.get("filename")
                    if fn:
                        test_filter.add(fn.lower())
                        base_stem = os.path.splitext(fn)[0].lower()
                        test_filter.add(base_stem)
                        test_filter.add(base_stem.replace("s1", "s2"))
                        test_filter.add(base_stem.replace("_s1_", "_s2_"))
        print(f"[Info] --split_csv provided. Loaded {len(test_filter)} test-set filters.")

    pairs = []
    for f in pred_files:
        if test_filter is not None:
            f_lower = f.lower()
            f_stem = os.path.splitext(f)[0].lower()
            f_stem_s1 = f_stem.replace("s2", "s1").replace("_s2_", "_s1_")
            if f_lower not in test_filter and f_stem not in test_filter and f_stem_s1 not in test_filter:
                continue

        if f in gt_lookup:
            pairs.append((os.path.join(pred_dir, f), os.path.join(gt_dir, f)))
        else:
            # Try _s1_ → _s2_ mapping (Kaggle dataset naming)
            mapped = f.replace("_s1_", "_s2_")
            if mapped in gt_lookup:
                pairs.append((os.path.join(pred_dir, f), os.path.join(gt_dir, mapped)))

    if not pairs:
        raise FileNotFoundError(
            f"No matching image pairs found between:\n"
            f"  pred: {pred_dir}\n"
            f"  gt:   {gt_dir}\n"
            f"Ensure filenames match, or follow the _s1_/_s2_ Kaggle naming scheme."
        )

    return pairs


# ── Metric Computation ─────────────────────────────────────────────────────────

def compute_metrics(pred_dir: str, gt_dir: str, device: torch.device,
                    output_csv: str, output_json: str, split_csv: str = None):
    """
    Compute SSIM, PSNR, LPIPS per image and FID over the full set.
    Saves results to CSV and JSON.
    """
    pairs = match_files(pred_dir, gt_dir, split_csv)
    print(f"Matched pairs : {len(pairs)}")
    print(f"Device        : {device}\n")

    # Initialise LPIPS model
    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()

    ssim_scores  = []
    psnr_scores  = []
    lpips_scores = []
    rows         = []

    print(f"{'File':<42} {'SSIM':>8} {'PSNR':>8} {'LPIPS':>8}")
    print("-" * 70)

    for pred_path, gt_path in pairs:
        filename = os.path.basename(pred_path)
        pred_np  = load_image_np(pred_path)
        gt_np    = load_image_np(gt_path)

        # ── SSIM ──────────────────────────────────────────────────────────────
        ssim_val = float("nan")
        if SKIMAGE_AVAILABLE:
            ssim_val = ssim_fn(
                gt_np, pred_np,
                channel_axis=2,     # use channel_axis; multichannel is deprecated
                data_range=255,
            )
            ssim_scores.append(ssim_val)

        # ── PSNR ──────────────────────────────────────────────────────────────
        psnr_val = float("nan")
        if SKIMAGE_AVAILABLE:
            psnr_val = psnr_fn(gt_np, pred_np, data_range=255)
            psnr_scores.append(psnr_val)

        # ── LPIPS ─────────────────────────────────────────────────────────────
        lpips_val = float("nan")
        if LPIPS_AVAILABLE and lpips_model is not None:
            pred_t = load_image_tensor(pred_path, device)
            gt_t   = load_image_tensor(gt_path,   device)
            with torch.no_grad():
                lpips_val = lpips_model(pred_t, gt_t).item()
            lpips_scores.append(lpips_val)

        print(f"{filename:<42} {ssim_val:>8.4f} {psnr_val:>8.2f} {lpips_val:>8.4f}")
        rows.append({
            "filename": filename,
            "ssim":     ssim_val,
            "psnr":     psnr_val,
            "lpips":    lpips_val,
        })

    # ── FID (whole-directory) ──────────────────────────────────────────────────
    fid_val = float("nan")
    if FID_AVAILABLE:
        print("\nComputing FID (may take a moment) ...")
        try:
            fid_val = fid_module.calculate_fid_given_paths(
                [pred_dir, gt_dir],
                batch_size=8,
                device=str(device),
                dims=2048,
            )
        except Exception as e:
            print(f"[Warning] FID computation failed: {e}")

    # ── Aggregates ─────────────────────────────────────────────────────────────
    mean_ssim  = float(np.mean(ssim_scores))  if ssim_scores  else float("nan")
    mean_psnr  = float(np.mean(psnr_scores))  if psnr_scores  else float("nan")
    mean_lpips = float(np.mean(lpips_scores)) if lpips_scores else float("nan")

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Images evaluated : {len(pairs)}")
    print(f"  SSIM  (+)        : {mean_ssim:.4f}")
    print(f"  PSNR  (+) (dB)   : {mean_psnr:.2f}")
    print(f"  LPIPS (-)        : {mean_lpips:.4f}")
    print(f"  FID   (-)        : {fid_val:.2f}" if not np.isnan(fid_val) else "  FID   (-)        : N/A")
    print("=" * 60)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ssim", "psnr", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"filename": "MEAN",           "ssim": f"{mean_ssim:.6f}",
                         "psnr": f"{mean_psnr:.6f}",   "lpips": f"{mean_lpips:.6f}"})
        writer.writerow({"filename": "FID (full set)", "ssim": "",
                         "psnr": "",                   "lpips": f"{fid_val:.4f}"})
    print(f"\n  CSV  -> {output_csv}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    aggregate = {
        "n_images": len(pairs),
        "ssim":     round(mean_ssim,  4) if not np.isnan(mean_ssim)  else None,
        "psnr":     round(mean_psnr,  2) if not np.isnan(mean_psnr)  else None,
        "lpips":    round(mean_lpips, 4) if not np.isnan(mean_lpips) else None,
        "fid":      round(fid_val,    2) if not np.isnan(fid_val)    else None,
    }
    with open(output_json, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  JSON -> {output_json}")

    return aggregate


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    compute_metrics(
        pred_dir    = args.pred_dir,
        gt_dir      = args.gt_dir,
        device      = device,
        output_csv  = args.output_csv,
        output_json = args.output_json,
        split_csv   = args.split_csv,
    )
