

import os
import argparse
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from generator import Generator
from utils import denormalize


# Argument Parser

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAR-to-EO image translation — inference"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory of input SAR PNG patches (greyscale, 256×256).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where generated RGB EO images will be saved.",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to generator checkpoint (.pth file).",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Generator input/output size (default 256). Input is auto-resized.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto).",
    )
    return parser.parse_args()


# Inference

def run_inference(input_dir: str, output_dir: str, weights_path: str,
                  image_size: int = 256, device_str: str = "auto"):
    """
    Translate all SAR images in `input_dir` to EO images in `output_dir`.

    Handles:
      - Any input resolution (auto-resizes to image_size, outputs original size)
      - Any grayscale image format: .png, .jpg, .tif, .tiff
      - Both bare state_dicts and full checkpoint dicts (from train.py)
      - Proper [-1,1] → [0,1] de-normalisation of generator output

    Args:
        input_dir:    Path to directory containing SAR images.
        output_dir:   Path to save generated EO images.
        weights_path: Path to .pth checkpoint file.
        image_size:   Generator internal resolution (default 256).
        device_str:   "auto", "cuda", or "cpu".
    """
    # Validate inputs
    if not os.path.isdir(input_dir):
        print(f"[Error] Input directory not found: '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(weights_path):
        print(f"[Error] Weights file not found: '{weights_path}'\n"
              f"  Download the checkpoint from your training run and provide the path.",
              file=sys.stderr)
        sys.exit(1)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Device       : {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})", end="")
    print()

    # Load Generator
    gen = Generator(in_channels=1, out_channels=3).to(device)

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[Error] Failed to load weights: {e}", file=sys.stderr)
        sys.exit(1)

    # Support both bare state_dict and full checkpoint dict (from train.py)
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        gen.load_state_dict(checkpoint["generator"])
        print(f"Weights      : {weights_path}  (epoch {checkpoint.get('epoch', '?')})")
    else:
        gen.load_state_dict(checkpoint)
        print(f"Weights      : {weights_path}")

    gen.eval()
    print("Generator loaded successfully.\n")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect input files
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    input_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not input_files:
        print(f"[Error] No image files found in: '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Input dir    : {input_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Files found  : {len(input_files)}\n")

    # Transforms
    resize_in  = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )

    # Run inference
    n_ok = 0
    with torch.no_grad():
        for i, filename in enumerate(input_files, 1):
            sar_path = os.path.join(input_dir, filename)

            try:
                sar_img  = Image.open(sar_path).convert("L")
            except Exception as e:
                print(f"  [{i:>4}/{len(input_files)}] SKIP  {filename}  — {e}")
                continue

            orig_w, orig_h = sar_img.size  # keep original size for resizing back

            # Resize to generator input size
            sar_resized = resize_in(sar_img)

            # [0,255] → [0,1] tensor
            sar_tensor = transforms.ToTensor()(sar_resized).unsqueeze(0).to(device)
            # shape: [1, 1, image_size, image_size]

            # Generate EO image — output is in [-1, 1]
            generated = gen(sar_tensor)              # [1, 3, H, W]

            # De-normalise: [-1,1] → [0,1]  (avoids colour distortion)
            generated = denormalize(generated)

            # Resize output back to original input dimensions if needed
            if (orig_w, orig_h) != (image_size, image_size):
                resize_out = transforms.Resize(
                    (orig_h, orig_w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                generated = resize_out(generated)

            # Save with the exact same filename as the input
            out_path = os.path.join(output_dir, filename)

            # Ensure output is saved as PNG regardless of input extension
            out_basename = os.path.splitext(filename)[0] + ".png"
            out_path = os.path.join(output_dir, out_basename)

            save_image(generated, out_path, normalize=False)

            print(f"  [{i:>4}/{len(input_files)}] {filename}  ->  {out_path}")
            n_ok += 1

    print(f"\n[PASS] Done.  {n_ok}/{len(input_files)} images saved to: {output_dir}/")


# Entry Point

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        weights_path = args.weights,
        image_size  = args.image_size,
        device_str  = args.device,
    )