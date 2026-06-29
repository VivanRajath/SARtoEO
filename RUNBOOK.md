# SAR2EO — Operational Runbook

> **Pix2Pix SAR-to-EO Image Translation | GalaxEye Space Technical Assignment**
>
> Complete guide to running every part of this project. If something doesn't work,
> check the [Troubleshooting](#10-troubleshooting) section first.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [Running Unit Tests](#3-running-unit-tests)
4. [Training](#4-training)
5. [Inference (Standard — Directory)](#5-inference-standard--directory)
6. [Custom Single-Image Inference](#6-custom-single-image-inference)
7. [Running Evaluation](#7-running-evaluation)
8. [End-to-End Pipeline (All Steps)](#8-end-to-end-pipeline-all-steps)
9. [Google Colab Workflow](#9-google-colab-workflow)
10. [Troubleshooting](#10-troubleshooting)
11. [Config Reference](#11-config-reference)
12. [Output File Reference](#12-output-file-reference)

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 or higher (tested on 3.12)
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/VivanRajath/SARtoEO.git
cd SAR2EO
```

### Step 2 — Create a virtual environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows CMD
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

> **PowerShell execution policy error?** Run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: torch, torchvision, numpy, Pillow, matplotlib, tqdm, PyYAML,
scikit-image, lpips, pytorch-fid, opencv-python, reportlab.

### Step 4 — Download model weights

The checkpoint (~654 MB) is not included in this repo. Run this **one command** to download it automatically from HuggingFace:

```bash
python download_weights.py
```

This script will:
- Download `checkpoint_latest.pth` directly from [VivanRajath/SAR2EO on HuggingFace](https://huggingface.co/VivanRajath/SAR2EO)
- Save it to `checkpoints/checkpoint_latest.pth`
- Show a live progress bar
- Skip re-downloading if the file already exists

**Optional — faster/resumable downloads:**
```bash
pip install huggingface_hub
python download_weights.py
```
Without `huggingface_hub`, the script falls back to `urllib` automatically — no extra install required.

### Step 5 — Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import lpips; print('LPIPS: OK')"
python -c "import pytorch_fid; print('pytorch-fid: OK')"
```

---

## 2. Project Structure

```
SAR2EO/
|
|-- data/
|   +-- agri/
|       |-- s1/          <- SAR input images (Sentinel-1 VV, grayscale PNG, 256x256)
|       +-- s2/          <- EO target images (Sentinel-2 RGB PNG, 256x256)
|
|-- checkpoints/         <- Model checkpoints (.pth files)
|   +-- checkpoint_latest.pth       <- ALWAYS up-to-date (use for inference)
|   +-- checkpoint_epoch_NNN.pth    <- Per-epoch snapshots
|
|-- outputs/             <- All generated artifacts
|   |-- data_split.csv       <- Train/val/test split (reproducibility)
|   |-- training_log.csv     <- Per-epoch G and D losses
|   |-- loss_curve.png       <- 3-panel loss curve plot
|   |-- test_pred/           <- Generated EO predictions on test split
|   |-- eval_results.csv     <- Per-image SSIM / PSNR / LPIPS scores
|   |-- eval_results.json    <- Aggregate metrics (SSIM, PSNR, LPIPS, FID)
|   |-- test_metrics.json    <- L1 loss on test set (from train.py)
|   +-- test_example_0N.png  <- Qualitative SAR | Pred | GT triplets
|
|-- sample/              <- Small set of SAR images for quick testing
|
|-- train.py             <- Full training script with ablation logging
|-- infer.py             <- CLI inference (GalaxEye spec-compliant)
|-- eval.py              <- Evaluation: SSIM, PSNR, LPIPS, FID
|-- dataset.py           <- Paired SAR/EO dataset loader
|-- generator.py         <- 8-level Pix2Pix U-Net generator (~54M params)
|-- discriminator.py     <- 70x70 PatchGAN discriminator (~2.8M params)
|-- utils.py             <- Denormalize, triplet grid, seed, etc.
|-- config.yaml          <- All hyperparameters and paths
|-- requirements.txt     <- Pinned Python dependencies
|
|-- test_dataset.py      <- Unit test: dataset loader shapes and ranges
|-- test_generator.py    <- Unit test: generator input/output shapes
|-- test_disciminator.py <- Unit test: discriminator output shape
|
|-- architecture.md      <- Deep-dive: pipeline, filters, eval methodology
|-- RUNBOOK.md           <- This file
+-- README.md            <- Project overview (GitHub landing page)
```

---

## 3. Running Unit Tests

Run these before training or inference to verify your environment is correct.

### Test 1 — Dataset Loader

```bash
python test_dataset.py
```

**Expected output:**
```
Dataset size : 4002
SAR shape    : torch.Size([1, 256, 256])   range [0.000, 1.000]
EO  shape    : torch.Size([3, 256, 256])   range [-1.000, 1.000]
[PASS] Dataset test passed.
```

**If it fails:**
- `FileNotFoundError: data/agri/s1`: Your dataset is not in the expected location.
  Check `config.yaml → dataset_path`. Data must be in `data/agri/s1/` and `data/agri/s2/`.
- `Dataset size: 0`: No images were found. Verify filenames end in `.png`, `.jpg`, `.tif`, or `.tiff`.
- Wrong range: The dataset normalization is incorrect. SAR should be [0,1], EO should be [-1,1].

### Test 2 — Generator Architecture

```bash
python test_generator.py
```

**Expected output:**
```
Input  shape: torch.Size([2, 1, 256, 256])
Output shape: torch.Size([2, 3, 256, 256])
[PASS] Generator test passed.
```

**If it fails:**
- `RuntimeError: CUDA out of memory`: GPU OOM on test. Run with `--device cpu` or check GPU memory.
- `ModuleNotFoundError: torch`: PyTorch not installed. Run `pip install torch`.
- Shape mismatch: Your `generator.py` may have been modified incorrectly. Check the encoder/decoder channel counts.

### Test 3 — Discriminator Architecture

```bash
python test_disciminator.py
```

**Expected output:**
```
SAR shape   : torch.Size([2, 1, 256, 256])
EO shape    : torch.Size([2, 3, 256, 256])
Output shape: torch.Size([2, 1, 30, 30])
[PASS] Discriminator test passed.
```

**If output shape is not [2, 1, 30, 30]:** The PatchGAN receptive field calculation is off.
The 30x30 patch grid is correct for 256x256 input through 4 blocks. Do not modify the
stride/padding values in `discriminator.py`.

---

## 4. Training

### Standard Training

```bash
python train.py --config config.yaml
```

Training will print progress like:
```
[Epoch  1/50] train G=39.04 L1=35.26 D=0.217  val G=37.48 L1=36.58 D=0.833
[Epoch  2/50] train G=38.60 L1=35.42 D=0.269  val G=36.44 L1=34.69 D=1.060
...
```

### Resume from Checkpoint

Training automatically resumes from `checkpoints/checkpoint_latest.pth` if it exists:
```
[Resume] Loading checkpoint: checkpoints/checkpoint_latest.pth
[Resume] Resuming from epoch 16 / 50
```
No manual intervention needed. Just run `python train.py --config config.yaml` again.

### Override Paths at Runtime (No Config Edit Needed)

```bash
python train.py --config config.yaml \
    --dataset_path /path/to/your/data/agri \
    --checkpoint_dir /path/to/save/checkpoints \
    --output_dir /path/to/save/outputs
```

This is especially useful for Colab where paths differ from local setup.

### Key Config Options

Edit `config.yaml` to change training behavior:

```yaml
num_epochs: 50          # Increase to 100-200 for better quality
batch_size: 2           # Increase if you have more VRAM (4 on 16GB GPU)
learning_rate: 0.0002   # Adam LR — do not change unless training is unstable
lambda_l1: 100          # Higher = more faithful to structure, less creative
augment: true           # Random horizontal + vertical flips on train set only
ablation_log: true      # Log L1-only loss alongside GAN+L1 (ablation study)
sample_every: 5         # Save qualitative triplet image every N epochs
checkpoint_every: 1     # Save numbered checkpoint every N epochs (disk space!)
```

> **Note**: `num_epochs: 50` was used for this submission. The model is still learning at
> epoch 50 — extending to 100-200 epochs will improve SSIM, LPIPS, and FID significantly.
> The loss was not plateaued. See [architecture.md](architecture.md) for convergence analysis.

### Training Outputs Generated

| File | Description |
|------|-------------|
| `checkpoints/checkpoint_latest.pth` | Full checkpoint for resuming or inference |
| `checkpoints/checkpoint_epoch_NNN.pth` | Per-epoch snapshot |
| `outputs/training_log.csv` | Per-epoch G_total, G_L1, D loss for train+val |
| `outputs/loss_curve.png` | 3-panel loss curve (ablation + discriminator) |
| `outputs/data_split.csv` | Train/val/test indices for reproducibility |
| `outputs/sample_epoch_NNN.png` | SAR | Generated EO | GT triplet every N epochs |
| `outputs/test_example_01..05.png` | 5 qualitative triplets from test split |
| `outputs/test_metrics.json` | L1 loss on test set |

### Monitoring Training Health

Watch the training output for these signals:

| What you see | What it means |
|---|---|
| D loss stabilizes at ~0.3 | Healthy GAN — G and D are balanced |
| D loss < 0.05 | D too strong; consider lower D LR |
| D loss > 0.85 | G fooling D easily; check data loading |
| G loss oscillates but trends down | Normal GAN adversarial behavior |
| G loss monotonically increases | LR too high; reduce to 0.0001 |
| L1 loss not decreasing after 5 epochs | Data loading issue or LR too low |

---

## 5. Inference (Standard — Directory)

Translate a directory of SAR images to EO images using a trained checkpoint.

### Command (GalaxEye Assessment Spec)

```bash
python infer.py \
    --input_dir  <path_to_sar_directory> \
    --output_dir <path_to_output_directory> \
    --weights    checkpoints/checkpoint_latest.pth
```

### Example — Using the Included Samples

```bash
python infer.py \
    --input_dir  sample/ \
    --output_dir outputs/generated_eo/ \
    --weights    checkpoints/checkpoint_latest.pth
```

### All Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input_dir` | Yes | — | Directory of SAR PNG images to translate |
| `--output_dir` | Yes | — | Directory to save generated EO RGB images |
| `--weights` | Yes | — | Path to .pth checkpoint file |
| `--image_size` | No | 256 | Internal generator resolution |
| `--device` | No | auto | `cuda`, `cpu`, or `auto` (auto-detects GPU) |

### Input Requirements

- **Format**: .png, .jpg, .tif, .tiff
- **Channels**: Grayscale (single channel)
- **Bit depth**: 8-bit (values 0-255)
- **Preprocessing**: dB-scaled, normalized to [0, 255] (Sentinel-1 convention)
- **Resolution**: Any size accepted — auto-resized to 256x256 internally
- **Naming**: Output files get the same filename as input (with .png extension)

### Output Format

- **Format**: RGB PNG
- **Size**: Same as input (auto-resized back if input was not 256x256)
- **Range**: [0, 255] uint8
- **Naming**: Identical to input filename

### Debug — Verify a Single Output

After inference, open one output with any image viewer. Expected appearance:
- Color image (RGB, not grayscale)
- Green/brown agricultural fields (for agricultural dataset)
- Visible field boundaries and patterns

If the output is all black or all white, the checkpoint may be corrupted or the
input normalization is wrong. See Troubleshooting section.

---

## 6. Custom Single-Image Inference

### Step 1 — Prepare Your SAR Image

Your SAR image must be:
- Single-channel (grayscale) PNG
- dB-scaled and normalized to [0, 255]

If your image is raw linear SAR backscatter (not dB), preprocess it:
```python
from PIL import Image
import numpy as np

# Load your raw data (example: numpy array, float32)
raw = np.load("my_sar.npy")              # or load from .tif, .mat, etc.

# Convert to dB scale
db = 10 * np.log10(raw + 1e-6)          # add epsilon to avoid log(0)

# Normalize to [0, 255]
db_min, db_max = db.min(), db.max()
normalized = (db - db_min) / (db_max - db_min) * 255

# Save as 8-bit grayscale PNG
Image.fromarray(normalized.astype(np.uint8)).save("my_sar_input.png")
```

### Step 2 — Create an Input Directory

```bash
# Windows
mkdir my_sar_input
copy my_sar_image.png my_sar_input\

# Linux / macOS
mkdir -p my_sar_input
cp my_sar_image.png my_sar_input/
```

### Step 3 — Run Inference

```bash
python infer.py \
    --input_dir  my_sar_input/ \
    --output_dir my_eo_output/ \
    --weights    checkpoints/checkpoint_latest.pth
```

### Step 4 — View Result

```
my_eo_output/
└── my_sar_image.png   <- RGB color EO prediction
```

Open with any image viewer. For quantitative evaluation, you need a ground-truth EO image.

### Tips for Best Results

- **256x256 images work best**: The model is trained on 256x256. Other sizes are auto-resized.
- **Agricultural/rural areas**: The model was trained on agricultural SAR data and generalizes
  best to similar terrain types (fields, vegetation, rural land use).
- **Urban areas**: The model can generate urban structure but may hallucinate wrong colors
  for buildings vs roads vs parks — it has seen limited urban training data.
- **CPU vs GPU**: Use `--device cpu` if no GPU available. CPU inference is slower (~5-10 seconds
  per image on a modern CPU).
- **Batch inference**: Simply put all SAR images in one folder and run once. The script
  processes all images in the directory automatically.

---

## 7. Running Evaluation

Compute SSIM, PSNR, LPIPS, and FID comparing generated predictions to ground-truth EO images.

### Prerequisites

1. Run inference first to generate predictions (e.g., from `sample/` to `outputs/generated_eo/`):
   ```bash
   python infer.py --input_dir sample/ --output_dir outputs/generated_eo/ --weights checkpoints/checkpoint_latest.pth
   ```
2. Ground-truth EO images for the samples: `GT/`
3. Filenames between pred and GT must match (or follow the `_s1_` -> `_s2_` naming convention)

### Quick Evaluation (Minimal Flags)

```bash
python eval.py \
    --pred_dir outputs/generated_eo/ \
    --gt_dir   GT/
```

### Full Evaluation with All Outputs

```bash
python eval.py \
    --pred_dir    outputs/generated_eo/ \
    --gt_dir      GT/ \
    --output_csv  outputs/eval_results.csv \
    --output_json outputs/eval_results.json
```

### Evaluation on Test Split Only (Recommended for Full Dataset — No Data Leakage)

Use `--split_csv` when evaluating the full dataset to restrict evaluation to only the held-out test images:

```bash
python eval.py \
    --pred_dir    outputs/test_pred/ \
    --gt_dir      data/agri/s2/ \
    --split_csv   outputs/data_split.csv \
    --output_csv  outputs/eval_results.csv \
    --output_json outputs/eval_results.json
```

### All Evaluation Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--pred_dir` | Yes | — | Directory of generated EO predictions |
| `--gt_dir` | Yes | — | Directory of ground-truth EO images |
| `--output_csv` | No | outputs/eval_results.csv | Per-image results |
| `--output_json` | No | outputs/eval_results.json | Aggregate metrics JSON |
| `--split_csv` | No | None | Restrict to test split (recommended for standard dataset) |
| `--device` | No | auto | `cuda`, `cpu`, or `auto` |

### Reading the Output

**Terminal output** — per-image table + summary:
```
File                                   SSIM     PSNR    LPIPS
----------------------------------------------------------------------
ROIs1868_summer_s1_59_p10.png        0.3076   13.28   0.5051
ROIs1868_summer_s1_59_p11.png        0.3094   12.93   0.5053
...
============================================================
  EVALUATION RESULTS
============================================================
  Images evaluated : 24
  SSIM  (+)        : 0.3357
  PSNR  (+) (dB)   : 14.31
  LPIPS (-)        : 0.4705
  FID   (-)        : 328.83
============================================================
```

**eval_results.json** — aggregate metrics (machine-readable):
```json
{
  "n_images": 24,
  "ssim": 0.3357,
  "psnr": 14.31,
  "lpips": 0.4705,
  "fid": 328.83
}
```

**eval_results.csv** — per-image table with MEAN and FID rows at the bottom.

### Current Benchmark Results (Sample Evaluation)

| Metric | Score | n_images |
|--------|-------|----------|
| SSIM (higher is better) | 0.3357 | 24 |
| PSNR dB (higher is better) | 14.31 | 24 |
| LPIPS (lower is better) | 0.4705 | 24 |
| FID (lower is better) | 328.83 | 24 |

### Training Loss Progression (50 Epochs)

Key epochs from outputs/training_log.csv:

| Epoch | G Total (train) | G L1-Only (train) | D Loss (train) | G Total (val) | G L1 (val) | D Loss (val) | Note |
|-------|-----------------|-------------------|----------------|---------------|------------|--------------|------|
| 1  | 39.04 | 35.26 | 0.217 | 37.48 | 36.58 | 0.833 | Early training |
| 5  | 37.60 | 34.54 | 0.271 | 34.37 | 33.22 | 0.663 | Learning |
| 10 | 35.66 | 32.99 | 0.311 | 48.46 | 46.84 | 0.637 | Stable |
| 15 | 34.22 | 31.61 | 0.308 | 37.18 | 36.19 | 0.581 | Still improving |
| 20 | 34.03 | 31.66 | 0.298 | 35.30 | 32.78 | 0.529 | Plateau beginning |
| 25 | 33.93 | 31.56 | 0.300 | 35.23 | 32.69 | 0.614 | Slow descent |
| 30 | 33.81 | 31.53 | 0.298 | 34.93 | 32.66 | 0.514 | Converging |
| 35 | 33.68 | 31.45 | 0.313 | 34.93 | 32.56 | 0.546 | Converging |
| 40 | 33.66 | 31.36 | 0.325 | 34.85 | 32.47 | 0.695 | Near convergence |
| 45 | 33.73 | 31.30 | 0.317 | 45.85 | 41.36 | 0.702 | Near convergence |
| 50 | 33.66 | 31.22 | 0.309 | 34.95 | 32.42 | 0.627 | Final epoch |

D loss stable at ~0.3 throughout = healthy GAN. Full log: outputs/training_log.csv.

> FID computation runs Inception-v3 over all images — takes ~10-15 seconds for 24 images on CPU.
> This is expected behavior. Do not interrupt it.

### Common Evaluation Issues and Fixes

**"No matching image pairs found"**
The pred filenames have `_s1_` suffix (from SAR input filenames) but GT files have `_s2_`.
The script handles this automatically if `--split_csv` is passed. Without it, ensure
filenames match exactly OR use the `_s1_` -> `_s2_` convention.

**"Only 14 images evaluated instead of 600"**
Without `--split_csv`, the script tries exact filename matching. With only 14 exact
matches, pass `--split_csv outputs/data_split.csv` to use the full test split mapping.

**FID returns NaN**
Usually occurs when `pred_dir` or `gt_dir` is empty or has fewer than 2048 images
(FID needs many samples for reliable estimation). With 600 images, expect some
statistical noise in the FID estimate.

**LPIPS returns 0.0 for all images**
LPIPS library may not be installed: `pip install lpips`

---

## 8. End-to-End Pipeline (All Steps)

Run this sequence to reproduce the full pipeline from scratch:

```bash
# Step 1: Activate environment
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # Linux/macOS

# Step 2: Sanity check — run all unit tests
python test_dataset.py
python test_generator.py
python test_disciminator.py

# Step 3: Train from scratch
# (Or skip if using pre-trained weights from HuggingFace)
python train.py --config config.yaml

# Step 4: Run inference on the full training dataset's SAR images
# (This generates predictions for test-split evaluation)
python infer.py \
    --input_dir  data/agri/s1/ \
    --output_dir outputs/test_pred/ \
    --weights    checkpoints/checkpoint_latest.pth

# Step 5: Evaluate predictions vs. ground truth (test split only)
python eval.py \
    --pred_dir    outputs/test_pred/ \
    --gt_dir      data/agri/s2/ \
    --split_csv   outputs/data_split.csv \
    --output_csv  outputs/eval_results.csv \
    --output_json outputs/eval_results.json

# Step 6 (Optional): Evaluate on a custom image directory
python infer.py \
    --input_dir  my_custom_sar_images/ \
    --output_dir my_eo_results/ \
    --weights    checkpoints/checkpoint_latest.pth
```

---

## 9. Google Colab Workflow

A complete interactive architecture study and training walkthrough is available on Colab:
**[📓 SAR2EO Colab Notebook](https://colab.research.google.com/drive/1mnGETON8wCK6dQDFWhzME9ehlSSfzhyd?usp=sharing)**

For training on Colab (free GPU):

```python
# Cell 1: Mount Google Drive (for persistent storage)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone repo to fast local disk
%cd /content
!git clone https://github.com/VivanRajath/SARtoEO.git
%cd SARtoEO

# Cell 3: Install dependencies
!pip install -q -r requirements.txt

# Cell 4: Copy dataset from Drive to local (faster I/O during training)
!cp -r "/content/drive/MyDrive/SAR2EO/data" /content/SARtoEO/

# Cell 5: Train with Drive checkpointing (survives Colab disconnection)
!python train.py --config config.yaml \
    --dataset_path "/content/SARtoEO/data/agri" \
    --checkpoint_dir "/content/drive/MyDrive/SAR2EO/checkpoints" \
    --output_dir "/content/drive/MyDrive/SAR2EO/outputs"

# Cell 6: After training, run inference
!python infer.py \
    --input_dir  "/content/SARtoEO/data/agri/s1/" \
    --output_dir "/content/drive/MyDrive/SAR2EO/outputs/test_pred/" \
    --weights    "/content/drive/MyDrive/SAR2EO/checkpoints/checkpoint_latest.pth"
```

**Why use Drive checkpointing?**
- Colab sessions disconnect after ~12 hours (free tier) or may crash
- By writing checkpoints directly to Drive, training automatically resumes
  from the last saved epoch when you restart the session
- The CLI `--checkpoint_dir` override means you never have to edit config.yaml

---

## 10. Troubleshooting

### ModuleNotFoundError: No module named 'lpips'
```bash
pip install lpips
```

### ModuleNotFoundError: No module named 'pytorch_fid'
```bash
pip install pytorch-fid
```

### ModuleNotFoundError: No module named 'skimage'
```bash
pip install scikit-image
```

### ModuleNotFoundError: No module named 'reportlab'
```bash
pip install reportlab
```

### Weights file not found / KeyError loading checkpoint
Ensure the file is named exactly `checkpoint_latest.pth` (not with spaces):
```powershell
# Windows — rename if downloaded from Google Drive
Rename-Item "checkpoints\checkpoint_latest (1).pth" "checkpoint_latest.pth"
```

### CUDA out of memory during training
```yaml
# config.yaml
batch_size: 1   # reduce from 2 to 1
```
Or run on CPU:
```bash
python train.py --config config.yaml --device cpu
```
CPU training is slow (~10min/epoch) but functional.

### CUDA out of memory during inference
Add `--device cpu` flag:
```bash
python infer.py --input_dir ... --output_dir ... --weights ... --device cpu
```

### Generated images are all black or all white
- Checkpoint may be from an interrupted/corrupted save. Try `checkpoint_epoch_010.pth` instead.
- Verify checkpoint loads correctly: run `python test_generator.py` first.
- Check that input images are grayscale PNGs with values in [0, 255].

### Generated images look like random noise (not EO-like)
- Wrong checkpoint loaded. Verify the file is > 200MB. Small files indicate incomplete save.
- Model may need more training epochs. The generator requires enough epochs to learn the SAR->EO mapping.

### Dataset size shows 0 or too few images
Check that your data folder structure is correct:
```
data/agri/s1/   <- must contain SAR PNGs
data/agri/s2/   <- must contain EO PNGs with matching filenames
```
The pairing logic tries:
1. Exact filename match
2. Replace `_s1_` with `_s2_` in filename
3. Replace `s1` with `s2` anywhere in filename

### Evaluation matched 0 pairs (no pairs found)
Your prediction filenames do not match GT filenames. Check:
- Pred files (from infer.py) preserve the SAR `_s1_` naming
- GT files (in s2/) have `_s2_` naming
- Pass `--split_csv outputs/data_split.csv` to use automatic mapping

### FID takes very long (>10 minutes)
FID runs Inception-v3 on all images. For 600 images on CPU: ~5 minutes. This is normal.
On GPU: ~30 seconds. If no GPU: this is expected behavior, wait for completion.

### Training loss diverges (G loss increases rapidly)
```yaml
# config.yaml
learning_rate: 0.0001   # reduce from 0.0002
```

### Windows PowerShell: "cannot be loaded because running scripts is disabled"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Git push fails (remote rejected)
```bash
git remote -v                          # verify remote URL
git pull --rebase origin main          # sync before pushing
git push origin main
```

---

## 11. Config Reference

Full explanation of every option in `config.yaml`:

| Option | Default | Description |
|--------|---------|-------------|
| `dataset_path` | `"data/agri"` | Root folder containing `s1/` and `s2/` subdirs |
| `image_size` | `256` | All images resized to this square dimension |
| `sar_channels` | `1` | SAR input channels (1 for VV only) |
| `eo_channels` | `3` | EO output channels (3 for RGB) |
| `train_split` | `0.70` | Fraction of data for training |
| `val_split` | `0.15` | Fraction for validation |
| `test_split` | `0.15` | Fraction for held-out test |
| `seed` | `42` | Global random seed (numpy, torch, python random) |
| `augment` | `true` | Enable horizontal/vertical flip augmentation on train |
| `batch_size` | `2` | Images per forward pass |
| `num_epochs` | `50` | Total training epochs |
| `learning_rate` | `0.0002` | Adam optimizer LR (both G and D) |
| `optimizer` | `"adam"` | Optimizer type |
| `betas` | `[0.5, 0.999]` | Adam beta1, beta2 (paper values) |
| `lambda_l1` | `100` | L1 loss weight in G total loss |
| `ablation_log` | `true` | Also log G's L1-only loss each epoch |
| `ngf` | `64` | Base generator filter count |
| `ndf` | `64` | Base discriminator filter count |
| `checkpoint_dir` | `"checkpoints"` | Where to save .pth files |
| `checkpoint_every` | `1` | Save numbered checkpoint every N epochs |
| `gdrive_checkpoint_dir` | `""` | Google Drive mirror path (Colab only) |
| `emergency_save_every_n_batches` | `0` | Mid-epoch save frequency (0=disabled) |
| `output_dir` | `"outputs"` | Where to save logs, curves, samples |
| `sample_every` | `5` | Save qualitative triplet every N epochs |
| `device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `num_workers` | `"auto"` | `"auto"` = 0 on Windows, 2 on Linux |

---

## 12. Output File Reference

All outputs land in `outputs/` unless overridden with `--output_dir`.

| File | When Created | Description |
|------|-------------|-------------|
| `data_split.csv` | Training start | Train/val/test filename assignments. Columns: filename, split |
| `training_log.csv` | Each epoch | Per-epoch losses: epoch, train_g_total, train_g_l1_only, train_d_loss, val_g_total, val_g_l1_only, val_d_loss |
| `loss_curve.png` | Training end | 3-panel matplotlib figure: top=G total, middle=G L1-only (ablation), bottom=D loss |
| `test_metrics.json` | Training end | `{"test_g_l1_loss": X.XX}` — L1 loss on test split |
| `sample_epoch_NNN.png` | Every N epochs | Triplet grid: SAR input | Generated EO | Ground Truth EO |
| `test_example_01..05.png` | Training end | 5 qualitative test examples in triplet format |
| `test_pred/` | After infer.py | Directory of generated EO PNG images (one per SAR input) |
| `eval_results.csv` | After eval.py | Per-image SSIM, PSNR, LPIPS + MEAN row + FID row |
| `eval_results.json` | After eval.py | `{"n_images": N, "ssim": X, "psnr": X, "lpips": X, "fid": X}` |

---

*For detailed architecture explanation, see [architecture.md](architecture.md).*  
*For the interactive Colab study, see [Google Colab Architecture Study](https://colab.research.google.com/drive/1mnGETON8wCK6dQDFWhzME9ehlSSfzhyd?usp=sharing).*  
*For model weights, see [HuggingFace: VivanRajath/SAR2EO](https://huggingface.co/VivanRajath/SAR2EO).*
