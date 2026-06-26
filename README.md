# SAR-to-EO Image Translation using Pix2Pix cGAN

> **GalaxEye Space — Technical Assignment | AI Research Intern**

A **Conditional Generative Adversarial Network (Pix2Pix)** that translates single-channel Sentinel-1 SAR imagery into Sentinel-2 RGB Electro-Optical images. SAR sensors image Earth through clouds and darkness, but produce grayscale, texture-heavy outputs with no colour information. This project addresses the inherently ill-posed problem of hallucinating perceptually realistic optical imagery from radar backscatter.

---

## Table of Contents
1. [Project Description](#1-project-description)
2. [Requirements and Versions](#2-requirements-and-versions)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Structure](#4-dataset-structure)
5. [Training Command](#5-training-command)
6. [Inference Command](#6-inference-command)
7. [Evaluation Command](#7-evaluation-command)
8. [Model Weights](#8-model-weights)
9. [Results](#9-results)
10. [References](#10-references)

---

## 1. Project Description

### Task
Given a 256×256 Sentinel-1 VV SAR image, generate the corresponding Sentinel-2 RGB optical image.

### Approach
We implement **Pix2Pix** — a conditional GAN where a U-Net generator produces EO images conditioned on SAR input, supervised by a PatchGAN discriminator that classifies overlapping 70×70 patches as real or fake.

**Why Pix2Pix?**
- Pixel-aligned paired training data is available (Sentinel-1/2 co-registered)
- L1 loss enforces global structure; GAN loss enforces local texture realism
- PatchGAN is computationally efficient and captures high-frequency detail

### Architecture

```
SAR Input [1×256×256]
       │
  ┌────▼──────────────────────────── U-Net Encoder ────────────────────────────────┐
  │  e1: 1→64   e2: 64→128   e3: 128→256   e4: 256→512   e5→e7: 512→512          │
  └────────────────────────────────────────────────────────── Bottleneck: 512→512 ──┘
                                   (1×1 global context)
  ┌────────────────────────────── U-Net Decoder (with skip connections) ────────────┐
  │  d7: 512+512→512  d6: →512  d5: →512  d4: →512  d3: →256  d2: →128  d1: →64  │
  └──────────────────────────────────────── final: 128→3 ch + Tanh ────────────────┘
                                       │
                               EO Output [3×256×256]
                             (values in [-1, 1] → de-normalize → [0, 1] PNG)
```

**Generator**: 8-level U-Net (standard Pix2Pix, Isola et al. 2017) — 7 encoder stages + bottleneck + 7 decoder stages, skip connections preserve spatial detail.

**Discriminator**: 70×70 PatchGAN — classifies overlapping patches rather than the whole image, encouraging high-frequency texture realism.

**Loss**:
```
G_loss = GAN_loss(D(SAR, fake_EO), real) + λ × L1(fake_EO, real_EO)
D_loss = [GAN_loss(D(SAR, real_EO), real) + GAN_loss(D(SAR, fake_EO), fake)] / 2

λ = 100  (paper default)
```

### Dataset Selection
We use the **Sentinel-1 & Sentinel-2 Terrain-Separated Image Pairs (Kaggle)** dataset, specifically the **agricultural subset** (`data/agri/`). This subset was chosen because:
- Agricultural areas have characteristic SAR textures (field boundaries, furrow patterns) that map consistently to optical colour
- The terrain-separated split avoids geographic leakage between train/val/test
- Subset size is manageable for resource-constrained training

**Preprocessing**:
- SAR images: assumed dB-scaled and normalised to [0, 255] by the dataset; `ToTensor()` maps to [0, 1]
- EO images: `ToTensor()` maps to [0, 1], then `Normalize(0.5, 0.5)` maps to [-1, 1] to match the generator's Tanh output range
- All images auto-resized to 256×256 (antialias bilinear)

**Split**: 70 / 15 / 15 (train / val / test) — reproducible random split seeded at 42, indices saved to `outputs/data_split.csv`.

### Repository Structure

```
SAR2EO/
├── data/
│   └── agri/
│       ├── s1/              ← Sentinel-1 VV SAR patches (256×256 grayscale PNG)
│       └── s2/              ← Sentinel-2 RGB EO patches (256×256 RGB PNG)
│
├── checkpoints/             ← Saved model checkpoints (.pth)
├── outputs/                 ← Training logs, loss curves, samples, eval results
│
├── dataset.py               ← Robust SAR/EO paired dataset loader
├── generator.py             ← 8-level Pix2Pix U-Net generator
├── discriminator.py         ← 70×70 PatchGAN discriminator
├── train.py                 ← Training script (3-way split, ablation log, CSV/PNG output)
├── infer.py                 ← CLI inference (GalaxEye spec-compliant)
├── eval.py                  ← Evaluation: LPIPS, FID, SSIM, PSNR
├── utils.py                 ← Shared helpers (denormalize, triplet grid, seed, etc.)
│
├── config.yaml              ← All hyperparameters and paths
├── requirements.txt         ← Pinned Python dependencies
└── README.md
```

---

## 2. Requirements and Versions

- **Python**: 3.10 or higher (tested on 3.12)

| Package | Min Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.2.0 | Deep learning framework |
| `torchvision` | ≥ 0.17.0 | Image transforms, save_image |
| `numpy` | ≥ 1.24.0 | Array operations |
| `Pillow` | ≥ 10.0.0 | Image I/O |
| `matplotlib` | ≥ 3.8.0 | Loss curve plots |
| `tqdm` | ≥ 4.66.0 | Progress bars |
| `PyYAML` | ≥ 6.0.1 | Config file parsing |
| `scikit-image` | ≥ 0.22.0 | SSIM and PSNR metrics |
| `lpips` | ≥ 0.1.4 | Perceptual image similarity |
| `pytorch-fid` | ≥ 0.3.0 | Fréchet Inception Distance |
| `opencv-python` | ≥ 4.9.0 | Optional image utilities |

---

## 3. Environment Setup

### Local (VSCode / Windows / Linux)

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd SAR2EO

# 2. Create and activate a virtual environment
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab (Training)

```python
# Cell 1 — Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone <your-repository-url> SAR2EO
%cd SAR2EO

# Cell 2 — Install dependencies
!pip install -q -r requirements.txt

# Cell 3 — Upload / link your dataset
# Option A: upload the agri/ folder directly to Colab
# Option B: it's already in Drive at /content/drive/MyDrive/SAR2EO/data/agri/

# Cell 4 — Train
!python train.py --config config.yaml

# Cell 5 — Download the checkpoint after training
from google.colab import files
files.download('checkpoints/checkpoint_latest.pth')
# Or download a specific epoch: files.download('checkpoints/checkpoint_epoch_100.pth')
```

**After downloading the checkpoint**, place it in your local `checkpoints/` folder and use `infer.py` locally in VSCode.

---

## 4. Dataset Structure

The pipeline expects this layout (subfolder names are case-insensitive — `s1`, `S1`, `sar` all work):

```
data/
└── agri/
    ├── s1/
    │   ├── patch_001_s1_agri.png
    │   ├── patch_002_s1_agri.png
    │   └── ...
    └── s2/
        ├── patch_001_s2_agri.png
        ├── patch_002_s2_agri.png
        └── ...
```

**Pairing rules** (tried in order):
1. Replace `_s1_` → `_s2_` in the filename (Kaggle convention)
2. Replace `s1` → `s2` anywhere in the filename
3. Identical filenames (SEN1-2 / SEN12MS convention)

**Supported formats**: `.png`, `.jpg`, `.tif`, `.tiff`

**Any resolution is accepted** — images are auto-resized to 256×256 during loading.

---

## 5. Training Command

```bash
# Standard training (reads all settings from config.yaml)
python train.py --config config.yaml

# Custom config file
python train.py --config my_config.yaml
```

**Training produces:**

| File | Description |
|---|---|
| `checkpoints/checkpoint_latest.pth` | Always-up-to-date checkpoint (use for inference) |
| `checkpoints/checkpoint_epoch_NNN.pth` | Numbered checkpoint every `checkpoint_every` epochs |
| `outputs/training_log.csv` | Per-epoch: G(total), G(L1-only), D losses for train + val |
| `outputs/loss_curve.png` | 3-panel loss curve (ablation + discriminator + decomposition) |
| `outputs/data_split.csv` | Exact train/val/test indices for reproducibility |
| `outputs/sample_epoch_NNN.png` | SAR \| Generated EO \| Ground Truth triplet (every N epochs) |
| `outputs/test_example_01..05.png` | 5 qualitative triplets from the held-out test set |
| `outputs/test_metrics.json` | Test-set L1 loss (run eval.py for full LPIPS/FID/SSIM/PSNR) |

**Key config options** (`config.yaml`):

```yaml
num_epochs: 100         # increase for better quality
batch_size: 2           # increase if you have more VRAM (4 on 16GB GPU)
lambda_l1: 100          # higher = more faithful to structure, less creative
augment: true           # random flips during training
ablation_log: true      # log L1-only loss alongside GAN+L1 for comparison
```

---

## 6. Inference Command

**Exact CLI required by the GalaxEye assessment:**

```bash
python infer.py \
    --input_dir  <path_to_sar_dir> \
    --output_dir <path_to_output_dir> \
    --weights    checkpoints/checkpoint_latest.pth
```

**Example:**
```bash
python infer.py \
    --input_dir  sample_sar_images/ \
    --output_dir generated_eo/ \
    --weights    checkpoints/checkpoint_latest.pth
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | (required) | Directory of SAR PNG patches |
| `--output_dir` | (required) | Directory to save generated EO images |
| `--weights` | (required) | Path to `.pth` checkpoint |
| `--image_size` | 256 | Generator internal resolution |
| `--device` | auto | `cuda`, `cpu`, or `auto` |

**Notes:**
- Input images can be any resolution — they are auto-resized internally and the output is returned to the original size
- Output filenames match input filenames exactly (`.png` extension enforced)
- Works with CPU only (no GPU required for inference)

---

## 7. Evaluation Command

After running inference, compute metrics against ground-truth:

```bash
python eval.py \
    --pred_dir generated_eo/ \
    --gt_dir   data/agri/s2/
```

**With JSON output:**
```bash
python eval.py \
    --pred_dir    generated_eo/ \
    --gt_dir      data/agri/s2/ \
    --output_csv  outputs/eval_results.csv \
    --output_json outputs/eval_results.json
```

**Outputs:**
- Per-image SSIM, PSNR, LPIPS printed to terminal and saved to CSV
- FID computed over the full directory
- `outputs/eval_results.json` — aggregate metrics for documentation

---

## 8. Model Weights

Download the trained generator checkpoint:

> **Weights Download Link**: *(Add Google Drive / HuggingFace link here after training)*

Place the downloaded `.pth` file in the `checkpoints/` directory:
```
checkpoints/
└── checkpoint_latest.pth   ← place downloaded weights here
```

Then run inference as described in [Section 6](#6-inference-command).

---

## 9. Results

### Quantitative Metrics

| Split | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ (dB) |
|---|---|---|---|---|
| **Validation** | TBD | TBD | TBD | TBD |
| **Test** | TBD | TBD | TBD | TBD |

*Metrics will be populated after full training run. Run `eval.py` on the generated outputs.*

### Qualitative Examples

After training, qualitative triplet comparisons (SAR Input | Generated EO | Ground Truth EO) are saved to:
- `outputs/sample_epoch_NNN.png` — periodic training samples
- `outputs/test_example_01..05.png` — 5 held-out test examples

### Loss Curves

Training and validation loss curves (including ablation comparison: GAN+L1 vs L1-only) are saved to `outputs/loss_curve.png`.

---

## 10. References

1. **Pix2Pix**: Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*. [[Paper]](https://arxiv.org/abs/1611.07004)

2. **SEN1-2 Dataset**: Schmitt, M., Hughes, L. H., & Zhu, X. X. (2018). SEN1-2: A Dataset for Deep Learning in SAR-Optical Data Fusion. *ISPRS Annals*. [[Dataset]](https://mediatum.ub.tum.de/1474000)

3. **SEN12MS Dataset**: Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). SEN12MS — A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion. *ISPRS Annals*. [[Dataset]](https://mediatum.ub.tum.de/1474000)

4. **Kaggle Sentinel-1/2 Pairs**: Sentinel-1 & Sentinel-2 Image Pairs (Terrain-Separated). [[Dataset]](https://www.kaggle.com/datasets) — Agricultural subset used in this work.

5. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. [[Paper]](https://arxiv.org/abs/1505.04597)

6. **LPIPS**: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*. [[Paper]](https://arxiv.org/abs/1801.03924)
