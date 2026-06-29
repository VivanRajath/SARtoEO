# SAR-to-EO Image Translation using Pix2Pix cGAN

> **GalaxEye Space — Technical Assignment | AI Research Intern**

A **Conditional Generative Adversarial Network (Pix2Pix)** that translates single-channel Sentinel-1 SAR imagery
into Sentinel-2 RGB Electro-Optical images. SAR sensors image Earth through clouds and darkness, but produce
grayscale, speckle-heavy outputs with no colour information. This project addresses the inherently ill-posed
problem of hallucinating perceptually realistic optical imagery from radar backscatter.

| Resource | Link |
|----------|------|
| Model Weights | [HuggingFace — VivanRajath/SAR2EO](https://huggingface.co/VivanRajath/SAR2EO) |
| Technical Report | [Google Drive Link](YOUR_GOOGLE_DRIVE_LINK_HERE) |
| Architecture Deep-Dive | [architecture.md](architecture.md) |
| Operational Runbook | [RUNBOOK.md](RUNBOOK.md) |

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
We implement **Pix2Pix** — a conditional GAN where a U-Net generator produces EO images conditioned on SAR input,
supervised by a PatchGAN discriminator that classifies overlapping 70×70 patches as real or fake.

**Why Pix2Pix?**
- Pixel-aligned paired training data is available (Sentinel-1/2 co-registered)
- L1 loss enforces global structure; GAN loss enforces local texture realism
- PatchGAN is computationally efficient and captures high-frequency detail
- Fast inference: single forward pass, no iterative denoising (unlike diffusion)
- Well within the 16GB VRAM limit (uses <3GB at batch size 2)

For the full technical rationale comparing Pix2Pix vs CycleGAN vs Diffusion Models, see [architecture.md](architecture.md).

### Architecture Overview

```
                    [SAR Input] (1 x 256 x 256)
                         │
                  ┌──────▼──────┐
                  │ Encoder (e1)│ (64 x 128 x 128)
                  └──────┬──────┘
                         ├─────────────────────────────────────────┐ (Skip)
                  ┌──────▼──────┐                                  │
                  │ Encoder (e2)│ (128 x 64 x 64)                  │
                  └──────┬──────┘                                  │
                         ├───────────────────────────────────┐     │
                  ┌──────▼──────┐                            │     │
                  │ Encoder (e3)│ (256 x 32 x 32)            │     │
                  └──────┬──────┘                            │     │
                         ├─────────────────────────────┐     │     │
                  ┌──────▼──────┐                      │     │     │
                  │ Encoder (e4)│ (512 x 16 x 16)      │     │     │
                  └──────┬──────┘                      │     │     │
                         ├───────────────────────┐     │     │     │
                  ┌──────▼──────┐                │     │     │     │
                  │ Encoder (e5)│ (512 x 8 x 8)  │     │     │     │
                  └──────┬──────┘                │     │     │     │
                         ├─────────────────┐     │     │     │     │
                  ┌──────▼──────┐          │     │     │     │     │
                  │ Encoder (e6)│ (512x4x4)│     │     │     │     │
                  └──────┬──────┘          │     │     │     │     │
                         ├───────────┐     │     │     │     │     │
                  ┌──────▼──────┐    │     │     │     │     │     │
                  │ Encoder (e7)│    │     │     │     │     │     │
                  └──────┬──────┘    │     │     │     │     │     │
                         │           │     │     │     │     │     │
                  ┌──────▼──────┐    │     │     │     │     │     │
                  │ Bottleneck  │    │     │     │     │     │     │ (512 x 1 x 1)
                  └──────┬──────┘    │     │     │     │     │     │
                         │           │     │     │     │     │     │
                  ┌──────▼──────┐    │     │     │     │     │     │
                  │ Decoder (d7)◄────┘     │     │     │     │     │ (1024 x 2 x 2)
                  └──────┬──────┘          │     │     │     │     │
                  ┌──────▼──────┐          │     │     │     │     │
                  │ Decoder (d6)◄──────────┘     │     │     │     │ (1024 x 4 x 4)
                  └──────┬──────┘                │     │     │     │
                  ┌──────▼──────┐                │     │     │     │
                  │ Decoder (d5)◄────────────────┘     │     │     │ (1024 x 8 x 8)
                  └──────┬──────┘                      │     │     │
                  ┌──────▼──────┐                      │     │     │
                  │ Decoder (d4)◄──────────────────────┘     │     │ (1024 x 16 x 16)
                  └──────┬──────┘                            │     │
                  ┌──────▼──────┐                            │     │
                  │ Decoder (d3)◄────────────────────────────┘     │ (512 x 32 x 32)
                  └──────┬──────┘                                  │
                  ┌──────▼──────┐                                  │
                  │ Decoder (d2)◄──────────────────────────────────┘ (256 x 64 x 64)
                  └──────┬──────┘
                  ┌──────▼──────┐
                  │ Decoder (d1)◄──────────────────────────────────── (128 x 128 x 128)
                  └──────┬──────┘
                  ┌──────▼──────┐
                  │    Final    │ (ConvTranspose2d + Tanh)
                  └──────┬──────┘
                         │
                  [EO Output] (3 x 256 x 256)
```

**Generator**: 8-level U-Net — 7 encoder stages + bottleneck + 7 decoder stages, ~54M parameters.
Skip connections preserve spatial detail across the bottleneck.

**Discriminator**: 70×70 PatchGAN — classifies 70x70 pixel patches as real/fake, not the whole image.
~2.8M parameters. Conditioned on SAR input to enforce cross-modal consistency.

**Loss**:
```
G_loss = BCE(D(SAR, fake_EO), real) + 100 * L1(fake_EO, real_EO)
D_loss = [BCE(D(SAR, real_EO), real) + BCE(D(SAR, fake_EO), fake)] / 2
```

### Dataset Selection
We use the **Sentinel-1 & Sentinel-2 Terrain-Separated Image Pairs (Kaggle)** dataset, specifically the
**agricultural subset** (`data/agri/`). Agricultural regions were chosen because:
- Distinct SAR textures (field boundaries, furrow patterns, irrigation circles) map consistently to optical colour
- Terrain-separated split minimises geographic leakage between train/val/test
- Subset (4,000 pairs) is manageable for resource-constrained environments

**Split**: 70/15/15 (train/val/test) — reproducible random split seeded at 42.
Indices saved to `outputs/data_split.csv`.

### Repository Structure

```
SAR2EO/
├── checkpoints/          <- Saved model checkpoints (.pth, ignored by git except .gitkeep)
│   └── .gitkeep
├── data/
│   └── agri/             <- Training dataset root folder
│       ├── s1/           <- SAR input images (Sentinel-1 VV, grayscale PNG, 256x256)
│       └── s2/           <- EO target images (Sentinel-2 RGB PNG, 256x256)
├── GT/                   <- Sentinel-2 RGB ground truth for sample images (eval only)
├── outputs/              <- Generated outputs (loss curves, logs, prediction results)
│   └── .gitkeep
├── sample/               <- Sentinel-1 SAR sample images for quick inference testing
│   └── .gitkeep
├── config.yaml           <- Hyperparameters, paths, and training config settings
├── dataset.py            <- Custom PyTorch paired SAR/EO dataset loader
├── discriminator.py      <- PatchGAN discriminator network architecture (~2.8M params)
├── download_weights.py   <- Weight downloader utility from HuggingFace
├── eval.py               <- Metric evaluation script (SSIM, PSNR, LPIPS, FID)
├── generate_report.py    <- Automated PDF technical report generator
├── generator.py          <- U-Net generator model architecture (~54M params)
├── infer.py              <- Inference CLI script (translates directory of SAR -> EO)
├── plot_loss.py          <- Utility script to plot training curves from training_log.csv
├── requirements.txt      <- Pinned python dependencies
├── train.py              <- Model training and validation script (supports resume)
├── utils.py              <- Shared helper functions (seed, grids, normalization)
├── README.md             <- Project documentation landing page
├── RUNBOOK.md            <- Operations and setup guide
└── architecture.md       <- Detailed architecture description and convergence analysis
```

---

## 2. Requirements and Versions

- **Python**: 3.10 or higher (tested on 3.12)

| Package | Min Version | Purpose |
|---|---|---|
| `torch` | >= 2.2.0 | Deep learning framework |
| `torchvision` | >= 0.17.0 | Image transforms, save_image |
| `numpy` | >= 1.24.0 | Array operations |
| `Pillow` | >= 10.0.0 | Image I/O |
| `matplotlib` | >= 3.8.0 | Loss curve plots |
| `tqdm` | >= 4.66.0 | Progress bars |
| `PyYAML` | >= 6.0.1 | Config file parsing |
| `scikit-image` | >= 0.22.0 | SSIM and PSNR metrics |
| `lpips` | >= 0.1.4 | Learned perceptual image patch similarity |
| `pytorch-fid` | >= 0.3.0 | Frechet Inception Distance |
| `opencv-python` | >= 4.9.0 | Optional image utilities |
| `reportlab` | >= 4.0.0 | PDF report generation |

---

## 3. Environment Setup

### Local (Windows / Linux / macOS)

```bash
# 1. Clone the repository
git clone https://github.com/VivanRajath/SARtoEO.git
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

For Colab training, use CLI overrides to redirect checkpoints to Google Drive:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
%cd /content
!git clone https://github.com/VivanRajath/SARtoEO.git
%cd SARtoEO
!pip install -q -r requirements.txt

# Train with Drive checkpointing
!python train.py --config config.yaml \
    --dataset_path "/content/drive/MyDrive/SAR2EO/data/agri" \
    --checkpoint_dir "/content/drive/MyDrive/SAR2EO/checkpoints" \
    --output_dir "/content/drive/MyDrive/SAR2EO/outputs"
```

See [RUNBOOK.md#9-google-colab-workflow](RUNBOOK.md) for complete Colab instructions.

---

## 4. Dataset Structure

The pipeline expects this layout:

```
data/
+-- agri/
    |-- s1/
    |   |-- patch_001_s1_agri.png
    |   |-- patch_002_s1_agri.png
    |   +-- ...
    +-- s2/
        |-- patch_001_s2_agri.png
        |-- patch_002_s2_agri.png
        +-- ...
```

**Pairing rules** (tried in order):
1. Replace `_s1_` with `_s2_` in filename (Kaggle convention)
2. Replace `s1` with `s2` anywhere in filename
3. Identical filenames (SEN1-2 / SEN12MS convention)

**Supported formats**: `.png`, `.jpg`, `.tif`, `.tiff`

**Any resolution accepted** — images auto-resized to 256×256 during loading.

---

## 5. Training Command

```bash
# Standard training (reads all settings from config.yaml)
python train.py --config config.yaml
```

**Training produces:**

| File | Description |
|---|---|
| `checkpoints/checkpoint_latest.pth` | Always-up-to-date checkpoint (use for inference) |
| `checkpoints/checkpoint_epoch_NNN.pth` | Numbered checkpoint every N epochs |
| `outputs/training_log.csv` | Per-epoch: G(total), G(L1-only), D losses for train+val |
| `outputs/loss_curve.png` | 3-panel loss curve (ablation + discriminator) |
| `outputs/data_split.csv` | Exact train/val/test indices for reproducibility |
| `outputs/sample_epoch_NNN.png` | SAR | Generated EO | GT triplet every N epochs |
| `outputs/test_example_01..05.png` | 5 qualitative triplets from test split |
| `outputs/test_metrics.json` | Test-set L1 loss |

**Key config options** (`config.yaml`):

```yaml
num_epochs: 50          # increase to 100-200 for better quality
batch_size: 2           # increase if you have more VRAM
lambda_l1: 100          # higher = more structurally faithful
augment: true           # random flips during training
ablation_log: true      # log L1-only loss alongside GAN+L1
```

> **Note on training duration**: 50 epochs was used for this submission. The model is still
> learning at epoch 50 — extending to 100-200 epochs will further improve all metrics.
> See [architecture.md#11-training-epochs-and-convergence](architecture.md) for details.

---

## 6. Inference Command

**CLI required by the GalaxEye assessment:**

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

For custom single-image inference and preprocessing, see [RUNBOOK.md#6-custom-single-image-inference](RUNBOOK.md).

---

## 7. Evaluation Command

After running inference on sample SAR images, compute metrics against ground-truth EO images:

```bash
python eval.py \
    --pred_dir    outputs/generated_eo/ \
    --gt_dir      GT/ \
    --output_csv  outputs/eval_results.csv \
    --output_json outputs/eval_results.json
```

**Outputs:**
- Per-image SSIM, PSNR, LPIPS printed to terminal and saved to CSV
- FID computed over the full directory
- `outputs/eval_results.json` — aggregate metrics

---

## 8. Model Weights

The trained checkpoint (~654 MB) is hosted on HuggingFace and **not included in this repo**. Download it with one command:

```bash
python download_weights.py
```

This will:
- Automatically download `checkpoint_latest.pth` from [VivanRajath/SAR2EO](https://huggingface.co/VivanRajath/SAR2EO)
- Save it directly to `checkpoints/checkpoint_latest.pth`
- Skip re-downloading if the file already exists
- Show a live progress bar

> **Faster downloads** (optional): install `huggingface_hub` for a resumable download:
> ```bash
> pip install huggingface_hub
> python download_weights.py
> ```
> Without it, the script falls back to `urllib` automatically — no extra install needed.

Once downloaded, run inference:
```bash
python infer.py \
    --input_dir  sample/ \
    --output_dir outputs/generated_eo/ \
    --weights    checkpoints/checkpoint_latest.pth
```


---

## 9. Results

### Quantitative Metrics (24 Sample Images)

| Split | LPIPS (lower better) | FID (lower better) | SSIM (higher better) | PSNR dB (higher better) |
|---|---|---|---|---|
| **Sample (n=24)** | **0.4705** | **328.83** | **0.3357** | **14.31** |

*Metrics computed on 24 sample images using `eval.py`. Run `python eval.py --pred_dir outputs/generated_eo/ --gt_dir GT/` to verify.*

### Metric Interpretation
- **LPIPS 0.4705**: Moderate perceptual distance — model captures broad structure but misses some fine texture
- **FID 328.83**: Generated images match basic EO statistical distributions, though with higher variance due to the small sample size
- **SSIM 0.3357**: Captures spatial structure and boundaries well, but penalized slightly by color variations
- **PSNR 14.31 dB**: Expected range for different-modality cross-modal synthesis tasks

For detailed analysis including success cases, failure modes, and why these scores are expected for
this ill-posed task, see [Technical Report (Google Drive)](YOUR_GOOGLE_DRIVE_LINK_HERE).

### Qualitative Examples

After training, qualitative triplet comparisons (SAR Input | Generated EO | Ground Truth EO) are saved to:
- `outputs/sample_epoch_NNN.png` — periodic training samples
- `outputs/test_example_01..05.png` — 5 held-out test examples

### Loss Curves

Training and validation loss curves (including ablation comparison: GAN+L1 vs L1-only) are saved to
`outputs/loss_curve.png`.

---

## 10. References

1. **Pix2Pix**: Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*. [[Paper]](https://arxiv.org/abs/1611.07004)

2. **SEN1-2 Dataset**: Schmitt, M., Hughes, L. H., & Zhu, X. X. (2018). SEN1-2: A Dataset for Deep Learning in SAR-Optical Data Fusion. *ISPRS Annals*. [[Dataset]](https://mediatum.ub.tum.de/1474000)

3. **SEN12MS Dataset**: Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). SEN12MS — A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery. *ISPRS Annals*. [[Dataset]](https://mediatum.ub.tum.de/1474000)

4. **Kaggle Sentinel-1/2 Pairs**: Sentinel-1 & Sentinel-2 Image Pairs (Terrain-Separated). [[Dataset]](https://www.kaggle.com/datasets)

5. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. [[Paper]](https://arxiv.org/abs/1505.04597)

6. **LPIPS**: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*. [[Paper]](https://arxiv.org/abs/1801.03924)

7. **FID**: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*. [[Paper]](https://arxiv.org/abs/1706.08500)
