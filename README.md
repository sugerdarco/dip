# Medical Image Compression Pipeline
### DWT + Cross-Attention Learning (CAL) + VAE
*Implements: Fan Dai, Scientific Reports (2025) 15:40008*

---

## Directory structure

```
working_dir/
├── contents/                    ← YOUR INPUT IMAGES (put them here)
│   ├── mri/
│   │   ├── brain_001.dcm
│   │   └── brain_002.png
│   └── ct/
│       ├── thorax_001.dcm
│       └── lung_002.tiff
│
├── output/                      ← ALL RESULTS (auto-created)
│   └── <image_title>/
│       └── results_of_<image_title>/
│           ├── 00_input_normalised.png
│           ├── 01_dwt_LL.png
│           ├── 01_dwt_LH.png
│           ├── 01_dwt_HL.png
│           ├── 01_dwt_HH.png
│           ├── 01_dwt_subbands.npy
│           ├── 03_cal_attention_map.png
│           ├── 03_encoder_features.npy
│           ├── 04_vae_z.npy
│           ├── 04_vae_mu.npy
│           ├── 04_vae_logvar.npy
│           ├── 05_compressed.bin          ← compressed bitstream
│           ├── 05_entropy_meta.json
│           ├── 05_compression_stats.txt
│           ├── 06_coarse_reconstruction.png
│           ├── 07_decoded_subbands.npy
│           ├── 07_decoded_feature_vis.png
│           ├── 08_dnn_pixel_refined_3x3.png
│           ├── 08_dnn_pixel_refined_1x8.png   ← only if --window both
│           ├── 08_dnn_pixel_refined_final.png
│           ├── 09_spatial_filtered.png        ← FINAL reconstructed image
│           ├── 09_filter_kernel.npy
│           ├── 09_filter_kernel.txt
│           ├── 10_comparison.png              ← side-by-side visual
│           ├── 10_metrics.json
│           └── 10_report.txt
│
└── checkpoints/
    ├── best_model.pth
    └── training_history.json

medical_compression/              ← THIS PROJECT
├── config/
│   └── config.py                 ← ALL hyperparameters here
├── stages/
│   ├── stage1_loader.py          ← DICOM + raster image loading
│   ├── stage2_dwt.py             ← DWT decompose / reconstruct
│   ├── stage3_cal_encoder.py     ← Cross-Attention Learning encoder
│   ├── stage4_vae_encoder.py     ← VAE encoder (μ, σ², z)
│   ├── stage5_entropy.py         ← Entropy coding (quantise + zlib)
│   ├── stage6_vae_decoder.py     ← VAE decoder (z → subbands)
│   ├── stage7_dnn_pixel.py       ← DNN pixel estimation (3×3 + 1×8)
│   └── stage8_spatial_filter.py  ← Trainable spatial filter
├── utils/
│   └── metrics.py                ← PSNR, SSIM, MSE, NCC
├── train.py                      ← End-to-end training
├── run_pipeline.py               ← Inference on all images
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone / copy this project into your working directory
cd working_dir

# 2. Install PyTorch with CUDA 12.1 (for your 24GB GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r medical_compression/requirements.txt
```

---

## Quick start

### Step 1 — Put your images in the contents folder
```
working_dir/contents/
    mri/brain_scan.dcm
    ct/chest_001.png
    ...
```
Supports: `.dcm`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`  
All images are auto-resized to 512×512 grayscale.

---

### Step 2 — Train the model
```bash
cd working_dir
python -m medical_compression.train \
    --contents working_dir/contents \
    --epochs   150
```

Training config (from paper Table 2):
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Batch size | 32 |
| Epochs | 150 |
| Weight init | Xavier Normal |
| Dropout | 0.3 |
| Early stopping | patience=10 |
| Loss | λ₁·MSE + λ₂·Perceptual + λ₃·KL |

Best checkpoint saved to `working_dir/checkpoints/best_model.pth`

---

### Step 3 — Run inference on all images
```bash
python -m medical_compression.run_pipeline
```

### Run on a single image
```bash
python -m medical_compression.run_pipeline \
    --image working_dir/contents/mri/brain_001.dcm
```

### Run both DNN pixel window modes
```bash
python -m medical_compression.run_pipeline --window both
```

---

## Pipeline stages

```
Input image (512×512 grayscale)
    │
    ▼ Stage 1 — Load + preprocess (DICOM / PNG / TIFF)
    │
    ▼ Stage 2 — DWT decomposition → {LL, LH, HL, HH}  [Eq. 1]
    │           LL kept intact · high-freq selectively compressed
    │
    ▼ Stage 3 — Convolutional encoder + Cross-Attention Learning (CAL)
    │           Q from LL-path, K/V from HF-path
    │           Attention(Q,K,V) = softmax(QKᵀ/√dk)·V  [Eq. 3]
    │           MultiHead = Concat(head₁…headₕ)·Wₒ     [Eq. 4]
    │           8 heads · embed_dim=64
    │
    ▼ Stage 4 — VAE encoder → (μ, log_var, z)
    │           Loss = λ₁MSE + λ₂Perceptual + λ₃KL     [Eq. 2]
    │           latent_dim=256
    │
    ▼ Stage 5 — Entropy coding (quantise int8 + zlib deflate)
    │           → compressed.bin  (~2.9 bpp average)
    │
    ════════════ TRANSMISSION / STORAGE ════════════
    │
    ▼ Stage 5 — Entropy decode
    │
    ▼ Stage 6+7 — VAE decoder → reconstructed DWT subbands
    │
    ▼ Stage 8 — Inverse DWT → coarse spatial image
    │
    ▼ Stage 8 — DNN pixel estimation (two modes):
    │           • 3×3 window: layers [20,18,5,1]  R=0.997  (default)
    │           • 1×8 window: layers [20,18,5,2]  R=0.951  (faster)
    │
    ▼ Stage 9 — Trainable spatial filter (3×3 attention-aware conv)
    │           Learns via SSIM + MSE backprop
    │           PSNR=31.7dB  SSIM=0.91  (vs GWO 27.4dB, WHO 28.1dB)
    │
    ▼ Stage 10 — Metrics (PSNR, SSIM, MSE, NCC) + comparison PNG
```

---

## Output files explained

| File | Description |
|------|-------------|
| `00_input_normalised.png` | Preprocessed input (512×512 grayscale) |
| `01_dwt_LL/LH/HL/HH.png` | Four DWT frequency subbands visualised |
| `01_dwt_subbands.npy` | Raw DWT coefficients [4, 256, 256] |
| `03_cal_attention_map.png` | CAL attention heatmap (JET colormap) |
| `05_compressed.bin` | Compressed bitstream |
| `05_compression_stats.txt` | bpp, ratio, sizes |
| `06_coarse_reconstruction.png` | After inverse DWT (before DNN refine) |
| `08_dnn_pixel_refined_3x3.png` | After 3×3 DNN pixel estimation |
| `09_spatial_filtered.png` | **Final output** — after spatial filter |
| `10_comparison.png` | 3-panel: Original / Coarse / Final + diff map |
| `10_metrics.json` | PSNR, SSIM, MSE, NCC, bpp, time |
| `10_report.txt` | Human-readable summary |

---

## Adjusting hyperparameters

Edit `medical_compression/config/config.py` — all parameters are documented there.

Key settings:
```python
CFG.vae.latent_dim          = 256     # latent vector size
CFG.cal.num_heads           = 8       # attention heads
CFG.cal.embed_dim           = 64      # attention embedding
CFG.dnn_pixel.default_window = "3x3" # "3x3" | "1x8" | "both"
CFG.training.learning_rate  = 0.0002
CFG.training.num_epochs     = 150
```

---

## Expected performance (from paper)

| Metric | Proposed | JPEG2000 | BPG |
|--------|----------|----------|-----|
| PSNR   | 40.43 dB | ~37 dB   | ~38 dB |
| SSIM   | 0.9715   | ~0.95    | ~0.96 |
| MSE    | 0.613    | ~0.81    | ~0.77 |
| NCC    | 0.9975   | ~0.994   | ~0.996 |
| bpp    | 2.91     | 3.1      | 2.85 |

---

## Reference

Fan Dai, *"Deep learning based medical image compression using cross attention
learning and wavelet transform"*, Scientific Reports (2025) 15:40008.  
https://doi.org/10.1038/s41598-025-23582-y
