"""
Training — end-to-end training of all learnable modules.

Trains:  DWTEncoder (Stage 3)  +  VAEEncoder (Stage 4)  +
         VAEDecoder (Stage 6)  +  TrainableSpatialFilter (Stage 9)
         in one joint optimisation.

Paper §Simulation results:
    Adam  lr=0.0002  β1=0.9  β2=0.999
    batch=32  epochs=150  dropout=0.3
    Early stopping: patience=10
    Init: Xavier Normal
    Loss: λ1·MSE + λ2·Perceptual + λ3·KL-divergence  [Eq. 2]

Usage:
    python -m medical_compression.train \
        --contents working_dir/contents \
        --output   working_dir/output \
        --epochs   150
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

# Enable cuDNN benchmark for faster convolution
torch.backends.cudnn.benchmark = True

from config.config import CFG
from stages.stage1_loader import scan_contents, load_image, preprocess
from stages.stage2_dwt import decompose, stack_subbands
from stages.stage3_cal_encoder import DWTEncoder
from stages.stage4_vae_encoder import VAEEncoder, reparameterise, kl_loss
from stages.stage6_vae_decoder import VAEDecoder
from stages.stage7_dnn_pixel import PixelDNN
from stages.stage8_spatial_filter import TrainableSpatialFilter, combined_loss
from utils.metrics import compute_all

logging.basicConfig(
    level=logging.INFO,  # Keep INFO level for important training logs
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MedicalImageDataset(Dataset):
    """
    Loads images from a list of records, applies DWT, returns stacked subbands
    and original image as target.
    """

    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        try:
            arr = load_image(rec["path"])
            arr = preprocess(arr)
        except Exception as e:
            log.warning(f"Failed to load {rec['path']}: {e} — skipping.")
            arr = np.zeros(CFG.image.target_size, dtype=np.float32)

        subbands = decompose(arr)
        stacked  = stack_subbands(subbands)          # [4, H/2, W/2]

        # Normalise to [0, 1]
        x = torch.from_numpy(stacked / 255.0).float()
        y = torch.from_numpy(arr / 255.0).float().unsqueeze(0)  # [1, H, W]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual loss (VGG-style — simplified as Laplacian gradient matching)
# ─────────────────────────────────────────────────────────────────────────────

def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient-domain loss as a lightweight perceptual proxy."""
    # Sobel-style gradient
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=pred.dtype, device=pred.device).view(1,1,3,3)
    ky = kx.transpose(-2, -1)

    def grad(x):
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    # pred is [B, 4, H, W]; target is [B, 1, H, W]
    # Use LL channel (index 0) from pred vs target
    pred_ll   = pred[:, 0:1, :, :]
    # Resize target to match pred spatial size if needed
    target_rs = F.interpolate(target, size=pred_ll.shape[-2:], mode="bilinear",
                              align_corners=False)
    return F.mse_loss(grad(pred_ll), grad(target_rs))


# ─────────────────────────────────────────────────────────────────────────────
# Build all models
# ─────────────────────────────────────────────────────────────────────────────

def build_models(device: str):
    enc  = DWTEncoder().to(device)
    venc = VAEEncoder().to(device)
    vdec = VAEDecoder().to(device)
    sflt = TrainableSpatialFilter().to(device)
    return enc, venc, vdec, sflt


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(contents_dir: str = None,
          output_dir: str = None,
          epochs: int = None,
          device_str: str = None,
          max_images: int = None):

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)

    device = device_str or (CFG.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if "cuda" in device:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Mixed precision scaler disabled for now
    scaler = None

    contents_dir = contents_dir or CFG.paths.contents_dir
    output_dir   = output_dir or CFG.paths.output_dir
    epochs       = epochs or CFG.training.num_epochs
    max_images   = max_images if max_images is not None else CFG.training.max_records
    ckpt_dir     = Path(CFG.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load records ──────────────────────────────────────────────────────────
    all_records = scan_contents(contents_dir)
    if not all_records:
        log.error("No images found — check contents_dir path.")
        return

    if max_images > 0 and len(all_records) > max_images:
        log.info(f"Limiting dataset to {max_images} images from {len(all_records)} total")
        all_records = all_records[:max_images]

    random.shuffle(all_records)
    n = len(all_records)
    n_val  = max(1, int(n * CFG.training.validation_split))
    n_test = max(1, int(n * CFG.training.test_split))
    n_train = n - n_val - n_test

    train_records = all_records[:n_train]
    val_records   = all_records[n_train:n_train + n_val]

    log.info(f"Dataset split — train: {len(train_records)}, "
             f"val: {len(val_records)}, test: {n_test}")

    train_ds = MedicalImageDataset(train_records)
    val_ds   = MedicalImageDataset(val_records)

    train_dl = DataLoader(train_ds, batch_size=CFG.training.batch_size,
                          shuffle=True,  num_workers=CFG.num_workers,
                          pin_memory=CFG.pin_memory, drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=CFG.training.batch_size,
                          shuffle=False, num_workers=CFG.num_workers,
                          pin_memory=CFG.pin_memory)

    n_train_batches = len(train_dl)
    stage2_progress_step = max(1, n_train_batches // 10)

    # ── Models + optimiser ────────────────────────────────────────────────────
    enc, venc, vdec, sflt = build_models(device)
    log.info(f"Parameters — Encoder: {count_params(enc):,}  "
             f"VAE-enc: {count_params(venc):,}  "
             f"VAE-dec: {count_params(vdec):,}  "
             f"Filter: {count_params(sflt):,}")

    all_params = (list(enc.parameters()) + list(venc.parameters()) +
                  list(vdec.parameters()) + list(sflt.parameters()))

    optimizer = torch.optim.Adam(
        all_params,
        lr=CFG.training.learning_rate,
        betas=(CFG.training.beta1, CFG.training.beta2),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    cfg_v  = CFG.vae
    lambda_mse = cfg_v.lambda_mse
    lambda_prc = cfg_v.lambda_perceptual
    lambda_kl  = cfg_v.lambda_kl

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        enc.train(); venc.train(); vdec.train(); sflt.train()
        train_loss_sum = 0.0

        for batch_idx, (x, y) in enumerate(train_dl):
            x = x.to(device)   # [B, 4, H/2, W/2]
            y = y.to(device)   # [B, 1, H, W]

            if (batch_idx + 1) % stage2_progress_step == 0 or batch_idx + 1 == n_train_batches:
                progress = int((batch_idx + 1) * 100 / n_train_batches)
                log.info(f"[Stage 2] DWT progress: {progress}% "
                         f"({batch_idx + 1}/{n_train_batches} batches)")

            optimizer.zero_grad()

            # Mixed precision training (temporarily disabled)
            # with torch.amp.autocast(device, enabled=scaler is not None):
            # Forward
            features, _  = enc(x)            # [B, 128, H/2, W/2]
            mu, log_var  = venc(features)     # [B, latent_dim]
            z = reparameterise(mu, log_var)   # sampled
            decoded      = vdec(z)            # [B, 4, H/2, W/2]

            # Resize decoded LL to full resolution for spatial filter
            decoded_ll = decoded[:, 0:1, :, :]   # [B, 1, H/2, W/2]
            full_res   = F.interpolate(decoded_ll, size=y.shape[-2:],
                                       mode="bilinear", align_corners=False)
            filtered   = sflt(full_res)           # [B, 1, H, W]

            # Losses
            l_mse  = F.mse_loss(filtered, y) * lambda_mse
            # l_perc = perceptual_loss(decoded, y)  * lambda_prc  # Temporarily disabled
            l_kl   = kl_loss(mu, log_var)         * lambda_kl
            loss   = l_mse + l_kl  # + l_perc

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train = train_loss_sum / max(len(train_dl), 1)

        # ── Validation ────────────────────────────────────────────────────────
        enc.eval(); venc.eval(); vdec.eval(); sflt.eval()
        val_loss_sum = 0.0
        val_metrics  = []

        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)

                with torch.amp.autocast(device, enabled=scaler is not None):
                    features, _  = enc(x)
                    mu, log_var  = venc(features)
                    z = mu                           # deterministic at eval
                    decoded      = vdec(z)
                    decoded_ll   = decoded[:, 0:1, :, :]
                    full_res     = F.interpolate(decoded_ll, size=y.shape[-2:],
                                                 mode="bilinear", align_corners=False)
                    filtered     = sflt(full_res)

                    l_mse  = F.mse_loss(filtered, y) * lambda_mse
                    l_kl   = kl_loss(mu, log_var)    * lambda_kl
                    val_loss_sum += (l_mse + l_kl).item()

                # Compute image metrics on CPU
                pred_np = (filtered[0, 0].cpu().numpy() * 255.0).astype(np.float32)
                gt_np   = (y[0, 0].cpu().numpy()        * 255.0).astype(np.float32)
                val_metrics.append(compute_all(gt_np, pred_np))

        avg_val   = val_loss_sum / max(len(val_dl), 1)
        avg_psnr  = np.mean([m["psnr"] for m in val_metrics])
        avg_ssim  = np.mean([m["ssim"] for m in val_metrics])

        scheduler.step(avg_val)

        elapsed = time.time() - t0
        log.info(f"Epoch {epoch:03d}/{epochs}  "
                 f"train={avg_train:.5f}  val={avg_val:.5f}  "
                 f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}  "
                 f"({elapsed:.1f}s)")

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_psnr"].append(float(avg_psnr))
        history["val_ssim"].append(float(avg_ssim))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_count = 0
            ckpt_path = ckpt_dir / CFG.paths.checkpoint_name
            torch.save({
                "epoch":      epoch,
                "enc":        enc.state_dict(),
                "vae_enc":    venc.state_dict(),
                "vae_dec":    vdec.state_dict(),
                "sflt":       sflt.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   avg_val,
                "val_psnr":   float(avg_psnr),
                "val_ssim":   float(avg_ssim),
            }, ckpt_path)
            log.info(f"  ✓ New best checkpoint saved ({ckpt_path})")
        else:
            patience_count += 1
            if patience_count >= CFG.training.early_stopping_patience:
                log.info(f"Early stopping at epoch {epoch} "
                         f"(no improvement for {patience_count} epochs).")
                break

    # ── Save training history ─────────────────────────────────────────────────
    hist_path = ckpt_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training complete. History saved → {hist_path}")

    return enc, venc, vdec, sflt


# ─────────────────────────────────────────────────────────────────────────────
# Load trained checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(device: str = None):
    """Load best saved checkpoint → (enc, venc, vdec, sflt) all on device."""
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(CFG.paths.checkpoint_dir) / CFG.paths.checkpoint_name
    if not ckpt_path.exists():
        log.warning(f"No checkpoint found at {ckpt_path} — using random weights.")
        return build_models(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    enc, venc, vdec, sflt = build_models(device)
    enc.load_state_dict(ckpt["enc"])
    venc.load_state_dict(ckpt["vae_enc"])
    vdec.load_state_dict(ckpt["vae_dec"])
    sflt.load_state_dict(ckpt["sflt"])

    log.info(f"Loaded checkpoint (epoch {ckpt['epoch']}, "
             f"val PSNR={ckpt.get('val_psnr', 'N/A'):.2f}dB)")
    return enc, venc, vdec, sflt


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train medical image compression pipeline")
    parser.add_argument("--contents", default=CFG.paths.contents_dir)
    parser.add_argument("--output",   default=CFG.paths.output_dir)
    parser.add_argument("--epochs",   type=int, default=CFG.training.num_epochs)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_images", type=int, default=CFG.training.max_records,
                        help="Limit the total number of images used for train/val/test; 0 means all")
    args = parser.parse_args()

    train(contents_dir=args.contents,
          output_dir=args.output,
          epochs=args.epochs,
          device_str=args.device,
          max_images=args.max_images)
