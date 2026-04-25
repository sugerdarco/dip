"""
Main inference pipeline — runs all 9 stages for every image found in
working_dir/contents/ and saves all outputs to:

    working_dir/output/<image_title>/results_of_<image_title>/

Stages executed per image:
    1. Load + preprocess          → 00_input_normalised.png
    2. DWT decomposition          → 01_dwt_LL/LH/HL/HH.png + .npy
    3. CAL encoder                → 03_cal_attention_map.png + features.npy
    4. VAE encoder                → 04_vae_z/mu/logvar.npy
    5. Entropy coding             → 05_compressed.bin + meta.json + stats.txt
    6+7. VAE decoder              → 07_decoded_subbands.npy + vis.png
    8. DNN pixel estimation       → 08_dnn_pixel_refined_*.png
    9. Inverse DWT                → (internal, feeds Stage 8)
   10. Trainable spatial filter   → 09_spatial_filtered.png + kernel.npy
   11. Metrics + report           → 10_metrics.json + 10_report.txt
   12. Side-by-side comparison    → 10_comparison.png

Usage:
    # Train first (optional — uses random weights if no checkpoint)
    python train.py --contents working_dir/contents --epochs 150

    # Run inference on all images
    python run_pipeline.py

    # Run on specific image
    python run_pipeline.py --image working_dir/contents/mri/brain_001.dcm

    # Run with specific window mode for DNN pixel estimation
    python run_pipeline.py --window both
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config.config import CFG
from stages import (
    stage1_loader,
    stage2_dwt,
    stage3_cal_encoder,
    stage4_vae_encoder,
    stage5_entropy,
    stage6_vae_decoder,
    stage7_dnn_pixel,
    stage8_spatial_filter,
)
from utils.metrics import compute_all
from train import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison image builder
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison(original: np.ndarray,
                    compressed: np.ndarray,
                    reconstructed: np.ndarray,
                    metrics: dict,
                    result_dir: Path):
    """
    Build a 3-panel side-by-side PNG:
        [Original] | [Compressed / coarse] | [Reconstructed (final)]
    Annotated with PSNR, SSIM, MSE, NCC.
    """
    H, W = original.shape

    def to_bgr(arr):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    orig_bgr  = to_bgr(original)
    comp_bgr  = to_bgr(cv2.resize(compressed.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_LINEAR))
    recon_bgr = to_bgr(cv2.resize(reconstructed.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_LINEAR))

    # Labels
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_sc   = max(0.4, W / 1200)
    thickness = 1
    pad       = 30

    def label(img, text, sub=""):
        out = img.copy()
        cv2.putText(out, text, (8, pad - 8), font, font_sc,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        if sub:
            cv2.putText(out, sub, (8, pad + 12), font, font_sc * 0.8,
                        (200, 200, 200), thickness, cv2.LINE_AA)
        return out

    orig_bgr  = label(orig_bgr,  "Original")
    comp_bgr  = label(comp_bgr,  "Compressed (coarse)")
    recon_bgr = label(recon_bgr, "Reconstructed (final)",
                      f"PSNR={metrics['psnr']:.2f}dB  "
                      f"SSIM={metrics['ssim']:.4f}  "
                      f"MSE={metrics['mse']:.3f}  "
                      f"NCC={metrics['ncc']:.4f}")

    panel = np.hstack([orig_bgr, comp_bgr, recon_bgr])

    # Difference map (original vs final, enhanced ×5 for visibility)
    diff = np.abs(original.astype(np.float32) -
                  cv2.resize(reconstructed.astype(np.float32),
                             (W, H), interpolation=cv2.INTER_LINEAR))
    diff_norm = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    diff_color = label(diff_color, "Difference map (×5)")
    diff_panel = np.hstack([
        diff_color,
        np.zeros_like(diff_color),
        np.zeros_like(diff_color),
    ])

    full = np.vstack([panel, diff_panel])
    out_path = result_dir / "10_comparison.png"
    cv2.imwrite(str(out_path), full)
    log.info(f"  Saved comparison → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_image(record: dict,
              models: dict,
              device: str,
              window: str = "3x3") -> dict:
    """
    Run the full 10-stage pipeline on a single image record.

    Args:
        record : dict from stage1_loader.scan_contents()
        models : {"enc", "venc", "vdec", "sflt", "dnn3x3", "dnn1x8"}
        device : "cuda" or "cpu"
        window : "3x3" | "1x8" | "both"

    Returns:
        result_summary dict (metrics + paths)
    """
    t_start = time.time()
    image_path = record["path"]
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {image_path}")

    # ── Stage 1: Load + preprocess ───────────────────────────────────────────
    arr, result_dir = stage1_loader.run(record)
    H, W = arr.shape
    log.info(f"  Stage 1 ✓  shape={arr.shape}  → {result_dir}")

    # ── Stage 2: DWT decomposition ───────────────────────────────────────────
    subbands, stacked = stage2_dwt.run(arr, result_dir)
    log.info(f"  Stage 2 ✓  DWT subbands stacked: {stacked.shape}")

    # ── Stage 3: CAL encoder ─────────────────────────────────────────────────
    features, attn_map, enc = stage3_cal_encoder.run(
        stacked, result_dir,
        model=models.get("enc"),
        device=device,
    )
    models["enc"] = enc
    log.info(f"  Stage 3 ✓  features: {features.shape}")

    # ── Stage 4: VAE encoder ─────────────────────────────────────────────────
    z, mu, log_var, venc = stage4_vae_encoder.run(
        features, result_dir,
        model=models.get("venc"),
        device=device,
    )
    models["venc"] = venc
    log.info(f"  Stage 4 ✓  z: {z.shape}")

    # ── Stage 5: Entropy coding ──────────────────────────────────────────────
    meta = stage5_entropy.run(z, result_dir, original_shape=(H, W))
    log.info(f"  Stage 5 ✓  {meta['bpp']:.3f} bpp  "
             f"ratio={meta['compression_ratio']:.2f}×")

    # ── Stage 5→6: Entropy decode ────────────────────────────────────────────
    z_dec = stage5_entropy.decompress(result_dir, device=device)

    # ── Stage 6+7: VAE decoder ───────────────────────────────────────────────
    rec_subbands, vdec = stage6_vae_decoder.run(
        z_dec, result_dir,
        original_shape=(H, W),
        model=models.get("vdec"),
        device=device,
    )
    models["vdec"] = vdec
    log.info(f"  Stage 6 ✓  decoded subbands reconstructed")

    # ── Stage 9 (intermediate): Inverse DWT → coarse image ───────────────────
    coarse_arr = stage2_dwt.reconstruct(rec_subbands)
    coarse_arr = np.clip(coarse_arr, 0, 255).astype(np.float32)
    # Resize to original size if needed
    if coarse_arr.shape != (H, W):
        coarse_arr = cv2.resize(coarse_arr, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(result_dir / "06_coarse_reconstruction.png"),
                coarse_arr.astype(np.uint8))
    log.info(f"  Stage 7 ✓  coarse reconstruction: {coarse_arr.shape}")

    # ── Stage 8: DNN pixel estimation ────────────────────────────────────────
    refined_img, dnn3x3, dnn1x8 = stage7_dnn_pixel.run(
        coarse_arr, result_dir,
        model_3x3=models.get("dnn3x3"),
        model_1x8=models.get("dnn1x8"),
        device=device,
        window=window,
    )
    models["dnn3x3"] = dnn3x3
    models["dnn1x8"] = dnn1x8
    log.info(f"  Stage 8 ✓  DNN pixel refined ({window} window)")

    # ── Stage 9 (final): Trainable spatial filter ────────────────────────────
    final_img, sflt = stage8_spatial_filter.run(
        refined_img, result_dir,
        model=models.get("sflt"),
        device=device,
    )
    models["sflt"] = sflt
    log.info(f"  Stage 9 ✓  Spatial filter applied")

    # ── Stage 10: Metrics ────────────────────────────────────────────────────
    metrics = compute_all(arr, final_img)
    metrics["bpp"]               = meta["bpp"]
    metrics["compression_ratio"] = meta["compression_ratio"]
    metrics["time_ms"]           = round((time.time() - t_start) * 1000, 1)

    # Save metrics JSON
    metrics_path = result_dir / "10_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save human-readable report
    report = (
        f"{'='*50}\n"
        f"Image        : {image_path}\n"
        f"Modality     : {record['modality']}\n"
        f"Output dir   : {result_dir}\n"
        f"{'─'*50}\n"
        f"MSE          : {metrics['mse']:.6f}\n"
        f"PSNR         : {metrics['psnr']:.4f} dB\n"
        f"SSIM         : {metrics['ssim']:.6f}\n"
        f"NCC          : {metrics['ncc']:.6f}\n"
        f"Bit rate     : {metrics['bpp']:.4f} bpp\n"
        f"Comp. ratio  : {metrics['compression_ratio']:.2f}×\n"
        f"Total time   : {metrics['time_ms']:.1f} ms\n"
        f"{'='*50}\n"
    )
    (result_dir / "10_report.txt").write_text(report)
    log.info(report)

    # ── Stage 10: Comparison image ───────────────────────────────────────────
    make_comparison(arr, coarse_arr, final_img, metrics, result_dir)

    return {
        "image":      image_path,
        "result_dir": str(result_dir),
        "metrics":    metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(contents_dir: str = None,
            output_dir: str = None,
            single_image: str = None,
            window: str = "3x3",
            device_str: str = None):
    """
    Run the full pipeline on all images in contents_dir,
    or on a single image if --image is specified.
    """
    device = device_str or (CFG.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if "cuda" in device and torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Override config paths if provided
    if contents_dir:
        CFG.paths.contents_dir = contents_dir
    if output_dir:
        CFG.paths.output_dir = output_dir

    # Load trained models (or initialise fresh if no checkpoint)
    enc, venc, vdec, sflt = load_checkpoint(device)
    models = {
        "enc":    enc,
        "venc":   venc,
        "vdec":   vdec,
        "sflt":   sflt,
        "dnn3x3": None,   # built on first use
        "dnn1x8": None,
    }

    # Collect image records
    if single_image:
        stem = Path(single_image).stem
        ext  = Path(single_image).suffix.lower()
        records = [{
            "path":     single_image,
            "stem":     stem,
            "modality": "CT_or_MRI" if ext == ".dcm" else "standard",
            "rel_dir":  ".",
        }]
    else:
        records = stage1_loader.scan_contents(CFG.paths.contents_dir)

    if not records:
        log.error("No images found. Check --contents path.")
        return

    log.info(f"\nProcessing {len(records)} image(s)  |  window={window}\n")

    all_results = []
    failed      = []

    for i, record in enumerate(records, 1):
        log.info(f"[{i}/{len(records)}] {record['stem']}")
        try:
            result = run_image(record, models, device, window=window)
            all_results.append(result)
        except Exception as e:
            log.error(f"  FAILED: {record['path']} — {e}", exc_info=True)
            failed.append({"image": record["path"], "error": str(e)})

    # ── Global summary ────────────────────────────────────────────────────────
    if all_results:
        avg_psnr = np.mean([r["metrics"]["psnr"] for r in all_results])
        avg_ssim = np.mean([r["metrics"]["ssim"] for r in all_results])
        avg_mse  = np.mean([r["metrics"]["mse"]  for r in all_results])
        avg_ncc  = np.mean([r["metrics"]["ncc"]  for r in all_results])
        avg_bpp  = np.mean([r["metrics"]["bpp"]  for r in all_results])

        summary = {
            "total_images":   len(records),
            "successful":     len(all_results),
            "failed":         len(failed),
            "average_metrics": {
                "psnr": round(avg_psnr, 4),
                "ssim": round(avg_ssim, 6),
                "mse":  round(avg_mse,  6),
                "ncc":  round(avg_ncc,  6),
                "bpp":  round(avg_bpp,  4),
            },
            "failed_images": failed,
        }

        summary_path = Path(CFG.paths.output_dir) / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        log.info(f"\n{'='*60}")
        log.info(f"PIPELINE COMPLETE — {len(all_results)}/{len(records)} images")
        log.info(f"  Avg PSNR : {avg_psnr:.4f} dB")
        log.info(f"  Avg SSIM : {avg_ssim:.6f}")
        log.info(f"  Avg MSE  : {avg_mse:.6f}")
        log.info(f"  Avg NCC  : {avg_ncc:.6f}")
        log.info(f"  Avg bpp  : {avg_bpp:.4f}")
        log.info(f"  Summary  : {summary_path}")
        log.info(f"{'='*60}\n")

        if failed:
            log.warning(f"  {len(failed)} image(s) failed — see summary JSON.")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical image compression pipeline (DWT + CAL + VAE)"
    )
    parser.add_argument(
        "--contents", default=CFG.paths.contents_dir,
        help="Path to folder containing MRI/CT images"
    )
    parser.add_argument(
        "--output", default=CFG.paths.output_dir,
        help="Root output directory"
    )
    parser.add_argument(
        "--image", default=None,
        help="Run on a single image file instead of the whole contents folder"
    )
    parser.add_argument(
        "--window", default="3x3", choices=["3x3", "1x8", "both"],
        help="DNN pixel estimation window mode"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu"
    )
    args = parser.parse_args()

    run_all(
        contents_dir  = args.contents,
        output_dir    = args.output,
        single_image  = args.image,
        window        = args.window,
        device_str    = args.device,
    )
