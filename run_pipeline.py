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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
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

# Silence noisy intermediate logs from individual stages
logging.getLogger("stages").setLevel(logging.WARNING)


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


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_image(record: dict,
              models: dict,
              device: str,
              window: str = "3x3",
              fast_io: bool = True,
              real_output_root: str = None) -> dict:
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
    stage_times = {}

    # ── Stage 1: Load + preprocess ───────────────────────────────────────────
    t1 = time.time()
    
    ram_dir = None
    # Optional FAST IO RAM Disk Redirection
    if fast_io and real_output_root:
        import uuid
        from pathlib import Path
        import shutil
        original_output_dir = CFG.paths.output_dir
        # Create a unique isolated RAM-based working directory
        ram_dir = Path(f"/dev/shm/dip_{uuid.uuid4().hex[:8]}")
        CFG.paths.output_dir = str(ram_dir)
    
    arr, result_dir = stage1_loader.run(record)
    
    if fast_io and real_output_root:
        # Revert CFG so other functions continue to work correctly if they rely on it
        CFG.paths.output_dir = original_output_dir

    H, W = arr.shape
    stage_times['S1_load'] = time.time() - t1

    # ── Stage 2: DWT decomposition ───────────────────────────────────────────
    t2 = time.time()
    subbands, stacked = stage2_dwt.run(arr, result_dir)
    stage_times['S2_dwt'] = time.time() - t2

    # ── Stage 3: CAL encoder ─────────────────────────────────────────────────
    t3 = time.time()
    features, attn_map, enc = stage3_cal_encoder.run(
        stacked, result_dir,
        model=models.get("enc"),
        device=device,
    )
    models["enc"] = enc
    stage_times['S3_cal_encoder'] = time.time() - t3

    # ── Stage 4: VAE encoder ─────────────────────────────────────────────────
    t4 = time.time()
    z, mu, log_var, venc = stage4_vae_encoder.run(
        features, result_dir,
        model=models.get("venc"),
        device=device,
    )
    models["venc"] = venc
    stage_times['S4_vae_encoder'] = time.time() - t4

    # ── Stage 5: Entropy coding ──────────────────────────────────────────────
    t5 = time.time()
    meta = stage5_entropy.run(z, result_dir, original_shape=(H, W))
    stage_times['S5_entropy_encode'] = time.time() - t5

    # ── Stage 5→6: Entropy decode ────────────────────────────────────────────
    t5b = time.time()
    z_dec = stage5_entropy.decompress(result_dir, device=device)
    stage_times['S5_entropy_decode'] = time.time() - t5b

    # ── Stage 6+7: VAE decoder ───────────────────────────────────────────────
    t6 = time.time()
    rec_subbands, vdec = stage6_vae_decoder.run(
        z_dec, result_dir,
        original_shape=(H, W),
        model=models.get("vdec"),
        device=device,
    )
    models["vdec"] = vdec
    stage_times['S6_vae_decoder'] = time.time() - t6

    # ── Stage 9 (intermediate): Inverse DWT → coarse image ───────────────────
    t7 = time.time()
    coarse_arr = stage2_dwt.reconstruct(rec_subbands)
    coarse_arr = np.clip(coarse_arr, 0, 255).astype(np.float32)
    # Resize to original size if needed
    if coarse_arr.shape != (H, W):
        coarse_arr = cv2.resize(coarse_arr, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(result_dir / "06_coarse_reconstruction.png"),
                coarse_arr.astype(np.uint8))
    stage_times['S7_inverse_dwt'] = time.time() - t7

    # ── Stage 8: DNN pixel estimation ────────────────────────────────────────
    t8 = time.time()
    refined_img, dnn3x3, dnn1x8 = stage7_dnn_pixel.run(
        coarse_arr, result_dir,
        model_3x3=models.get("dnn3x3"),
        model_1x8=models.get("dnn1x8"),
        device=device,
        window=window,
    )
    models["dnn3x3"] = dnn3x3
    models["dnn1x8"] = dnn1x8
    stage_times['S8_dnn_pixel'] = time.time() - t8

    # ── Stage 9 (final): Trainable spatial filter ────────────────────────────
    t9 = time.time()
    final_img, sflt = stage8_spatial_filter.run(
        refined_img, result_dir,
        model=models.get("sflt"),
        device=device,
    )
    models["sflt"] = sflt
    stage_times['S9_spatial_filter'] = time.time() - t9

    # ── Stage 10: Metrics ────────────────────────────────────────────────────
    t10 = time.time()
    metrics = compute_all(arr, final_img)
    metrics["bpp"]               = meta["bpp"]
    metrics["compression_ratio"] = meta["compression_ratio"]
    metrics["time_ms"]           = round((time.time() - t_start) * 1000, 1)
    stage_times['S10_metrics'] = time.time() - t10

    # Save metrics JSON
    metrics_path = result_dir / "10_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save human-readable report
    t_report = time.time()
    
    # Build stage timing breakdown
    timing_str = f"{'─'*50}\nStage Timing Breakdown:\n"
    gpu_time = (stage_times.get('S3_cal_encoder', 0) + 
                stage_times.get('S4_vae_encoder', 0) + 
                stage_times.get('S6_vae_decoder', 0) + 
                stage_times.get('S8_dnn_pixel', 0) + 
                stage_times.get('S9_spatial_filter', 0))
    cpu_time = (stage_times.get('S2_dwt', 0) + 
                stage_times.get('S7_inverse_dwt', 0))
    
    for stage, secs in sorted(stage_times.items()):
        pct = (secs / metrics['time_ms'] * 1000) * 100 if metrics['time_ms'] > 0 else 0
        timing_str += f"  {stage:20s}: {secs:7.3f}s ({pct:5.1f}%)\n"
    
    timing_str += f"  {'GPU_total':20s}: {gpu_time:7.3f}s ({(gpu_time/metrics['time_ms']*1000)*100:5.1f}%)\n"
    timing_str += f"  {'CPU_total':20s}: {cpu_time:7.3f}s ({(cpu_time/metrics['time_ms']*1000)*100:5.1f}%)\n"
    
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
        f"{timing_str}"
        f"{'='*50}\n"
    )
    (result_dir / "10_report.txt").write_text(report)

    # ── Stage 11: Comparison image ───────────────────────────────────────────
    t_comp = time.time()
    make_comparison(arr, coarse_arr, final_img, metrics, result_dir)
    stage_times['S11_comparison'] = time.time() - t_comp

    # ── Fast I/O Transfer ───────────────────────────────────────────────────
    if fast_io and real_output_root:
        # We process in RAM, now move strictly only the final outputs to permanent storage
        import shutil
        # deduce the target real directory matching what stage1_loader would have done
        rel_dir = record.get("rel_dir", "")
        image_title = record["stem"]
        perm_dir = Path(real_output_root) / rel_dir / image_title / f"results_of_{image_title}"
        perm_dir.mkdir(parents=True, exist_ok=True)
        
        # We save ONLY the required inference endpoints, dropping 100MB+ of intermediates
        keep_files = ["10_metrics.json", "10_report.txt", "10_comparison.png", "05_compressed.bin"]
        for f in keep_files:
            src = result_dir / f
            if src.exists():
                shutil.copy2(str(src), str(perm_dir / f))
                
        # Clean up the exact RAM disk root for this image to prevent memory leaks
        if ram_dir and ram_dir.exists():
            shutil.rmtree(ram_dir, ignore_errors=True)
        result_dir = perm_dir

    return {
        "image":      image_path,
        "result_dir": str(result_dir),
        "metrics":    metrics,
        "stage_times": stage_times,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker Process Initializer
# ─────────────────────────────────────────────────────────────────────────────
_worker_models = None

def init_worker(device_str):
    """Initializes PyTorch models once per worker process."""
    global _worker_models
    from train import load_checkpoint
    
    enc, venc, vdec, sflt = load_checkpoint(device_str)
    
    if enc: enc.eval()
    if venc: venc.eval()
    if vdec: vdec.eval()
    if sflt: sflt.eval()
    
    _worker_models = {
        "enc":    enc,
        "venc":   venc,
        "vdec":   vdec,
        "sflt":   sflt,
        "dnn3x3": None,   # built on first use
        "dnn1x8": None,
    }

def run_image_wrapper(args):
    """Wrapper to unpack arguments and call run_image with global _worker_models."""
    record, device, window, fast_io, real_output_root = args
    global _worker_models
    return run_image(record, _worker_models, device, window, fast_io, real_output_root)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(contents_dir: str = None,
            output_dir: str = None,
            single_image: str = None,
            window: str = "3x3",
            device_str: str = None,
            batch_size: int = 64,
            fast_io: bool = True,
            max_images: int = 0):
    """
    Run the full pipeline on all images in contents_dir,
    or on a single image if --image is specified.
    """
    device = device_str or (CFG.device if torch.cuda.is_available() else "cpu")

    # Override config paths if provided
    if contents_dir:
        CFG.paths.contents_dir = contents_dir
    if output_dir:
        CFG.paths.output_dir = output_dir

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

    if max_images > 0 and len(records) > max_images:
        log.info(f"Limiting dataset to {max_images} images from {len(records)} total")
        records = records[:max_images]

    log.info(f"Processing {len(records)} images in parallel...\n")

    all_results = []
    failed      = []

    # Use ProcessPoolExecutor for true multi-core CPU and GPU processing
    # The 'batch_size' parameter dynamically translates to the number of parallel workers
    # processing independent images at the same time. Since inference runs sequentially
    # per image, scaling this parameter effectively saturates GPU and RAM up to its limit.
    max_workers = min(batch_size, len(records))
    ctx = mp.get_context('spawn')
    
    if fast_io:
        log.info("FAST I/O is ENABLED. Intermediate files will be discarded to save Disk I/O bottlenecks.")
    
    with ProcessPoolExecutor(max_workers=max_workers, 
                             mp_context=ctx,
                             initializer=init_worker, 
                             initargs=(device,)) as executor:
        futures = {
            executor.submit(run_image_wrapper, (record, device, window, fast_io, CFG.paths.output_dir)): (i, record)
            for i, record in enumerate(records, 1)
        }
        
        completed_count = 0
        for future in as_completed(futures):
            idx, record = futures[future]
            completed_count += 1
            try:
                result = future.result()
                all_results.append(result)
                
                # Clean progress tracking instead of per-image spam
                if completed_count % 100 == 0 or completed_count == len(records):
                    pct = (completed_count / len(records)) * 100
                    log.info(f"Progress: {completed_count}/{len(records)} images complete ({pct:.1f}%)")
            except Exception as e:
                log.error(f"✗ [{idx}/{len(records)}] {record['stem']} — {e}")
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

        # ── Timing Analysis ──────────────────────────────────────────────────
        if all_results and "stage_times" in all_results[0]:
            stage_names = all_results[0]["stage_times"].keys()
            avg_times = {}
            for stage in stage_names:
                times = [r["stage_times"][stage] for r in all_results if "stage_times" in r]
                avg_times[stage] = np.mean(times) if times else 0
            
            gpu_stages = ['S3_cal_encoder', 'S4_vae_encoder', 'S6_vae_decoder', 'S8_dnn_pixel', 'S9_spatial_filter']
            cpu_stages = ['S2_dwt', 'S7_inverse_dwt']
            
            total_gpu = sum(avg_times.get(s, 0) for s in gpu_stages)
            total_cpu = sum(avg_times.get(s, 0) for s in cpu_stages)
            total = total_gpu + total_cpu + sum(avg_times.values())
            
            timing_line = f"Timing: GPU {total_gpu:.2f}s ({total_gpu/total*100:.0f}%) | CPU {total_cpu:.2f}s ({total_cpu/total*100:.0f}%)"
        else:
            timing_line = ""

        log.info(f"\n✓ DONE — {len(all_results)}/{len(records)} images")
        log.info(f"  Metrics: PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f} | MSE={avg_mse:.4f} | bpp={avg_bpp:.3f}")
        if timing_line:
            log.info(f"  {timing_line}")
        log.info(f"  Summary: {summary_path}\n")

        if failed:
            log.warning(f"✗ {len(failed)} failed")

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
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Number of images to process in parallel (increases memory and GPU utilization)"
    )
    parser.add_argument(
        "--disable-fast-io", action="store_true",
        help="Disable RAM disk fast I/O and instead write all intermediate stages to original disk"
    )
    parser.add_argument(
        "--max-images", type=int, default=100000,
        help="Limit the total number of images processed (runs much faster for validation)"
    )
    args = parser.parse_args()

    run_all(
        contents_dir  = args.contents,
        output_dir    = args.output,
        single_image  = args.image,
        window        = args.window,
        device_str    = args.device,
        batch_size    = args.batch_size,
        fast_io       = not args.disable_fast_io,
        max_images    = args.max_images,
    )
