"""
Stage 2 — Discrete Wavelet Transform (DWT) decomposition.

Paper equation (1):  {LL, LH, HL, HH} = DWT(I)

  • LL  — low-frequency approximation  (kept intact, structural content)
  • LH  — horizontal high-frequency    (edges)
  • HL  — vertical high-frequency      (edges)
  • HH  — diagonal high-frequency      (texture detail)

Outputs are saved as:
  result_dir/01_dwt_LL.png
  result_dir/01_dwt_LH.png
  result_dir/01_dwt_HL.png
  result_dir/01_dwt_HH.png
  result_dir/01_dwt_subbands.npy   ← stacked float32 array fed to next stage
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
import pywt

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DWT
# ─────────────────────────────────────────────────────────────────────────────

def decompose(arr: np.ndarray,
              wavelet: str = None,
              level: int = None) -> Dict[str, np.ndarray]:
    """
    Apply single-level 2D DWT.

    Args:
        arr     : float32 [H, W]  normalised to [0, 255]
        wavelet : wavelet name (default from config: "haar")
        level   : decomposition level (default 1)

    Returns:
        dict with keys "LL", "LH", "HL", "HH" — all float32 arrays
    """
    if wavelet is None:
        wavelet = CFG.wavelet.wavelet
    if level is None:
        level = CFG.wavelet.level

    # pywt wavedec2 returns:  [cA, (cH, cV, cD), ...]
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)

    # Level-1 decomposition
    cA = coeffs[0]                     # LL approximation
    cH, cV, cD = coeffs[1]            # LH, HL, HH detail subbands

    subbands = {
        "LL": cA.astype(np.float32),
        "LH": cH.astype(np.float32),
        "HL": cV.astype(np.float32),
        "HH": cD.astype(np.float32),
    }

    # Removed per-image logging to reduce verbosity
    # log.info(f"[Stage 2] DWT done — subband shapes: "
    #          f"LL={cA.shape} LH={cH.shape} HL={cV.shape} HH={cD.shape}")
    return subbands


def stack_subbands(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack {LL, LH, HL, HH} into a single [4, H/2, W/2] float32 tensor
    ready for the encoder.
    """
    return np.stack([
        subbands["LL"],
        subbands["LH"],
        subbands["HL"],
        subbands["HH"],
    ], axis=0)   # → [4, H, W]


def _normalise_for_save(arr: np.ndarray) -> np.ndarray:
    """Normalise arbitrary float array to uint8 [0, 255] for PNG save."""
    a = arr - arr.min()
    if a.max() > 0:
        a = a / a.max() * 255.0
    return a.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(arr: np.ndarray,
        result_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Run DWT on a single image.

    Args:
        arr        : float32 [H, W] from Stage 1
        result_dir : output folder for this image

    Returns:
        subbands   : dict {"LL", "LH", "HL", "HH"}  — raw float32 coefficients
        stacked    : np.ndarray [4, H/2, W/2]        — model input
    """
    subbands = decompose(arr)
    stacked = stack_subbands(subbands)

    # ── Save subband images ──────────────────────────────────────────────────
    for name, sb in subbands.items():
        out_path = result_dir / f"01_dwt_{name}.png"
        cv2.imwrite(str(out_path), _normalise_for_save(sb))

    # ── Save raw coefficients for reconstruction ─────────────────────────────
    npy_path = result_dir / "01_dwt_subbands.npy"
    np.save(str(npy_path), stacked)

    log.info(f"[Stage 2] Saved DWT subbands → {result_dir}/01_dwt_*.png")
    return subbands, stacked


# ─────────────────────────────────────────────────────────────────────────────
# Inverse DWT (used in reconstruction path — Stage 9)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(subbands: Dict[str, np.ndarray],
                wavelet: str = None) -> np.ndarray:
    """
    Inverse DWT: reconstruct full-resolution image from {LL, LH, HL, HH}.

    Returns float32 [H, W] in approximately [0, 255].
    """
    if wavelet is None:
        wavelet = CFG.wavelet.wavelet

    cA  = subbands["LL"]
    cH  = subbands["LH"]
    cV  = subbands["HL"]
    cD  = subbands["HH"]

    coeffs = [cA, (cH, cV, cD)]
    reconstructed = pywt.waverec2(coeffs, wavelet=wavelet)
    return reconstructed.astype(np.float32)
