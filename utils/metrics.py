"""
Metrics — PSNR, SSIM, MSE, NCC (Normalised Correlation Coefficient).

All equations match the paper §Evaluation criteria exactly.

    MSE  [Eq. 6]   = (1/mn) Σ(I - Î)²
    PSNR [Eq. 7]   = 20·log10(255 / √MSE)
    SSIM [Eq. 5]   = (2μxμy + C1)(2σxy + C2) / ((μx²+μy²+C1)(σx²+σy²+C2))
    NCC  [Eq. 8]   = ΣW·Ŵ / (√ΣW² · √ΣŴ²)

All functions accept float32 numpy arrays in [0, 255].
"""

import math
import numpy as np
from scipy.ndimage import uniform_filter


# ─────────────────────────────────────────────────────────────────────────────
# MSE  [Eq. 6]
# ─────────────────────────────────────────────────────────────────────────────

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error — lower is better."""
    assert original.shape == reconstructed.shape, \
        f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# PSNR  [Eq. 7]
# ─────────────────────────────────────────────────────────────────────────────

def psnr(original: np.ndarray, reconstructed: np.ndarray,
         max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB — higher is better.
    Returns inf if images are identical.
    """
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(err))


# ─────────────────────────────────────────────────────────────────────────────
# SSIM  [Eq. 5]
# ─────────────────────────────────────────────────────────────────────────────

def ssim(original: np.ndarray, reconstructed: np.ndarray,
         window_size: int = 11,
         C1: float = (0.01 * 255) ** 2,
         C2: float = (0.03 * 255) ** 2) -> float:
    """
    Structural Similarity Index Measure — closer to 1 is better.
    """
    x = original.astype(np.float64)
    y = reconstructed.astype(np.float64)

    mu_x  = uniform_filter(x, window_size)
    mu_y  = uniform_filter(y, window_size)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = uniform_filter(x * x, window_size) - mu_x2
    sigma_y2 = uniform_filter(y * y, window_size) - mu_y2
    sigma_xy = uniform_filter(x * y, window_size) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(ssim_map.mean())


# ─────────────────────────────────────────────────────────────────────────────
# NCC  [Eq. 8]
# ─────────────────────────────────────────────────────────────────────────────

def ncc(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Normalised Correlation Coefficient — closer to 1 is better.
    """
    W  = original.astype(np.float64).flatten()
    Wh = reconstructed.astype(np.float64).flatten()

    num = np.sum(W * Wh)
    den = math.sqrt(np.sum(W ** 2)) * math.sqrt(np.sum(Wh ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


# ─────────────────────────────────────────────────────────────────────────────
# All-in-one
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(original: np.ndarray,
                reconstructed: np.ndarray) -> dict:
    """
    Compute MSE, PSNR, SSIM, NCC in one call.

    Args:
        original      : float32 [H, W] in [0, 255]
        reconstructed : float32 [H, W] in [0, 255]

    Returns:
        dict with keys: mse, psnr, ssim, ncc
    """
    reconstructed_clipped = np.clip(reconstructed, 0, 255)
    return {
        "mse":  round(mse(original, reconstructed_clipped), 6),
        "psnr": round(psnr(original, reconstructed_clipped), 4),
        "ssim": round(ssim(original, reconstructed_clipped), 6),
        "ncc":  round(ncc(original, reconstructed_clipped), 6),
    }
