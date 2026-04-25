"""
Stage 7+8 — DNN Pixel Estimation (two window models).

Paper §Using DNN to estimate image pixels:

  Model A — 3×3 window  (paper default)
    • 8 neighbouring pixels → predict centre pixel
    • Hidden layers: [20, 18, 5, 1]
    • R = 0.997,  MSE ≈ 6.1×10⁻⁶  at epoch 333

  Model B — 1×8 linear window
    • 8 preceding pixels → predict next pixel(s)
    • Hidden layers: [20, 18, 5, 2]
    • R = 0.951,  MSE ≈ 1.4×10⁻⁶  at epoch 294
    • Enables efficient serial prediction during encoding

Both models are feedforward networks trained on local pixel context.
At inference they refine the coarse VAE reconstruction pixel-by-pixel.

Outputs:
    result_dir/08_dnn_pixel_refined_3x3.png
    result_dir/08_dnn_pixel_refined_1x8.png
    result_dir/08_dnn_pixel_refined_final.png   ← whichever is configured
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generic feedforward DNN
# ─────────────────────────────────────────────────────────────────────────────

class PixelDNN(nn.Module):
    """
    Fully connected feedforward network for pixel prediction.

    input_size  : number of context pixels (8 for both windows)
    hidden_dims : tuple of hidden layer widths
    output_size : 1 for 3×3 model, 2 for 1×8 model
    """

    def __init__(self,
                 input_size: int,
                 hidden_dims: Tuple,
                 output_size: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for h in hidden_dims[:-1]:            # all but last entry
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True),
                       nn.Dropout(CFG.training.dropout_rate)]
            prev = h
        # Final layer — no activation (raw pixel value)
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3×3 window extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_3x3_patches(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every interior pixel, extract 8 neighbours as context.

    Returns:
        X : [N, 8]   context pixels (normalised [0,1])
        y : [N, 1]   centre pixel values (normalised [0,1])
        coords : list of (row, col) for reconstruction
    """
    H, W = img.shape
    img_n = img / 255.0
    rows, cols, X_list, y_list = [], [], [], []

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            patch = img_n[r-1:r+2, c-1:c+2].flatten()   # 9 values
            context = np.delete(patch, 4)                 # remove centre → 8
            X_list.append(context)
            y_list.append([img_n[r, c]])
            rows.append(r)
            cols.append(c)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            list(zip(rows, cols)))


# ─────────────────────────────────────────────────────────────────────────────
# 1×8 window extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_1x8_patches(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every position (after the 8th pixel in scan order),
    use 8 preceding pixels to predict the next pixel.

    Returns:
        X : [N, 8]  context
        y : [N, 2]  next two pixels (model outputs 2)
        positions
    """
    flat = img.flatten().astype(np.float32) / 255.0
    N = len(flat)
    X_list, y_list, pos_list = [], [], []

    for i in range(8, N - 1):
        X_list.append(flat[i-8:i])
        y_list.append(flat[i:i+2] if i+2 <= N else flat[i:i+1])
        pos_list.append(i)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            pos_list)


# ─────────────────────────────────────────────────────────────────────────────
# Inference: refine a coarse image with DNN pixel estimation
# ─────────────────────────────────────────────────────────────────────────────

def refine_3x3(coarse: np.ndarray,
               model: PixelDNN,
               device: str) -> np.ndarray:
    """Apply 3×3 DNN to refine interior pixels of coarse image."""
    model.eval()
    refined = coarse.copy()
    H, W = coarse.shape
    img_n = coarse / 255.0

    # Build context patches for all interior pixels
    contexts = []
    coords = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            patch = img_n[r-1:r+2, c-1:c+2].flatten()
            contexts.append(np.delete(patch, 4))
            coords.append((r, c))

    if not contexts:
        return refined

    X = torch.from_numpy(np.array(contexts, dtype=np.float32)).to(device)

    batch_size = 65536
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i+batch_size]).cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()

    for (r, c), p in zip(coords, preds):
        refined[r, c] = np.clip(p * 255.0, 0, 255)

    return refined.astype(np.float32)


def refine_1x8(coarse: np.ndarray,
               model: PixelDNN,
               device: str) -> np.ndarray:
    """Apply 1×8 DNN to refine pixels in scan order."""
    model.eval()
    flat = coarse.flatten().astype(np.float32) / 255.0
    refined_flat = flat.copy()
    N = len(flat)

    batch_size = 65536
    contexts = []
    positions = []

    for i in range(8, N - 1):
        contexts.append(flat[i-8:i])
        positions.append(i)

    if not contexts:
        return coarse

    X = torch.from_numpy(np.array(contexts, dtype=np.float32)).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i+batch_size]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    for j, pos in enumerate(positions):
        pred_vals = preds[j]
        for k, p in enumerate(pred_vals):
            if pos + k < N:
                refined_flat[pos + k] = np.clip(p, 0, 1)

    H, W = coarse.shape
    return (refined_flat.reshape(H, W) * 255.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(coarse_img: np.ndarray,
        result_dir: Path,
        model_3x3: PixelDNN = None,
        model_1x8: PixelDNN = None,
        device: str = None,
        window: str = None) -> Tuple[np.ndarray, PixelDNN, PixelDNN]:
    """
    Refine coarse reconstructed image using DNN pixel estimation.

    Args:
        coarse_img  : float32 [H, W] coarse reconstruction from inverse DWT
        result_dir  : output directory
        model_3x3   : pre-built 3×3 PixelDNN
        model_1x8   : pre-built 1×8 PixelDNN
        device      : "cuda" or "cpu"
        window      : "3x3" | "1x8" | "both"  (default from config)

    Returns:
        refined     : float32 [H, W] — best refined image
        model_3x3   : (for reuse)
        model_1x8   : (for reuse)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"
    if window is None:
        window = CFG.dnn_pixel.default_window

    # Build models if not provided
    if model_3x3 is None:
        cfg_p = CFG.dnn_pixel
        model_3x3 = PixelDNN(
            input_size=8,
            hidden_dims=cfg_p.window_3x3_hidden,   # (20, 18, 5, 1)
            output_size=1
        ).to(device)
        log.info("[Stage 8] 3×3 PixelDNN initialised.")

    if model_1x8 is None:
        cfg_p = CFG.dnn_pixel
        model_1x8 = PixelDNN(
            input_size=8,
            hidden_dims=cfg_p.window_1x8_hidden,   # (20, 18, 5, 2)
            output_size=2
        ).to(device)
        log.info("[Stage 8] 1×8 PixelDNN initialised.")

    refined_3x3 = refined_1x8 = None

    if window in ("3x3", "both"):
        log.info("[Stage 8] Running 3×3 DNN refinement …")
        refined_3x3 = refine_3x3(coarse_img, model_3x3, device)
        cv2.imwrite(str(result_dir / "08_dnn_pixel_refined_3x3.png"),
                    np.clip(refined_3x3, 0, 255).astype(np.uint8))

    if window in ("1x8", "both"):
        log.info("[Stage 8] Running 1×8 DNN refinement …")
        refined_1x8 = refine_1x8(coarse_img, model_1x8, device)
        cv2.imwrite(str(result_dir / "08_dnn_pixel_refined_1x8.png"),
                    np.clip(refined_1x8, 0, 255).astype(np.uint8))

    # Pick best output
    if window == "3x3":
        refined = refined_3x3
    elif window == "1x8":
        refined = refined_1x8
    else:  # both — use 3×3 as primary (higher R)
        refined = refined_3x3

    cv2.imwrite(str(result_dir / "08_dnn_pixel_refined_final.png"),
                np.clip(refined, 0, 255).astype(np.uint8))

    log.info(f"[Stage 8] DNN pixel refinement done ({window} window).")
    return refined, model_3x3, model_1x8
