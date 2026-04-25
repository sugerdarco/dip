"""
Stage 9 — Trainable Spatial Filter (post-reconstruction).

Paper §Entropy-based compression and reconstruction:
  "A trainable spatial filter (e.g., a 3×3 convolutional kernel) is
   incorporated into the neural network. The filter weights are refined
   according to picture quality criteria like SSIM, PSNR, and CC."

  "This method guarantees superior edge preservation and structural
   augmentation without the need for manual optimization techniques."

Paper Table 1 comparison:
  GWO   → PSNR 27.4 dB, SSIM 0.86, Obj. fn 4.56
  WHO   → PSNR 28.1 dB, SSIM 0.88, Obj. fn 4.36
  Ours  → PSNR 31.7 dB, SSIM 0.91, Obj. fn 3.21  (learnt via backprop)

Implementation:
  • Single 3×3 conv layer (attention-aware: weights gated by local variance)
  • At inference: apply learned kernel to DNN-refined image
  • Kernel is Xavier-initialised; refines during training via SSIM+MSE loss

Outputs:
    result_dir/09_spatial_filtered.png   ← final edge-enhanced image
    result_dir/09_filter_kernel.npy      ← learned 3×3 kernel weights
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trainable Spatial Filter module
# ─────────────────────────────────────────────────────────────────────────────

class TrainableSpatialFilter(nn.Module):
    """
    Attention-aware 3×3 convolutional spatial filter.

    The attention gate modulates filter weights by local image variance —
    high-variance regions (edges, lesions) get stronger filtering,
    smooth regions get weaker filtering.
    """

    def __init__(self,
                 in_channels: int = 1,
                 kernel_size: int = 3,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        pad = kernel_size // 2

        # Primary trainable conv  (3×3, learns edge/detail weights)
        self.conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, padding=pad, bias=True
        )

        if use_attention:
            # Attention gate: local variance → weight map in [0, 1]
            self.attn_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=pad, bias=False),
                nn.Sigmoid()
            )

        # Residual scale — keeps identity useful at start of training
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        if self.use_attention:
            nn.init.xavier_normal_(self.attn_conv[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, H, W] normalised [0, 1]

        Returns:
            out : [B, 1, H, W] filtered image
        """
        filtered = self.conv(x)

        if self.use_attention:
            attn = self.attn_conv(x)          # local attention gate
            filtered = filtered * attn

        # Residual: out = x + scale * filtered
        out = x + self.residual_scale * filtered
        return torch.clamp(out, 0.0, 1.0)

    def get_kernel(self) -> np.ndarray:
        """Return the learned 3×3 kernel weights as numpy array."""
        return self.conv.weight[0, 0].detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SSIM-based loss for filter training
# ─────────────────────────────────────────────────────────────────────────────

def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11, C1: float = 0.01**2,
              C2: float = 0.03**2) -> torch.Tensor:
    """
    1 - SSIM  (minimise to maximise structural similarity).
    Fast single-scale version suitable for per-batch training.
    """
    mu_x = F.avg_pool2d(pred,   window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred   * pred,   window_size, 1, window_size//2) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu_y2
    sigma_xy = F.avg_pool2d(pred   * target, window_size, 1, window_size//2) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return 1.0 - ssim_map.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.8) -> torch.Tensor:
    """
    alpha * MSE + (1-alpha) * SSIM_loss
    Paper trains filter against SSIM, PSNR, and CC.
    """
    mse = F.mse_loss(pred, target)
    ss  = ssim_loss(pred, target)
    return alpha * mse + (1 - alpha) * ss


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(refined_img: np.ndarray,
        result_dir: Path,
        model: TrainableSpatialFilter = None,
        device: str = None) -> tuple:
    """
    Apply trainable spatial filter to DNN-refined image.

    Args:
        refined_img : float32 [H, W] in [0, 255] from Stage 8
        result_dir  : output directory
        model       : pre-built TrainableSpatialFilter
        device      : "cuda" or "cpu"

    Returns:
        filtered_img : float32 [H, W] in [0, 255]
        model        : filter model (for reuse / continued training)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    if model is None:
        model = TrainableSpatialFilter(
            in_channels=CFG.spatial_filter.in_channels,
            kernel_size=CFG.spatial_filter.kernel_size,
            use_attention=CFG.spatial_filter.use_attention_aware,
        ).to(device)
        log.info("[Stage 9] TrainableSpatialFilter initialised.")

    model.eval()
    with torch.no_grad():
        # [H, W] → [1, 1, H, W] tensor in [0, 1]
        t = torch.from_numpy(refined_img / 255.0).float()
        t = t.unsqueeze(0).unsqueeze(0).to(device)

        filtered_t = model(t)                      # [1, 1, H, W]

    filtered_img = (filtered_t[0, 0].cpu().numpy() * 255.0).astype(np.float32)

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_path = result_dir / "09_spatial_filtered.png"
    cv2.imwrite(str(out_path),
                np.clip(filtered_img, 0, 255).astype(np.uint8))

    kernel = model.get_kernel()
    np.save(str(result_dir / "09_filter_kernel.npy"), kernel)

    # Save human-readable kernel
    kernel_txt = result_dir / "09_filter_kernel.txt"
    kernel_txt.write_text(
        "Learned 3×3 spatial filter kernel:\n" +
        np.array2string(kernel, precision=5, separator=", ")
    )

    log.info(f"[Stage 9] Spatial filter done → {out_path}")
    return filtered_img, model
