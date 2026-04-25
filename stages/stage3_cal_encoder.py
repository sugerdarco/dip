"""
Stage 3 — Convolutional encoder + Cross-Attention Learning (CAL).

Paper §Cross-attention learning for adaptive feature selection:

    Attention(Q, K, V) = softmax(QKᵀ / √dk) · V           [Eq. 3]
    MultiHead(Q,K,V)   = Concat(head₁,...,headₕ) · Wₒ     [Eq. 4]

Architecture (encoder):
  DWT subbands [4, H/2, W/2]
    → Conv block 1  (32 ch, 3×3, BN, ReLU)
    → Conv block 2  (64 ch, 3×3, BN, ReLU)
    → CAL module    (Q from low-freq, K/V from high-freq)
    → Conv block 3  (128 ch, 3×3, BN, ReLU)
    → [B, 128, H/2, W/2]  → passed to VAE encoder

CAL specifics (paper Table 2):
  num_heads  = 8
  embed_dim  = 64
  scoring    = softmax

Saved outputs:
  result_dir/03_cal_attention_map.png
  result_dir/03_encoder_features.npy
"""

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, kernel: int = 3,
                use_bn: bool = True) -> nn.Sequential:
    """Conv → BN → ReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Learning (CAL) module.

    Queries  (Q) — from structural (low-freq LL subband path)
    Keys     (K) — from high-freq detail path
    Values   (V) — from high-freq detail path

    Both paths are projected to embed_dim before attention.
    """

    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 num_heads: int = None,
                 dropout: float = None):
        super().__init__()
        self.embed_dim = embed_dim or CFG.cal.embed_dim
        self.num_heads = num_heads or CFG.cal.num_heads
        self.dropout   = dropout   or CFG.cal.dropout
        self.head_dim  = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        # Project spatial features → Q, K, V sequences
        self.proj_q = nn.Conv2d(in_channels, self.embed_dim, 1)
        self.proj_k = nn.Conv2d(in_channels, self.embed_dim, 1)
        self.proj_v = nn.Conv2d(in_channels, self.embed_dim, 1)
        self.proj_out = nn.Conv2d(self.embed_dim, in_channels, 1)

        self.scale = math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(self.dropout)
        self.out_drop  = nn.Dropout(self.dropout)

        # Store attention weights for visualisation
        self.last_attn_weights: torch.Tensor = None

    def forward(self,
                x_struct: torch.Tensor,
                x_detail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_struct : [B, C, H, W]  structural (LL-path) features  → Q
            x_detail : [B, C, H, W]  high-freq detail features       → K, V

        Returns:
            out      : [B, C, H, W]  attention-weighted feature map
            attn_map : [B, num_heads, HW, HW]  attention weights (for vis)
        """
        B, C, H, W = x_struct.shape
        N = H * W

        def project_reshape(proj, x):
            # [B, embed_dim, H, W] → [B, num_heads, N, head_dim]
            x = proj(x)                                      # [B, E, H, W]
            x = x.reshape(B, self.num_heads, self.head_dim, N)
            return x.permute(0, 1, 3, 2)                    # [B, H, N, D]

        Q = project_reshape(self.proj_q, x_struct)
        K = project_reshape(self.proj_k, x_detail)
        V = project_reshape(self.proj_v, x_detail)

        # Scaled dot-product attention  [Eq. 3]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        self.last_attn_weights = attn.detach()

        attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)                          # [B, H, N, D]

        # Reshape back to spatial  [Eq. 4]
        out = out.permute(0, 1, 3, 2)                        # [B, H, D, N]
        out = out.reshape(B, self.embed_dim, H, W)
        out = self.proj_out(out)
        out = self.out_drop(out)

        return out, attn


# ─────────────────────────────────────────────────────────────────────────────
# Full encoder
# ─────────────────────────────────────────────────────────────────────────────

class DWTEncoder(nn.Module):
    """
    3-block conv encoder with CAL inserted after block 2.

    Input:  [B, 4, H/2, W/2]   (4 DWT subbands)
    Output: [B, 128, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        cfg = CFG.encoder

        # Block 1 — both LL and HH paths share initial conv
        self.conv1 = _conv_block(cfg.in_channels, 32)     # 4 → 32

        # Block 2 — split into structural and detail streams
        self.conv2_struct = _conv_block(32, 64)            # LL path → Q
        self.conv2_detail = _conv_block(32, 64)            # HF path → K, V

        # Cross-Attention Learning
        self.cal = CrossAttentionBlock(in_channels=64)

        # Block 3 — fuse attended features
        self.conv3 = _conv_block(64, 128)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : [B, 4, H, W]  stacked DWT subbands

        Returns:
            features : [B, 128, H, W]
            attn_map : [B, 8, HW, HW]
        """
        # Shared first conv
        x1 = self.conv1(x)                                 # [B, 32, H, W]

        # Split: structural (LL = channel 0) vs detail (LH,HL,HH = ch 1-3)
        # After conv1 the channels are mixed, so we use both streams from x1
        x_struct = self.conv2_struct(x1)                   # [B, 64, H, W]
        x_detail = self.conv2_detail(x1)                   # [B, 64, H, W]

        # Cross-attention: Q from struct, K/V from detail
        attended, attn_map = self.cal(x_struct, x_detail)  # [B, 64, H, W]

        # Residual + final conv block
        fused = attended + x_struct                         # residual
        features = self.conv3(fused)                        # [B, 128, H, W]

        return features, attn_map


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(stacked: np.ndarray,
        result_dir: Path,
        model: DWTEncoder = None,
        device: str = None) -> Tuple[torch.Tensor, torch.Tensor, DWTEncoder]:
    """
    Run CAL encoder on stacked DWT subbands.

    Args:
        stacked    : [4, H, W] float32 numpy array from Stage 2
        result_dir : output directory
        model      : pre-built DWTEncoder (created once and reused)
        device     : "cuda" or "cpu"

    Returns:
        features   : [1, 128, H, W] torch tensor (on device)
        attn_map   : [1, 8, HW, HW] attention weights
        model      : the encoder (for reuse / training)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    if model is None:
        model = DWTEncoder().to(device)
        log.info("[Stage 3] DWTEncoder initialised.")

    model.eval()
    with torch.no_grad():
        # [4, H, W] → [1, 4, H, W] tensor
        t = torch.from_numpy(stacked).unsqueeze(0).to(device)
        # Normalise to [0, 1] for the model
        t = t / 255.0

        features, attn_map = model(t)

    # ── Save attention map visualisation ────────────────────────────────────
    # Average across heads and spatial dims for a 2D heatmap
    avg_attn = attn_map[0].mean(0)                          # [HW, HW]
    h = w = int(avg_attn.shape[0] ** 0.5)
    if h * w == avg_attn.shape[0]:
        heatmap = avg_attn.mean(-1).reshape(h, w).cpu().numpy()
    else:
        heatmap = avg_attn.mean(-1).cpu().numpy().reshape(1, -1)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(result_dir / "03_cal_attention_map.png"), heatmap_color)

    # Save feature maps (mean across channels)
    feat_np = features[0].mean(0).cpu().numpy()
    feat_vis = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-8)
    np.save(str(result_dir / "03_encoder_features.npy"), features[0].cpu().numpy())

    log.info(f"[Stage 3] CAL done → features {features.shape}, "
             f"saved attention map to {result_dir}")

    return features, attn_map, model
