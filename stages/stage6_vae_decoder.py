"""
Stage 6+7 — VAE Decoder  (entropy decode → feature maps).

Paper §Integrated compression framework:
  "The latent representation is decoded back into feature maps and then
   transformed back into the original image domain using an inverse
   wavelet transform during decompression."

Architecture (mirrors the encoder):
    z  [B, latent_dim=256]
      → FC → [B, 256 × 4 × 4]
      → reshape → [B, 256, 4, 4]
      → 4 × (TransposedConv 3×3 + BN + ReLU)  — each doubles spatial res
      → [B, 4, H/2, W/2]   (4 channels = one per DWT subband)

Outputs:
    result_dir/07_decoded_subbands.npy   [4, H/2, W/2] — passed to Stage 8/9
    result_dir/07_decoded_feature_vis.png
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VAE Decoder network
# ─────────────────────────────────────────────────────────────────────────────

class VAEDecoder(nn.Module):
    """
    Symmetric decoder that mirrors VAEEncoder.

    Accepts z of shape [B, latent_dim] and outputs [B, 4, H, W]
    representing the four reconstructed DWT subbands.
    """

    def __init__(self, latent_dim: int = None, out_hw: int = 16):
        """
        Args:
            latent_dim : size of latent vector (256)
            out_hw     : starting spatial size after FC unflattening (4)
        """
        super().__init__()
        cfg_v = CFG.vae
        self.latent_dim = latent_dim or cfg_v.latent_dim
        self.out_hw = out_hw

        ch = list(reversed(cfg_v.encoder_channels))   # [256, 128, 64, 32]
        k  = cfg_v.kernel_size

        self.start_ch = ch[0]                  # 256
        flat_dim = ch[0] * out_hw * out_hw     # 256 * 16 * 16 = 65536

        self.fc = nn.Linear(self.latent_dim, flat_dim)

        # 4 transposed-conv blocks, each doubling spatial resolution
        self.dec = nn.Sequential(
            # Block 1: 256 → 128
            nn.ConvTranspose2d(ch[0], ch[1], k, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),

            # Block 2: 128 → 64
            nn.ConvTranspose2d(ch[1], ch[2], k, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),

            # Block 3: 64 → 32
            nn.ConvTranspose2d(ch[2], ch[3], k, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),

            # Block 4: 32 → 4  (one channel per DWT subband)
            nn.ConvTranspose2d(ch[3], 4, k, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.Tanh(),            # outputs in [-1, 1] → rescaled later
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : [B, latent_dim]

        Returns:
            out : [B, 4, H, W]  decoded subband feature maps
        """
        x = self.fc(z)                              # [B, flat_dim]
        x = x.view(-1, self.start_ch,
                    self.out_hw, self.out_hw)        # [B, 256, out_hw, out_hw]
        x = self.dec(x)                             # [B, 4, H, W]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tensor → reconstructed subbands dict
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_subbands(decoded: torch.Tensor,
                        target_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Convert [B, 4, H, W] decoded tensor → subband dict ready for inverse DWT.

    Rescales Tanh output [-1,1] to approximate original DWT coefficient range
    using the target_shape to guide magnitude.
    """
    arr = decoded[0].cpu().numpy()   # [4, H, W]

    # Resize each subband to target_shape (H/2 × W/2 for single-level DWT)
    th, tw = target_shape[0] // 2, target_shape[1] // 2

    def resize(x):
        return cv2.resize(x.astype(np.float32), (tw, th),
                          interpolation=cv2.INTER_LINEAR)

    # Rescale [-1,1] → a reasonable coefficient range; refine with training
    scale = 128.0

    return {
        "LL": resize(arr[0]) * scale,
        "LH": resize(arr[1]) * scale * 0.5,
        "HL": resize(arr[2]) * scale * 0.5,
        "HH": resize(arr[3]) * scale * 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(z: torch.Tensor,
        result_dir: Path,
        original_shape: Tuple[int, int] = (512, 512),
        model: VAEDecoder = None,
        device: str = None) -> Tuple[Dict[str, np.ndarray], VAEDecoder]:
    """
    Decode latent z → reconstructed DWT subband dict.

    Args:
        z              : [B, latent_dim] from Stage 5 (decompressed)
        result_dir     : output directory
        original_shape : (H, W) of original image
        model          : pre-built VAEDecoder
        device         : "cuda" or "cpu"

    Returns:
        rec_subbands : dict {"LL","LH","HL","HH"} float32 arrays
        model        : decoder (for reuse)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    if model is None:
        model = VAEDecoder().to(device)
        log.info("[Stage 7] VAEDecoder initialised.")

    model.eval()
    with torch.no_grad():
        decoded = model(z)                         # [B, 4, H, W]

    rec_subbands = tensor_to_subbands(decoded, original_shape)

    # ── Save outputs ─────────────────────────────────────────────────────────
    arr4 = decoded[0].cpu().numpy()                # [4, H, W]
    np.save(str(result_dir / "07_decoded_subbands.npy"), arr4)

    # Visualise mean of 4 channels
    vis = arr4.mean(axis=0)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8) * 255
    cv2.imwrite(str(result_dir / "07_decoded_feature_vis.png"),
                vis.astype(np.uint8))

    log.info(f"[Stage 7] VAE decoded → tensor {decoded.shape}, "
             f"subbands: LL={rec_subbands['LL'].shape}")

    return rec_subbands, model
