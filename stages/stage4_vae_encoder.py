"""
Stage 4 — Variational Autoencoder (VAE) Encoder.

Paper §Integrated compression framework with variational autoencoder:

    Loss = λ₁·L_MSE + λ₂·L_Perceptual + λ₃·KL-divergence   [Eq. 2]

The VAE encoder maps attended features → (μ, log_var) of a Gaussian
latent distribution, then samples z = μ + ε·σ  (reparameterisation trick).

Architecture:
    features [B, 128, H/2, W/2]
      → 4 × (Conv 3×3 + BN + ReLU + MaxPool 2×2)
      → Flatten
      → FC → μ (latent_dim=256)
      → FC → log_var (latent_dim=256)
      → z  = reparameterise(μ, log_var)

Outputs saved:
    result_dir/04_vae_latent_z.npy
    result_dir/04_vae_mu.npy
    result_dir/04_vae_logvar.npy
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VAE Encoder network
# ─────────────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """
    4-layer convolutional encoder → latent Gaussian (μ, log_var).

    Input shape: [B, 128, H, W]   (H = W = 256 for 512-input + 1-level DWT)
    """

    def __init__(self, latent_dim: int = None):
        super().__init__()
        cfg_v = CFG.vae
        self.latent_dim = latent_dim or cfg_v.latent_dim

        ch = cfg_v.encoder_channels   # (32, 64, 128, 256)
        k  = cfg_v.kernel_size        # 3
        p  = cfg_v.pool_size          # 2

        # 4 conv blocks, each halving spatial resolution
        self.enc = nn.Sequential(
            # Block 1: 128 → ch[0]
            nn.Conv2d(128, ch[0], k, padding=k//2, bias=False),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(p),
            nn.Dropout2d(CFG.training.dropout_rate),

            # Block 2: ch[0] → ch[1]
            nn.Conv2d(ch[0], ch[1], k, padding=k//2, bias=False),
            nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
            nn.MaxPool2d(p),
            nn.Dropout2d(CFG.training.dropout_rate),

            # Block 3: ch[1] → ch[2]
            nn.Conv2d(ch[1], ch[2], k, padding=k//2, bias=False),
            nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
            nn.MaxPool2d(p),
            nn.Dropout2d(CFG.training.dropout_rate),

            # Block 4: ch[2] → ch[3]
            nn.Conv2d(ch[2], ch[3], k, padding=k//2, bias=False),
            nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
            nn.MaxPool2d(p),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = ch[3] * 4 * 4      # 256 * 16 = 4096

        # Two FC heads for μ and log_var
        self.fc_mu      = nn.Linear(flat_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, self.latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu      : [B, latent_dim]
            log_var : [B, latent_dim]
        """
        h = self.enc(x)
        h = self.adaptive_pool(h)
        h = h.flatten(start_dim=1)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# Reparameterisation trick
# ─────────────────────────────────────────────────────────────────────────────

def reparameterise(mu: torch.Tensor,
                   log_var: torch.Tensor) -> torch.Tensor:
    """
    z = μ + ε · σ,  ε ~ N(0, I)   (reparameterisation trick).
    During inference (eval mode) returns μ directly for determinism.
    """
    if not mu.requires_grad:      # inference — deterministic
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence loss term
# ─────────────────────────────────────────────────────────────────────────────

def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(μ, σ²) and N(0, I).
    KL = -½ · Σ(1 + log_var - μ² - exp(log_var))
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(features: torch.Tensor,
        result_dir: Path,
        model: VAEEncoder = None,
        device: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, VAEEncoder]:
    """
    Encode attended features into latent space.

    Args:
        features   : [B, 128, H, W] from Stage 3 (on device)
        result_dir : output directory
        model      : pre-built VAEEncoder (created once)
        device     : "cuda" or "cpu"

    Returns:
        z          : [B, latent_dim] sampled latent vector
        mu         : [B, latent_dim]
        log_var    : [B, latent_dim]
        model      : encoder (for reuse)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    if model is None:
        model = VAEEncoder().to(device)
        log.info("[Stage 4] VAEEncoder initialised.")

    model.eval()
    with torch.no_grad():
        mu, log_var = model(features)
        z = reparameterise(mu, log_var)

    # ── Save latent vectors ──────────────────────────────────────────────────
    np.save(str(result_dir / "04_vae_z.npy"),       z.cpu().numpy())
    np.save(str(result_dir / "04_vae_mu.npy"),      mu.cpu().numpy())
    np.save(str(result_dir / "04_vae_logvar.npy"),  log_var.cpu().numpy())

    log.info(f"[Stage 4] VAE encoded → z shape {z.shape} | "
             f"μ range [{mu.min():.3f}, {mu.max():.3f}]")

    return z, mu, log_var, model
