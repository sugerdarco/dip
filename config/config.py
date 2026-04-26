"""
Central configuration — edit this file to change any hyperparameter.
All paths, model settings, and training params live here.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Paths:
    # ── Input ──────────────────────────────────────────────────────────────
    working_dir: str = PROJECT_ROOT                                   # project root
    contents_dir: str = os.path.join(PROJECT_ROOT, "data/contents")        # your MRI/CT folders
    output_dir: str = os.path.join(PROJECT_ROOT, "data/outputs")           # all results saved here

    # ── Checkpoints ────────────────────────────────────────────────────────
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "data/checkpoints")
    checkpoint_name: str = "best_model.pth"


@dataclass
class ImageConfig:
    target_size: Tuple[int, int] = (512, 512)         # resize all images to this
    grayscale: bool = True                             # medical images → 1 channel
    intensity_range: Tuple[float, float] = (0, 255)   # normalise to this before model
    supported_extensions: Tuple = (
        ".png", ".jpg", ".jpeg", ".tiff", ".tif",     # standard raster
        ".dcm",                                        # DICOM (MRI/CT)
    )


@dataclass
class WaveletConfig:
    wavelet: str = "haar"                             # wavelet basis
    level: int = 1                                    # decomposition levels
    # subbands produced: LL, LH, HL, HH


@dataclass
class CALConfig:
    """Cross-Attention Learning module parameters (paper §Cross-attention)."""
    num_heads: int = 8
    embed_dim: int = 64
    dropout: float = 0.1


@dataclass
class EncoderConfig:
    """Convolutional encoder before CAL (paper §Deep neural network design)."""
    in_channels: int = 4          # 4 DWT subbands stacked as channels
    conv_channels: Tuple = (32, 64, 128)
    kernel_size: int = 3
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class VAEConfig:
    """Variational Autoencoder settings (paper §Integrated compression framework)."""
    latent_dim: int = 256
    encoder_channels: Tuple = (32, 64, 128, 256)    # 4 conv layers
    kernel_size: int = 3
    pool_size: int = 2
    prior: str = "gaussian"

    # Loss weights  L = λ1·MSE + λ2·Perceptual + λ3·KL
    lambda_mse: float = 1.0
    lambda_perceptual: float = 0.1
    lambda_kl: float = 0.001


@dataclass
class DNNPixelConfig:
    """Two DNN pixel-estimation approaches (paper §Using DNN to estimate image pixels)."""
    # 3×3 window model — higher accuracy
    window_3x3_hidden: Tuple = (20, 18, 5, 1)       # R=0.997, MSE≈6.1e-6
    window_3x3_epochs: int = 400
    window_3x3_target_mse: float = 6.1e-6

    # 1×8 window model — faster serial prediction
    window_1x8_hidden: Tuple = (20, 18, 5, 2)       # R=0.951, MSE≈1.4e-6
    window_1x8_epochs: int = 350

    default_window: str = "3x3"                      # which to use at inference


@dataclass
class SpatialFilterConfig:
    """Trainable spatial filter (paper §Entropy-based compression and reconstruction)."""
    kernel_size: int = 3
    in_channels: int = 1
    use_attention_aware: bool = True


@dataclass
class TrainingConfig:
    """Matches paper Table 2 exactly."""
    optimizer: str = "adam"
    learning_rate: float = 0.0002
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 32
    num_epochs: int = 150
    weight_init: str = "xavier_normal"
    dropout_rate: float = 0.3
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    test_split: float = 0.15
    # cross-validation
    n_folds: int = 5
    n_runs: int = 3                                  # independent runs per fold
    convergence_epoch: int = 120                     # expected stabilisation


@dataclass
class Config:
    """Master config — instantiate this in any module."""
    paths: Paths = field(default_factory=Paths)
    image: ImageConfig = field(default_factory=ImageConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    cal: CALConfig = field(default_factory=CALConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    dnn_pixel: DNNPixelConfig = field(default_factory=DNNPixelConfig)
    spatial_filter: SpatialFilterConfig = field(default_factory=SpatialFilterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    device: str = "cuda"          # "cuda" for your 24GB GPU, "cpu" fallback
    seed: int = 42
    num_workers: int = 4          # DataLoader workers
    pin_memory: bool = True       # faster GPU transfer


# Singleton — import this everywhere
CFG = Config()
