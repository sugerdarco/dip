"""
Stage 5 — Entropy coding (compression to bitstream).

Paper §Entropy-based compression and reconstruction:
  "adaptive learning-based methodology that dynamically modifies coding
   parameters according to the image content"

Implementation:
  • Quantise latent z to int8 using per-vector min-max scaling
  • Apply zlib (deflate) as the entropy coder — no external codec needed
  • Saves a .bin compressed bitstream + a metadata .json for reconstruction

Outputs:
    result_dir/05_compressed.bin     ← compressed bitstream
    result_dir/05_entropy_meta.json  ← scale / offset / shape for decoding
    result_dir/05_compression_stats.txt
"""

import json
import logging
import math
import zlib
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantise(z: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    """
    Linearly quantise float32 vector to int[bits].

    Returns:
        q      : int8 / int16 quantised array
        scale  : (max - min) / (2^bits - 1)
        offset : min value
    """
    z_min = float(z.min())
    z_max = float(z.max())
    levels = (2 ** bits) - 1

    scale = (z_max - z_min) / levels if z_max != z_min else 1.0
    z_norm = (z - z_min) / scale
    q = np.round(z_norm).clip(0, levels).astype(np.int16 if bits > 8 else np.int8)
    return q, scale, z_min


def dequantise(q: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Reverse quantisation → float32."""
    return (q.astype(np.float32) * scale + offset)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy encode / decode
# ─────────────────────────────────────────────────────────────────────────────

def entropy_encode(q: np.ndarray) -> bytes:
    """Compress quantised bytes with zlib (deflate level 9)."""
    raw = q.tobytes()
    return zlib.compress(raw, level=9)


def entropy_decode(bitstream: bytes, dtype, shape: tuple) -> np.ndarray:
    """Decompress bitstream → quantised array."""
    raw = zlib.decompress(bitstream)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Bit-rate / size helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_bpp(compressed_bytes: int,
                original_pixels: int) -> float:
    """bits per pixel = (compressed bytes × 8) / pixel count."""
    return (compressed_bytes * 8) / original_pixels


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / max(compressed_bytes, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry — compress
# ─────────────────────────────────────────────────────────────────────────────

def run(z: torch.Tensor,
        result_dir: Path,
        original_shape: Tuple[int, int] = (512, 512),
        bits: int = 8) -> Dict:
    """
    Quantise + entropy-encode the latent vector z.

    Args:
        z              : [B, latent_dim] latent vector from Stage 4
        result_dir     : output directory
        original_shape : (H, W) of input image (for bpp calculation)
        bits           : quantisation bits (8 → int8)

    Returns:
        meta : dict with all info needed for Stage 7 (decoding)
    """
    z_np = z.cpu().numpy().astype(np.float32)

    # Quantise
    q, scale, offset = quantise(z_np, bits=bits)

    # Entropy encode
    bitstream = entropy_encode(q)

    # Stats
    original_bytes  = z_np.nbytes
    compressed_bytes = len(bitstream)
    H, W = original_shape
    bpp = compute_bpp(compressed_bytes, H * W)
    cr  = compression_ratio(original_bytes, compressed_bytes)

    meta = {
        "z_shape":   list(z_np.shape),
        "q_dtype":   str(q.dtype),
        "scale":     scale,
        "offset":    offset,
        "bits":      bits,
        "original_bytes":   original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": round(cr, 4),
        "bpp":       round(bpp, 4),
        "original_shape": list(original_shape),
    }

    # ── Save outputs ─────────────────────────────────────────────────────────
    bin_path  = result_dir / "05_compressed.bin"
    meta_path = result_dir / "05_entropy_meta.json"
    stat_path = result_dir / "05_compression_stats.txt"

    with open(bin_path, "wb") as f:
        f.write(bitstream)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    stats_text = (
        f"=== Entropy Coding Stats ===\n"
        f"Latent shape      : {z_np.shape}\n"
        f"Quantisation bits : {bits}\n"
        f"Original size     : {original_bytes:,} bytes  "
        f"({original_bytes/1024:.1f} KB)\n"
        f"Compressed size   : {compressed_bytes:,} bytes  "
        f"({compressed_bytes/1024:.1f} KB)\n"
        f"Compression ratio : {cr:.2f}×\n"
        f"Bits per pixel    : {bpp:.4f} bpp\n"
    )
    stat_path.write_text(stats_text)

    log.info(f"[Stage 5] Compressed {original_bytes//1024} KB → "
             f"{compressed_bytes//1024} KB  ({bpp:.3f} bpp)")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry — decompress (Stage 7 calls this)
# ─────────────────────────────────────────────────────────────────────────────

def decompress(result_dir: Path,
               device: str = None) -> torch.Tensor:
    """
    Load bitstream + meta, entropy-decode, dequantise → z tensor.

    Returns:
        z : [B, latent_dim] float32 tensor (on device)
    """
    if device is None:
        device = CFG.device if torch.cuda.is_available() else "cpu"

    bin_path  = result_dir / "05_compressed.bin"
    meta_path = result_dir / "05_entropy_meta.json"

    with open(bin_path, "rb") as f:
        bitstream = f.read()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    dtype = np.int8 if meta["bits"] <= 8 else np.int16
    q = entropy_decode(bitstream, dtype, tuple(meta["z_shape"]))
    z_np = dequantise(q, meta["scale"], meta["offset"])

    z = torch.from_numpy(z_np).to(device)
    log.info(f"[Stage 7] Decompressed z shape: {z.shape}, bpp was {meta['bpp']}")
    return z
