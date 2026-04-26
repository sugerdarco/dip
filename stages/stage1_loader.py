"""
Stage 1 — Image loading and preprocessing.

Handles:
  • DICOM  (.dcm)  — MRI / CT from scanner
  • PNG / TIFF / JPEG — standard raster medical images
  • Resizes to 512×512, normalises to [0, 255] grayscale float32
  • Walks working_dir/contents/ recursively, preserving folder structure
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import pydicom
    DICOM_OK = True
except ImportError:
    DICOM_OK = False
    logging.warning("pydicom not installed — DICOM files will be skipped. "
                    "Run: pip install pydicom")

from config.config import CFG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom(path: str) -> np.ndarray:
    """Read a DICOM file and return a float32 [H, W] array normalised to [0,255]."""
    if not DICOM_OK:
        raise RuntimeError("pydicom required for DICOM files.")
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply DICOM rescale slope / intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Window to [0, 255]
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return arr


def load_raster(path: str) -> np.ndarray:
    """Read PNG / TIFF / JPEG and return float32 [H, W] in [0, 255]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cv2 could not read: {path}")
    return img.astype(np.float32)


def load_image(path: str) -> np.ndarray:
    """Unified loader — auto-detects format."""
    ext = Path(path).suffix.lower()
    if ext == ".dcm":
        arr = load_dicom(path)
    else:
        arr = load_raster(path)

    # If multi-frame (e.g. 3-channel DICOM), take first frame
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(arr: np.ndarray,
               target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    1. Resize to target_size (default 512×512).
    2. Clip to [0, 255].
    3. Return float32.
    """
    if target_size is None:
        target_size = CFG.image.target_size

    arr = np.clip(arr, 0, 255)

    h, w = arr.shape
    if (h, w) != target_size:
        arr = cv2.resize(arr, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LANCZOS4)

    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_contents(contents_dir: str = None) -> List[dict]:
    """
    Walk contents_dir recursively and return a list of dicts:
      { "path": str, "stem": str, "modality": str, "rel_dir": str }

    Modality is inferred from folder name or extension:
      dcm → "CT_or_MRI", else "standard"
    """
    if contents_dir is None:
        contents_dir = CFG.paths.contents_dir

    contents_dir = Path(contents_dir)
    if not contents_dir.exists():
        raise FileNotFoundError(f"Contents directory not found: {contents_dir}")

    supported = set(CFG.image.supported_extensions)
    records = []

    for fpath in sorted(contents_dir.rglob("*")):
        if fpath.suffix.lower() not in supported:
            continue
        if not fpath.is_file():
            continue

        rel = fpath.relative_to(contents_dir)
        modality = "CT_or_MRI" if fpath.suffix.lower() == ".dcm" else "standard"

        records.append({
            "path": str(fpath),
            "stem": fpath.stem,
            "modality": modality,
            "rel_dir": str(rel.parent),    # subfolder relative to contents/
        })

    log.info(f"Found {len(records)} images in {contents_dir}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Output path builder
# ─────────────────────────────────────────────────────────────────────────────

def build_output_dir(record: dict, output_root: str = None) -> Path:
    """
    Returns:   output_root / folder_name_of_origin / image_title / results_of_particular_image
    """
    if output_root is None:
        output_root = CFG.paths.output_dir

    image_title = record["stem"]
    rel_dir = record.get("rel_dir", "")
    result_dir = Path(output_root) / rel_dir / image_title / f"results_of_{image_title}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

def run(record: dict) -> Tuple[np.ndarray, Path]:
    """
    Load + preprocess a single image record.

    Returns:
      arr        — float32 [H, W] in [0, 255]
      result_dir — Path where all outputs for this image will be saved
    """
    path = record["path"]
    log.info(f"[Stage 1] Loading: {path}")

    arr = load_image(path)
    arr = preprocess(arr)

    result_dir = build_output_dir(record)

    # Save normalised input for reference
    cv2.imwrite(str(result_dir / "00_input_normalised.png"), arr.astype(np.uint8))
    log.info(f"[Stage 1] Saved normalised input → {result_dir / '00_input_normalised.png'}")

    return arr, result_dir
