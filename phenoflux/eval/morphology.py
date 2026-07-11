#!/usr/bin/env python
"""Per-crop morphology feature extraction for microalgae phenotype evaluation.

Each single-cell crop (128x128 RGB, one cell per crop) is reduced to an
interpretable morphology feature vector. The extractor runs on-the-fly (no
segmentation masks required): the crop itself is the segmentation unit. It is
applied IDENTICALLY to generated, real-treated, and control crops so that
distribution distances and the identity baseline are computed on the same
feature basis.

Feature basis mirrors the mask-based field summary
(``scripts/build_field_metadata.py``): shape + intensity + GLCM texture.

Foreground convention matches ``aggregate_microalgae.fg_mean_intensity``:
grayscale = max over channels, foreground = grayscale > 0.05.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

# Feature order is the public contract; distribution_eval relies on it.
FEATURES: list[str] = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "solidity",
    "eccentricity",
    "mean_intensity",
    "std_intensity",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
]

_GLCM_LEVELS = 256
_MIN_FG_PIXELS = 20  # below this the crop is treated as empty (no reliable region)


def _largest_component_mask(fg: np.ndarray) -> np.ndarray | None:
    """Return a boolean mask of the largest connected foreground component."""
    labelled = label(fg, connectivity=2)
    if labelled.max() == 0:
        return None
    # regionprops labels are 1-indexed; pick the region with the largest area.
    regions = regionprops(labelled)
    largest = max(regions, key=lambda r: r.area)
    return labelled == largest.label


def _texture_features(gray_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """GLCM texture on the 8-bit grayscale crop, averaged over 0 and 90 degrees.

    The GLCM is computed over the bounding region of the foreground so that the
    background does not dominate the co-occurrence statistics.
    """
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = gray_u8[y0:y1, x0:x1]
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return {
            "texture_contrast": 0.0,
            "texture_homogeneity": 1.0,
            "texture_energy": 1.0,
            "texture_correlation": 1.0,
        }
    glcm = graycomatrix(
        patch,
        distances=[1],
        angles=[0.0, np.pi / 2.0],
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return {
        "texture_contrast": float(graycoprops(glcm, "contrast").mean()),
        "texture_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "texture_energy": float(graycoprops(glcm, "energy").mean()),
        "texture_correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def extract_features(img: np.ndarray, fg_thresh: float = 0.05) -> np.ndarray | None:
    """Extract a morphology feature vector from one crop.

    Args:
        img: (H, W, 3) RGB image in [0, 1].
        fg_thresh: foreground threshold on the max-over-channels grayscale.

    Returns:
        Feature vector (len(FEATURES),) in FEATURES order, or None if the crop
        has no reliable foreground region.
    """
    gray = img.max(axis=2)  # (H, W) in [0, 1]; matches fg_mean_intensity convention
    fg = gray > fg_thresh
    if int(fg.sum()) < _MIN_FG_PIXELS:
        return None

    mask = _largest_component_mask(fg)
    if mask is None:
        return None

    props = regionprops(mask.astype(np.int32))[0]
    area = float(props.area)
    perimeter = float(props.perimeter)
    circularity = (
        float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    )
    minor = float(props.axis_minor_length)
    major = float(props.axis_major_length)
    aspect_ratio = float(major / minor) if minor > 1e-6 else 0.0
    solidity = float(props.solidity)
    eccentricity = float(props.eccentricity)

    fg_pixels = gray[mask]
    mean_intensity = float(fg_pixels.mean())
    std_intensity = float(fg_pixels.std())

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    tex = _texture_features(gray_u8, mask)

    values = {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        **tex,
    }
    return np.array([values[f] for f in FEATURES], dtype=np.float64)


def _pad_or_shrink_to_canvas_np(img: np.ndarray, image_size: int) -> np.ndarray:
    """NumPy port of training.data_utils._pad_or_shrink_to_canvas (RGB, [0,1]).

    Keeps crop pixel scale: small crops are centered on a black canvas, only
    oversized crops are shrunk. This MUST match the training/eval load path so
    generated 128x128 crops and raw variable-size crops occupy the same pixel
    scale before morphology/FID comparison.
    """
    from PIL import Image as _Image

    h, w = img.shape[:2]
    if h > image_size or w > image_size:
        scale = image_size / max(h, w)
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        pil = _Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        pil = pil.resize((new_w, new_h), _Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float64) / 255.0
        h, w = img.shape[:2]
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float64)
    y0 = (image_size - h) // 2
    x0 = (image_size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img
    return canvas


def _load_rgb(path: Path, canvas_size: int | None = None) -> np.ndarray | None:
    """Load an image as (H, W, 3) float in [0, 1], or None if missing/unreadable.

    If canvas_size is given, center the crop on a black canvas at native pixel
    scale (matching the training load path), so raw variable-size crops become
    comparable to generated 128x128 crops.
    """
    if not Path(path).exists():
        return None
    try:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    if canvas_size is not None and img.shape[:2] != (canvas_size, canvas_size):
        img = _pad_or_shrink_to_canvas_np(img, canvas_size)
    return img


def extract_population(
    paths: list[Path], fg_thresh: float = 0.05, canvas_size: int | None = None
) -> np.ndarray:
    """Extract a (N, F) feature matrix over a list of crop paths.

    Crops that are missing, unreadable, or have no foreground are dropped, so
    N may be smaller than len(paths). Returns an empty (0, len(FEATURES)) array
    if nothing could be extracted. Pass canvas_size (e.g. 128) to normalize raw
    crops to the training canvas before feature extraction.
    """
    rows: list[np.ndarray] = []
    for path in paths:
        img = _load_rgb(Path(path), canvas_size=canvas_size)
        if img is None:
            continue
        feat = extract_features(img, fg_thresh=fg_thresh)
        if feat is not None:
            rows.append(feat)
    if not rows:
        return np.empty((0, len(FEATURES)), dtype=np.float64)
    return np.stack(rows, axis=0)


__all__ = ["FEATURES", "extract_features", "extract_population", "_load_rgb"]
