"""Leaf masks by excess green index and Otsu threshold, as in PlantCV workflows.

Classical baseline with no learned weights. SAM 2 masks can replace it later
once a verified checkpoint and licence note are added to ``docs/models.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def excess_green(image: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on chromaticity coordinates; image is (H, W, 3) RGB."""
    rgb = np.asarray(image, float)
    total = rgb.sum(axis=2)
    total[total == 0] = 1
    r, g, b = (rgb[..., i] / total for i in range(3))
    return 2 * g - r - b


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold maximising between-class variance."""
    v = np.asarray(values, float).ravel()
    if v.min() == v.max():
        raise ValueError("Constant image; cannot threshold")
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[np.argmax(between[:-1])])


def leaf_mask(
    image: np.ndarray, min_area_fraction: float = 0.001, largest_only: bool = False
) -> np.ndarray:
    """Boolean leaf mask: ExG > Otsu, opening, hole filling, small object removal."""
    exg = excess_green(image)
    mask = exg > otsu_threshold(exg)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(areas >= min_area_fraction * mask.size) + 1
    if largest_only and len(keep):
        keep = [int(np.argmax(areas)) + 1]
    return np.isin(labels, keep)


def remove_background(image: np.ndarray, mask: np.ndarray, fill: int = 0) -> np.ndarray:
    """Background-removed variant for ablation: pixels outside mask set to ``fill``."""
    out = np.array(image, copy=True)
    out[~np.asarray(mask, bool)] = fill
    return out


def segment_image(path: str | Path, out_dir: str | Path) -> tuple[Path, Path, float]:
    """Save ``<stem>_mask.png`` and ``<stem>_nobg.png``; return paths and leaf fraction."""
    from PIL import Image

    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"))
    mask = leaf_mask(image)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    mask_path, nobg_path = out / f"{stem}_mask.png", out / f"{stem}_nobg.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    Image.fromarray(remove_background(image, mask)).save(nobg_path)
    return mask_path, nobg_path, float(mask.mean())
