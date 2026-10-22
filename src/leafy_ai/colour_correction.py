"""24-patch colour checker detection and 3x3 colour correction matrix per image.

Fitting happens in linear sRGB. Reference values come from colour-science's
``ColorChecker24 - After November 2014`` (X-Rite published xyY, D50), adapted
to D65 sRGB with Bradford. Requires the ``[image]`` extra for detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHECKER = "ColorChecker24 - After November 2014"


@dataclass(frozen=True)
class CCMFit:
    matrix: np.ndarray  # (3, 3); corrected = measured @ matrix
    rmse: float  # linear RGB residual on the 24 patches
    reversed_order: bool  # chart detected rotated 180 degrees


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def reference_linear_rgb() -> np.ndarray:
    """(24, 3) linear sRGB reference patch values, row-major from dark skin to black."""
    import colour

    checker = colour.CCS_COLOURCHECKERS[CHECKER]
    xyy = np.array(list(checker.data.values()))
    xyz = colour.xyY_to_XYZ(xyy)
    return colour.XYZ_to_RGB(
        xyz,
        colour.RGB_COLOURSPACES["sRGB"],
        illuminant=checker.illuminant,
        chromatic_adaptation_transform="Bradford",
    )


def fit_ccm(measured: np.ndarray, reference: np.ndarray) -> CCMFit:
    """Least-squares 3x3 matrix mapping measured linear patches to reference.

    Tries both patch orders because a chart rotated 180 degrees reverses them.
    """
    measured, reference = np.asarray(measured, float), np.asarray(reference, float)
    if measured.shape != (24, 3) or reference.shape != (24, 3):
        raise ValueError("Expected (24, 3) measured and reference patch arrays")
    if not np.isfinite(measured).all():
        raise ValueError("Measured patches must be finite")
    best = None
    for reverse in (False, True):
        m = measured[::-1] if reverse else measured
        matrix, *_ = np.linalg.lstsq(m, reference, rcond=None)
        rmse = float(np.sqrt(np.mean((m @ matrix - reference) ** 2)))
        if best is None or rmse < best.rmse:
            best = CCMFit(matrix, rmse, reverse)
    return best


def apply_ccm(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Correct an 8-bit sRGB image (H, W, 3); returns 8-bit sRGB."""
    linear = srgb_to_linear(np.asarray(image, float) / 255.0)
    corrected = linear @ np.asarray(matrix, float)
    return np.round(linear_to_srgb(corrected) * 255).astype(np.uint8)


def detect_patches(image: np.ndarray) -> np.ndarray | None:
    """Return (24, 3) linear patch means from an 8-bit sRGB image, or None if not found."""
    from colour_checker_detection import detect_colour_checkers_segmentation

    linear = srgb_to_linear(np.asarray(image, float) / 255.0)
    found = detect_colour_checkers_segmentation(linear)
    return np.asarray(found[0], float) if len(found) else None


def correct_image(path: str | Path, out_dir: str | Path) -> tuple[Path | None, CCMFit | None]:
    """Detect checker, fit CCM and save ``<stem>_cc.png``. Returns (None, None) if no checker.

    Images without a detected checker are left uncorrected and must be reported by QC.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"))
    patches = detect_patches(image)
    if patches is None:
        return None, None
    fit = fit_ccm(patches, reference_linear_rgb())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{Path(path).stem}_cc.png"
    Image.fromarray(apply_ccm(image, fit.matrix)).save(target)
    return target, fit
