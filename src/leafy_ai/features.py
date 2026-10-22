"""Calibrated optical features. Inputs must follow the explicit wavelength order."""

import numpy as np

from . import BANDS


def finite(value):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Optical measurements must be finite")
    return a


def reflectance(leaf, dark, white, white_reflectance=1.0):
    leaf, dark, white = map(finite, (leaf, dark, white))
    denominator = white - dark
    if np.any(denominator <= 0):
        raise ValueError("White reference must exceed dark reference")
    return (leaf - dark) / denominator * finite(white_reflectance)


def absorbance(signal, empty_clip):
    """Both inputs must already have LED-off readings subtracted."""
    signal, empty_clip = map(finite, (signal, empty_clip))
    if np.any(signal <= 0) or np.any(empty_clip <= 0):
        raise ValueError("Dark-corrected transmission must be positive")
    return np.log10(empty_clip / signal)


def spectral_features(r, a):
    """Return 45 columns: 18 R, 17 derivatives/nm, 5 indices, 3 A, M660, M710.

    Reflectance uses BANDS; absorbance uses (660, 710, 940) nm.
    """
    r, a = finite(r), finite(a)
    if r.ndim != 2 or r.shape[1] != 18 or a.shape != (len(r), 3):
        raise ValueError("Expected reflectance (n,18) and absorbance (n,3)")
    if np.any(r < 0):
        raise ValueError("Negative calibrated reflectance requires measurement QC")
    band = {b: r[:, i] for i, b in enumerate(BANDS)}

    def divide(x, y):
        if np.any(np.abs(y) <= 1e-12):
            raise ValueError("Spectral index denominator is zero")
        return x / y

    def nd(x, y):
        return divide(band[x] - band[y], band[x] + band[y])

    indices = np.column_stack(
        (nd(810, 680), nd(760, 705), divide(band[760], band[705]) - 1, nd(810, 560), nd(535, 560))
    )
    return np.column_stack(
        (r, np.diff(r) / np.diff(BANDS), indices, a, a[:, 0] - a[:, 2], a[:, 1] - a[:, 2])
    )


def positional_features(m710):
    """One row per plant/day; repeat-averaged oldest, youngest, lesion, off-lesion."""
    m = finite(m710)
    if m.ndim != 2 or m.shape[1] != 4:
        raise ValueError("Expected four positional M710 values per plant/day")
    return np.column_stack((m[:, 0] - m[:, 1], m[:, 3] - m[:, 2], m.var(axis=1)))
