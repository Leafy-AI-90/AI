"""Calibrated optical, thermal, positional and temporal features.

Inputs must follow the explicit wavelength order. Every equation here is listed
with unit, sensor and reference in :mod:`leafy_ai.registry` and ``docs/features.md``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import BANDS

INDEX_NAMES = ("ndvi", "ndre", "ci_red_edge", "gndvi", "pri_proxy")
SPECTRAL_NAMES = (
    *(f"r_{b}" for b in BANDS),
    *(f"dr_{a}_{b}" for a, b in zip(BANDS[:-1], BANDS[1:], strict=True)),
    *INDEX_NAMES,
    "a_660",
    "a_710",
    "a_940",
    "m660",
    "m710",
)


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
    """One row per plant/day; repeat-averaged oldest, youngest, lesion, off-lesion.

    Kept for backward compatibility; :func:`positional_contrasts` handles 3 or 4 positions.
    """
    m = finite(m710)
    if m.ndim != 2 or m.shape[1] != 4:
        raise ValueError("Expected four positional M710 values per plant/day")
    return np.column_stack((m[:, 0] - m[:, 1], m[:, 3] - m[:, 2], m.var(axis=1)))


def positional_contrasts(values, positions: Sequence[str]) -> dict[str, np.ndarray]:
    """Contrasts across leaf positions; ``values`` is (n, len(positions)).

    Returns only contrasts whose positions are configured:
    ``age`` = oldest - youngest, ``lesion`` = off_lesion - lesion,
    ``middle`` = middle - mean(oldest, youngest), ``variance`` = population variance.
    """
    v = finite(values)
    positions = tuple(positions)
    if v.ndim != 2 or v.shape[1] != len(positions) or len(positions) not in (3, 4):
        raise ValueError("Expected (n, 3 or 4) values matching positions")
    col = {p: v[:, i] for i, p in enumerate(positions)}
    out = {}
    if {"oldest", "youngest"} <= col.keys():
        out["age"] = col["oldest"] - col["youngest"]
        if "middle" in col:
            out["middle"] = col["middle"] - (col["oldest"] + col["youngest"]) / 2
    if {"lesion", "off_lesion"} <= col.keys():
        out["lesion"] = col["off_lesion"] - col["lesion"]
    out["variance"] = v.var(axis=1)
    return out


def water_index(a970, a940):
    """W970 = A970 - A940: liquid water absorption near 970 nm relative to 940 nm."""
    return finite(a970) - finite(a940)


def _nan_ok(value) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if np.isinf(a).any():
        raise ValueError("Infinite values are not valid measurements")
    return a


def delta_t(t_leaf, t_air):
    """Leaf minus air temperature (degrees C); missing input stays NaN."""
    return _nan_ok(t_leaf) - _nan_ok(t_air)


def cwsi(t_leaf, t_wet, t_dry):
    """Crop water stress index (T_leaf - T_wet) / (T_dry - T_wet); NaN if a reference is missing."""
    t_leaf, t_wet, t_dry = map(_nan_ok, (t_leaf, t_wet, t_dry))
    span = t_dry - t_wet
    if np.any(span[np.isfinite(span)] <= 0):
        raise ValueError("Dry reference must be warmer than wet reference")
    return (t_leaf - t_wet) / span


def temperature_spread(p90, p10):
    """Within-leaf temperature spread P90 - P10 (degrees C)."""
    spread = _nan_ok(p90) - _nan_ok(p10)
    if np.any(spread[np.isfinite(spread)] < 0):
        raise ValueError("P90 must not be below P10")
    return spread


def saturation_vapour_pressure(t_c):
    """Tetens equation as in FAO-56 eq. 11, kPa."""
    t = _nan_ok(t_c)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vpd(t_air, rh_pct):
    """Air vapour pressure deficit (kPa) = e_s(T_air) * (1 - RH / 100)."""
    rh = _nan_ok(rh_pct)
    if np.any((rh[np.isfinite(rh)] < 0) | (rh[np.isfinite(rh)] > 100)):
        raise ValueError("Relative humidity must be within 0 to 100 %")
    return saturation_vapour_pressure(t_air) * (1 - rh / 100)


def baseline_change(
    df: pd.DataFrame, columns: Sequence[str], baseline_day: int = -1
) -> pd.DataFrame:
    """``<col>_delta`` = value - same plant's value on ``baseline_day``; NaN if no baseline."""
    base = df.loc[df["day"] == baseline_day].set_index("plant_id")[list(columns)]
    if base.index.duplicated().any():
        raise ValueError("One baseline row per plant required")
    aligned = base.reindex(df["plant_id"]).to_numpy()
    return pd.DataFrame(
        df[list(columns)].to_numpy() - aligned,
        index=df.index,
        columns=[f"{c}_delta" for c in columns],
    )


def recent_slope(df: pd.DataFrame, columns: Sequence[str], window: int = 3) -> pd.DataFrame:
    """``<col>_slope``: OLS slope per day over this and previous measurement days.

    Uses the ``window`` most recent measurement days up to and including the
    current day of the same plant, so no future measurement leaks backwards.
    NaN when fewer than ``window`` days exist or any value in the window is missing.
    """
    if window < 2:
        raise ValueError("Slope needs a window of at least 2 days")
    out = np.full((len(df), len(columns)), np.nan)
    order = df.sort_values(["plant_id", "day"])
    position = {idx: i for i, idx in enumerate(df.index)}
    for _, group in order.groupby("plant_id", sort=False):
        days = group["day"].to_numpy(float)
        if len(np.unique(days)) != len(days):
            raise ValueError("One row per plant/day required")
        values = group[list(columns)].to_numpy(float)
        for end in range(window - 1, len(group)):
            x = days[end - window + 1 : end + 1]
            y = values[end - window + 1 : end + 1]
            xc = x - x.mean()
            slope = (xc[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (xc**2).sum()
            out[position[group.index[end]]] = slope
    return pd.DataFrame(out, index=df.index, columns=[f"{c}_slope" for c in columns])
