"""Build one feature row per plant/day from validated (and optionally QC'd) tables.

Steps: dark-correct and calibrate each reading against its session references,
average repeats per leaf position, compute spectral features per position,
average positions, add positional contrasts, thermal features and temporal
features. QC-excluded readings are left out of averages and counted; missing
inputs give NaN, never imputed values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import BANDS
from .config import StudyConfig
from .features import (
    SPECTRAL_NAMES,
    absorbance,
    baseline_change,
    cwsi,
    delta_t,
    positional_contrasts,
    recent_slope,
    reflectance,
    spectral_features,
    temperature_spread,
    vpd,
    water_index,
)

KEYS = ["plant_id", "day"]
SLOPE_WINDOW = 3


def _session_references(references: pd.DataFrame) -> pd.DataFrame:
    """Mean of every reference column per session (start and end readings)."""
    numeric = references.drop(columns="timestamp")
    return numeric.groupby("session_id").mean().add_prefix("ref_").reset_index()


def calibrate_readings(
    clip: pd.DataFrame, references: pd.DataFrame, cfg: StudyConfig
) -> pd.DataFrame:
    """Reflectance per band and absorbance per LED for every reading."""
    df = clip.merge(
        _session_references(references), on="session_id", how="left", validate="many_to_one"
    )
    out = df[[*KEYS, "leaf_position", "repeat"]].copy()
    for band in cfg.spectral.bands_nm:
        leaf = df[f"as_{band}"] - df[f"dark_{band}"]
        white = df[f"ref_white_{band}"] - df[f"ref_dark_{band}"]
        out[f"r_{band}"] = reflectance(leaf, 0, white, df[f"ref_tile_r_{band}"])
    for led in cfg.transmission.leds_nm:
        signal = df[f"led{led}_on"] - df[f"led{led}_off"]
        empty = df[f"ref_empty{led}_on"] - df[f"ref_empty{led}_off"]
        out[f"a_{led}"] = absorbance(signal, empty)
    return out


def position_features(calibrated: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Repeat-averaged spectral features per plant/day/position."""
    keys = [*KEYS, "leaf_position"]
    means = calibrated.groupby(keys).mean(numeric_only=True).drop(columns="repeat")
    counts = calibrated.groupby(keys).size().rename("n_repeats")
    r = means[[f"r_{b}" for b in BANDS]].to_numpy()
    a = means[["a_660", "a_710", "a_940"]].to_numpy()
    out = pd.DataFrame(spectral_features(r, a), index=means.index, columns=list(SPECTRAL_NAMES))
    if 970 in cfg.transmission.leds_nm:
        out["w970"] = water_index(means["a_970"], means["a_940"])
    return out.join(counts).reset_index()


def thermal_features(thermal: pd.DataFrame) -> pd.DataFrame:
    out = thermal[KEYS].copy()
    out["thermal_dt"] = delta_t(thermal["leaf_mean_c"], thermal["air_temp_c"])
    out["thermal_cwsi"] = cwsi(thermal["leaf_mean_c"], thermal["wet_ref_c"], thermal["dry_ref_c"])
    out["thermal_spread"] = temperature_spread(thermal["leaf_p90_c"], thermal["leaf_p10_c"])
    out["vpd_kpa"] = vpd(thermal["air_temp_c"], thermal["rh_pct"])
    return out


def build_features(
    tables: dict[str, pd.DataFrame], cfg: StudyConfig, readings: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Plant/day feature table. Pass QC'd ``readings`` to drop excluded rows from averages."""
    if tuple(cfg.spectral.bands_nm) != BANDS:
        raise ValueError("Spectral features require all 18 AS7265x bands")
    clip = tables["clip_readings"] if readings is None else readings
    n_excluded = pd.Series(0, index=pd.MultiIndex.from_frame(clip[KEYS].drop_duplicates()))
    if "excluded" in clip:
        n_excluded = clip.groupby(KEYS)["excluded"].sum().astype(int)
        clip = clip[~clip["excluded"].astype(bool)]
    per_position = position_features(calibrate_readings(clip, tables["references"], cfg), cfg)
    spectral = [*SPECTRAL_NAMES] + (["w970"] if "w970" in per_position else [])
    plant_day = per_position.groupby(KEYS)[spectral].mean()
    plant_day["n_positions"] = per_position.groupby(KEYS).size()
    plant_day["n_readings"] = per_position.groupby(KEYS)["n_repeats"].sum()
    plant_day["n_excluded_readings"] = n_excluded.reindex(plant_day.index).fillna(0).astype(int)
    m710 = per_position.pivot_table(index=KEYS, columns="leaf_position", values="m710")
    m710 = m710.reindex(index=plant_day.index, columns=list(cfg.leaf_positions))
    complete = m710.notna().all(axis=1).to_numpy()
    for name, values in positional_contrasts(m710.fillna(0).to_numpy(), cfg.leaf_positions).items():
        plant_day[f"pos_m710_{name}"] = np.where(complete, values, np.nan)
    table = plant_day.reset_index()
    plants = tables["plants"][["plant_id", "cycle", "class"]]
    table = plants.merge(table, on="plant_id", how="right", validate="one_to_many")
    if cfg.sensors.thermal:
        table = table.merge(
            thermal_features(tables["thermal"]), on=KEYS, how="left", validate="one_to_one"
        )
    table = table.sort_values(KEYS, ignore_index=True)
    base = [
        c for c in table.columns if c not in (*KEYS, "cycle", "class") and not c.startswith("n_")
    ]
    temporal = [
        baseline_change(table, base, cfg.design.days.first),
        recent_slope(table, base, SLOPE_WINDOW),
    ]
    return pd.concat([table, *temporal], axis=1)
