"""Plant-level H1 chlorophyll calibration and laboratory extraction equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .config import StudyConfig
from .validation import grouped_folds


@dataclass(frozen=True)
class Chlorophyll:
    """Extracted chlorophyll per sampled leaf area, in µg/cm²."""

    a: float
    b: float
    total: float


def lichtenthaler(
    absorbance_high: float,
    absorbance_low: float,
    *,
    solvent: str,
    extraction_volume_ml: float,
    dilution_factor: float,
    leaf_area_cm2: float,
    path_length_cm: float = 1.0,
) -> Chlorophyll:
    """Convert blank-corrected extract OD to µg/cm² using Lichtenthaler–Buschmann.

    80% acetone uses 663.2/646.8 nm; 95% ethanol uses 664.1/648.6 nm.
    Concentration coefficients are µg/mL for a 1 cm optical path.
    """
    values = np.asarray(
        [
            absorbance_high,
            absorbance_low,
            extraction_volume_ml,
            dilution_factor,
            leaf_area_cm2,
            path_length_cm,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or np.any(values[:2] < 0) or np.any(values[2:] <= 0):
        raise ValueError("Finite nonnegative absorbances and positive volumes/factors are required")
    if solvent == "acetone_80":
        a = 12.25 * absorbance_high - 2.79 * absorbance_low
        b = 21.50 * absorbance_low - 5.10 * absorbance_high
    elif solvent == "ethanol_95":
        a = 13.36 * absorbance_high - 5.19 * absorbance_low
        b = 27.43 * absorbance_low - 8.12 * absorbance_high
    else:
        raise ValueError("solvent must be acetone_80 or ethanol_95")
    factor = extraction_volume_ml * dilution_factor / (leaf_area_cm2 * path_length_cm)
    return Chlorophyll(a * factor, b * factor, (a + b) * factor)


def h1_columns(cfg: StudyConfig) -> list[str]:
    """The pre-specified 18 reflectances and available transmission absorbances."""
    return [
        *(f"r_{b}" for b in cfg.spectral.bands_nm),
        *(f"a_{b}" for b in cfg.transmission.leds_nm),
    ]


def prepare_h1(
    features: pd.DataFrame, ground_truth: pd.DataFrame, cfg: StudyConfig, day: int | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select one measured endpoint row per independent calibration plant.

    Missing biology and optical values remain missing in source data; selection counts
    are returned for the report. Treatment labels are never derived from optical data.
    """
    day = cfg.design.endpoint_day if day is None else day
    columns = h1_columns(cfg)
    needed = ["plant_id", "day", "class", "m660", "ci_red_edge", *columns]
    truth = ["plant_id", "day", "chl_total_ug_cm2", "extraction_solvent"]
    for name, frame, required in (
        ("features", features, needed),
        ("ground_truth", ground_truth, truth),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame.duplicated(["plant_id", "day"]).any():
            raise ValueError(f"{name} has duplicate plant/day rows")
    selected = features.loc[(features["day"] == day) & (features["class"] == "calibration"), needed]
    merged = selected.merge(
        ground_truth[truth], on=["plant_id", "day"], how="left", validate="one_to_one"
    )
    if merged["plant_id"].duplicated().any():
        raise ValueError("H1 requires one row per independent plant")
    report = {"calibration_plants_at_day": len(merged)}
    report["missing_extracted_chlorophyll"] = int(merged["chl_total_ug_cm2"].isna().sum())
    report["wrong_or_missing_solvent"] = int(
        merged["extraction_solvent"].ne(cfg.calibration.solvent).sum()
    )
    report["missing_optical_features"] = int(
        merged[["m660", "ci_red_edge", *columns]].isna().any(axis=1).sum()
    )
    usable = merged.loc[
        merged["chl_total_ug_cm2"].notna()
        & merged["extraction_solvent"].eq(cfg.calibration.solvent)
        & merged[["m660", "ci_red_edge", *columns]].notna().all(axis=1)
    ].copy()
    report["usable_plants"] = len(usable)
    if len(usable) < cfg.calibration.outer_folds * 2:
        raise ValueError(f"Too few measured calibration plants: {report}")
    return usable.reset_index(drop=True), report


def _choose_components(x: np.ndarray, y: np.ndarray, plants: np.ndarray, cfg: StudyConfig) -> int:
    splits = list(
        grouped_folds(np.zeros(len(y), dtype=int), plants, cfg.calibration.inner_folds, 42)
    )
    maximum = min(
        cfg.calibration.max_pls_components, x.shape[1], min(len(train) - 1 for train, _ in splits)
    )
    if maximum < 1:
        raise ValueError("Insufficient independent plants for inner PLS selection")
    losses = []
    for n in range(1, maximum + 1):
        fold_loss = []
        for train, test in splits:
            model = PLSRegression(n_components=n).fit(x[train], y[train])
            fold_loss.extend((model.predict(x[test]).ravel() - y[test]) ** 2)
        losses.append(np.mean(fold_loss))
    return int(np.argmin(losses) + 1)


def vip_scores(model: PLSRegression) -> np.ndarray:
    """Variable importance in projection for a single-response PLSR model."""
    weights = model.x_weights_
    scores = model.x_scores_
    loadings = model.y_loadings_.ravel()
    contribution = (scores**2).sum(axis=0) * loadings**2
    denominator = contribution.sum()
    if denominator <= 0:
        return np.full(weights.shape[0], np.nan)
    normalised = weights**2 / (weights**2).sum(axis=0)
    return np.sqrt(weights.shape[0] * (normalised @ contribution) / denominator)


def calibration_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Plant-level prediction and agreement metrics; difference is predicted minus measured."""
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    if measured.shape != predicted.shape or measured.ndim != 1 or len(measured) < 3:
        raise ValueError("At least three paired plant-level measurements are required")
    if not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite measurements and predictions")
    rmse = float(np.sqrt(mean_squared_error(measured, predicted)))
    diff = predicted - measured
    cov = float(np.cov(measured, predicted, ddof=0)[0, 1])
    concordance = (
        2
        * cov
        / (
            np.var(measured, ddof=0)
            + np.var(predicted, ddof=0)
            + (measured.mean() - predicted.mean()) ** 2
        )
    )
    bias = float(diff.mean())
    spread = float(diff.std(ddof=1))
    return {
        "r2": float(r2_score(measured, predicted)),
        "rmse_ug_cm2": rmse,
        "rpd": float(measured.std(ddof=1) / rmse) if rmse else float("inf"),
        "lin_ccc": float(concordance),
        "ba_bias_ug_cm2": bias,
        "ba_lower_ug_cm2": bias - 1.96 * spread,
        "ba_upper_ug_cm2": bias + 1.96 * spread,
    }


def evaluate_h1(
    data: pd.DataFrame, cfg: StudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested plant-grouped OOF predictions for PLSR and two fold-local baselines."""
    x = data[h1_columns(cfg)].to_numpy(float)
    y = data["chl_total_ug_cm2"].to_numpy(float)
    plants = data["plant_id"].to_numpy()
    prediction = {name: np.full(len(data), np.nan) for name in ("plsr", "m660", "ci_red_edge")}
    components, vip = [], []
    for train, test in grouped_folds(
        np.zeros(len(y), dtype=int), plants, cfg.calibration.outer_folds, 42
    ):
        n = _choose_components(x[train], y[train], plants[train], cfg)
        pls = PLSRegression(n_components=n).fit(x[train], y[train])
        prediction["plsr"][test] = pls.predict(x[test]).ravel()
        components.append(n)
        vip.append(vip_scores(pls))
        for name in ("m660", "ci_red_edge"):
            col = data[name].to_numpy(float).reshape(-1, 1)
            linear = LinearRegression().fit(col[train], y[train])
            prediction[name][test] = linear.predict(col[test])
    oof = data[["plant_id", "day", "chl_total_ug_cm2"]].copy()
    for name, values in prediction.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        oof[f"pred_{name}"] = values
    metrics = pd.DataFrame(
        [
            {"model": name, "n_plants": len(data), **calibration_metrics(y, values)}
            for name, values in prediction.items()
        ]
    )
    importance = pd.DataFrame(
        {
            "feature": h1_columns(cfg),
            "vip_mean": np.mean(vip, axis=0),
            "vip_sd": np.std(vip, axis=0, ddof=1),
        }
    )
    metrics.attrs["outer_pls_components"] = components
    return metrics, oof, importance
