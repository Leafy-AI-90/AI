"""Plant-level paired inference for held-out research predictions.

Each row is one plant for one seed and endpoint. Functions reject duplicate
plants so repeated leaves and measurements cannot inflate sample size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm
from scipy.stats import t as student_t
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


def validate_oof(frame: pd.DataFrame, seed: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Validate one held-out prediction per plant and select one seed."""
    data = frame.copy()
    required = {"plant_id", "class", "cycle", "day", "seed"}
    if missing := required - set(data):
        raise ValueError(f"OOF missing columns: {sorted(missing)}")
    seeds = data.seed.unique()
    if seed is None:
        if len(seeds) != 1:
            raise ValueError(
                "Select one seed explicitly; repeated seeds are not independent plants"
            )
        seed = seeds[0]
    data = data.loc[data.seed == seed].copy()
    if data.empty or data.plant_id.isna().any() or data.plant_id.duplicated().any():
        raise ValueError("Expected one nonmissing held-out row per plant")
    classes = [column[2:] for column in data if column.startswith("p_")]
    if len(classes) < 2 or not set(data["class"]).issubset(classes):
        raise ValueError("Probability columns must cover every true class")
    probabilities = data[[f"p_{name}" for name in classes]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Per-plant probabilities must sum to one")
    return data.sort_values("plant_id").reset_index(drop=True), classes


def pair_oof(
    first: pd.DataFrame, second: pd.DataFrame, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align two model runs on identical held-out plants and ground truth."""
    a, classes = validate_oof(first, seed)
    b, other_classes = validate_oof(second, seed)
    if set(classes) != set(other_classes) or set(a.plant_id) != set(b.plant_id):
        raise ValueError("Paired runs need identical classes and plant IDs")
    b = b.set_index("plant_id").loc[a.plant_id].reset_index()
    if not a[["class", "cycle", "day"]].equals(b[["class", "cycle", "day"]]):
        raise ValueError("Paired runs disagree on plant labels, cycle, or endpoint day")
    return a, b, classes


def predictions(frame: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    """Map probabilities to class names in an explicit shared order."""
    p = frame[[f"p_{name}" for name in classes]].to_numpy(dtype=float)
    return np.asarray(classes)[p.argmax(axis=1)]


def classification_metrics(frame: pd.DataFrame, classes: Sequence[str]) -> dict[str, float]:
    """Macro metrics and one-versus-rest AUROC for unique held-out plants."""
    y = frame["class"].to_numpy()
    pred = predictions(frame, classes)
    p = frame[[f"p_{name}" for name in classes]].to_numpy(dtype=float)
    result = {
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    for i, name in enumerate(classes):
        result[f"recall_{name}"] = float(
            recall_score(y, pred, labels=[name], average="macro", zero_division=0)
        )
        binary = y == name
        result[f"auroc_{name}"] = (
            float(roc_auc_score(binary, p[:, i])) if binary.any() and (~binary).any() else np.nan
        )
    result["macro_auroc"] = float(np.nanmean([result[f"auroc_{c}"] for c in classes]))
    return result


def plant_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    classes: Sequence[str],
    statistic: Callable[[pd.DataFrame, pd.DataFrame], float],
    reps: int = 2000,
    seed: int = 23,
) -> tuple[float, float, float]:
    """Paired stratified cluster bootstrap, resampling whole plants by class."""
    if reps < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if first.plant_id.duplicated().any() or second.plant_id.duplicated().any():
        raise ValueError("Bootstrap unit must be plant_id")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(first["class"].to_numpy() == c) for c in classes]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every study class needs held-out plants")
    point = float(statistic(first, second))
    if not np.isfinite(point):
        raise ValueError("Statistic is undefined for the observed paired plants")
    values = np.empty(reps)
    for i in range(reps):
        chosen = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values[i] = statistic(first.iloc[chosen], second.iloc[chosen])
    if not np.isfinite(values).all():
        raise ValueError("Statistic is undefined in bootstrap samples; report this limitation")
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def mcnemar(first_correct: np.ndarray, second_correct: np.ndarray) -> dict[str, float | int | str]:
    """Paired McNemar test, exact for fewer than 25 discordant plants."""
    a, b = np.asarray(first_correct, bool), np.asarray(second_correct, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Correctness arrays must be paired plant vectors")
    first_only = int(np.sum(a & ~b))
    second_only = int(np.sum(~a & b))
    discordant = first_only + second_only
    if discordant < 25:
        p = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
        method = "exact binomial"
    else:
        p = float(chi2.sf((abs(first_only - second_only) - 1) ** 2 / discordant, 1))
        method = "continuity-corrected chi-square"
    return {"first_only": first_only, "second_only": second_only, "p": p, "method": method}


def _delong_covariance(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC vector and paired covariance using positive-negative placement values."""
    positive = scores[:, y]
    negative = scores[:, ~y]
    if positive.shape[1] < 2 or negative.shape[1] < 2:
        raise ValueError("DeLong requires at least two positive and two negative plants")
    pair = (positive[:, :, None] > negative[:, None, :]).astype(float)
    pair += 0.5 * (positive[:, :, None] == negative[:, None, :])
    v_pos = pair.mean(axis=2)
    v_neg = pair.mean(axis=1)
    auc = pair.mean(axis=(1, 2))
    cov = np.cov(v_pos) / positive.shape[1] + np.cov(v_neg) / negative.shape[1]
    return auc, np.atleast_2d(cov)


def delong_paired(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Two-sided paired DeLong test for binary one-versus-rest AUROC."""
    binary = np.asarray(y, bool)
    scores = np.stack([np.asarray(first, float), np.asarray(second, float)])
    if scores.shape[1] != len(binary):
        raise ValueError("Scores must align one-to-one with plants")
    auc, cov = _delong_covariance(binary, scores)
    variance = max(float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]), 0.0)
    difference = float(auc[1] - auc[0])
    p = float(2 * norm.sf(abs(difference) / np.sqrt(variance))) if variance > 0 else 1.0
    return {
        "auc_first": float(auc[0]),
        "auc_second": float(auc[1]),
        "difference": difference,
        "p": p,
    }


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values, preserving input order."""
    p = np.asarray(p_values, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def confusion_pair(frame: pd.DataFrame, classes: Sequence[str], a: str, b: str) -> float:
    """Fraction of plants from a key pair confused with the other class."""
    true = frame["class"].to_numpy()
    pred = predictions(frame, classes)
    relevant = np.isin(true, [a, b])
    if not relevant.any():
        raise ValueError("Key confusion pair has no held-out plants")
    return float((((true == a) & (pred == b)) | ((true == b) & (pred == a))).sum() / relevant.sum())


def nadeau_bengio(differences: Sequence[float], test_fraction: float) -> dict[str, float]:
    """Corrected resampled t test for paired repeated-fold scores.

    The variance correction is 1/n + n_test/n_train; fold scores must share
    identical splits. This is descriptive when splits are cycle based.
    """
    d = np.asarray(differences, float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite paired fold differences required")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    variance = d.var(ddof=1) * (1 / len(d) + test_fraction / (1 - test_fraction))
    if np.isclose(variance, 0, atol=1e-15):
        variance = 0.0
    t_value = float(d.mean() / np.sqrt(variance)) if variance > 0 else 0.0
    return {
        "mean_difference": float(d.mean()),
        "corrected_se": float(np.sqrt(variance)),
        "t": t_value,
        "df": len(d) - 1,
        "p": float(2 * student_t.sf(abs(t_value), len(d) - 1)) if variance > 0 else 1.0,
    }
