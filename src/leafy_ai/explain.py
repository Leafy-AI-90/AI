"""Held-out SHAP and Grad-CAM explanations with plant-level provenance."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


def select_test_plants(
    oof: pd.DataFrame, seed: int, fold: int, per_class: int = 20, random_seed: int = 23
) -> pd.DataFrame:
    """Sample at most 20 distinct held-out plants from each treatment class."""
    needed = {"plant_id", "class", "seed", "fold"}
    if needed - set(oof):
        raise ValueError("OOF needs plant_id, class, seed, fold")
    if per_class < 1:
        raise ValueError("per_class must be positive")
    data = oof.loc[(oof.seed == seed) & (oof.fold == fold)]
    if data.empty or data.plant_id.duplicated().any():
        raise ValueError("Expected one held-out row per plant in selected fold")
    rng = np.random.default_rng(random_seed)
    choices = []
    for _, group in data.groupby("class", sort=True):
        chosen = rng.choice(group.index.to_numpy(), min(per_class, len(group)), replace=False)
        choices.extend(chosen.tolist())
    return data.loc[choices].sort_values(["class", "plant_id"]).reset_index(drop=True)


def explain_tabular(
    model,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_plants: Sequence[str],
    test_plants: Sequence[str],
    *,
    background_size: int = 20,
    seed: int = 23,
):
    """Permutation SHAP of a frozen fold model using training plants as background."""
    if set(train_plants) & set(test_plants):
        raise ValueError("SHAP background and explained plants must be disjoint")
    if len(train_features) != len(train_plants) or len(test_features) != len(test_plants):
        raise ValueError("One feature row per listed plant required")
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Training and test feature columns must align")
    if (
        not np.isfinite(train_features.to_numpy(dtype=float)).all()
        or not np.isfinite(test_features.to_numpy(dtype=float)).all()
    ):
        raise ValueError("SHAP features must be finite; no imputation")
    if background_size < 1:
        raise ValueError("background_size must be positive")
    import shap

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(train_features), min(background_size, len(train_features)), replace=False
    )
    background = train_features.iloc[chosen].to_numpy(dtype=float)
    explainer = shap.Explainer(
        model.predict_proba,
        background,
        algorithm="permutation",
        feature_names=train_features.columns.tolist(),
    )
    return explainer(test_features.to_numpy(dtype=float), max_evals=2 * train_features.shape[1] + 1)


def shap_importance(explanation, classes: Sequence[str]) -> pd.DataFrame:
    """Mean absolute SHAP score per feature and class."""
    values = np.asarray(explanation.values)
    if values.ndim != 3 or values.shape[2] != len(classes):
        raise ValueError("Expected sample × feature × class SHAP values")
    names = list(explanation.feature_names)
    return pd.DataFrame(
        [
            {
                "class": name,
                "feature": feature,
                "mean_abs_shap": float(np.abs(values[:, j, i]).mean()),
            }
            for i, name in enumerate(classes)
            for j, feature in enumerate(names)
        ]
    )


def band_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Keep raw reflectance and absorbance features with parsed wavelength."""
    data = importance.copy()
    parsed = data.feature.str.extract(r"^(r|a)_(\d{3})$")
    data["sensor_quantity"] = parsed[0].map({"r": "reflectance", "a": "absorbance"})
    data["wavelength_nm"] = pd.to_numeric(parsed[1], errors="coerce")
    return data.dropna(subset=["wavelength_nm"])


def gradcam_map(model, image, target_layer, target_class: int) -> np.ndarray:
    """Compute normalized Grad-CAM for one image and one predicted class."""
    from torch.nn import functional as torch_f

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single NCHW image")
    captured = {}

    def capture(_module, _input, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if target_class < 0 or target_class >= logits.shape[1]:
            raise ValueError("Target class outside model output")
        logits[0, target_class].backward()
        activation, gradient = captured["activation"], captured["gradient"]
        if activation.ndim != 4:
            raise ValueError("Grad-CAM target layer must have spatial activations")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch_f.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch_f.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        heat = cam[0, 0].detach().cpu().numpy()
        maximum = float(heat.max())
        return heat / maximum if maximum > 0 else np.zeros_like(heat)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def attention_inside_mask(heatmap: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Fraction of positive Grad-CAM intensity inside a registered leaf mask."""
    heat = np.asarray(heatmap, dtype=float)
    mask = np.asarray(leaf_mask, dtype=bool)
    if heat.shape != mask.shape or heat.ndim != 2 or not np.isfinite(heat).all():
        raise ValueError("Heatmap and leaf mask must be finite aligned 2-D arrays")
    if (heat < 0).any():
        raise ValueError("Grad-CAM heatmap must be nonnegative")
    total = heat.sum()
    return float(heat[mask].sum() / total) if total > 0 else float("nan")


def band_wavelength(name: str) -> int | None:
    """Return wavelength only for raw reflectance/absorbance feature names."""
    match = re.fullmatch(r"[ra]_(\d{3})", name)
    return int(match.group(1)) if match else None
