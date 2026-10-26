"""Fold-local SHAP explanations for held-out plants in a saved tabular run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, source_hash, write_summary

from leafy_ai.experiment import ExperimentConfig, _prepare
from leafy_ai.explain import band_importance, explain_tabular, select_test_plants, shap_importance


class _FlatPredictor:
    """Expose multi-view fusion probability prediction to permutation SHAP."""

    def __init__(self, fitted, widths: list[int]):
        self.fitted, self.widths = fitted, widths

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        from leafy_ai.fusion_training import F3Classifier
        from leafy_ai.models import GroupedLateFusion

        if isinstance(self.fitted, (GroupedLateFusion, F3Classifier)):
            cuts = np.cumsum(self.widths)[:-1]
            return self.fitted.predict_proba(np.split(data, cuts, axis=1))
        return self.fitted.predict_proba(data)


def _figures(explanation, classes: list[str], out: Path, synthetic: bool) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    feature_values = np.asarray(explanation.data)
    importance = shap_importance(explanation, classes)
    importance.to_csv(out / "shap_importance.csv", index=False)
    global_order = (
        importance.groupby("feature").mean_abs_shap.mean().sort_values(ascending=False).head(20)
    )
    fig, ax = plt.subplots(figsize=(8, max(5, 0.27 * len(global_order))))
    ax.barh(global_order.index[::-1], global_order.values[::-1], color=PALETTE[0])
    ax.set(
        xlabel="Mean |SHAP| probability contribution", title="Global held-out feature importance"
    )
    save_figure(fig, out / "shap_global", synthetic)
    for class_index, name in enumerate(classes):
        sub = importance.loc[importance["class"] == name].set_index("feature")
        order = sub.mean_abs_shap.sort_values(ascending=False).head(20).index.tolist()
        fig, ax = plt.subplots(figsize=(8, max(5, 0.27 * len(order))))
        ax.barh(
            order[::-1],
            sub.loc[order[::-1], "mean_abs_shap"],
            color=PALETTE[class_index % len(PALETTE)],
        )
        ax.set(xlabel="Mean |SHAP| probability contribution", title=f"SHAP: {name}")
        save_figure(fig, out / f"shap_{name}_bar", synthetic)
        fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(order))))
        for position, feature in enumerate(order):
            index = list(explanation.feature_names).index(feature)
            jitter = np.random.default_rng(23 + position).uniform(-0.2, 0.2, len(values))
            ax.scatter(
                values[:, index, class_index],
                position + jitter,
                c=feature_values[:, index],
                cmap="viridis",
                s=18,
                alpha=0.8,
            )
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set(
            yticks=range(len(order)),
            yticklabels=order,
            xlabel="SHAP probability contribution",
            title=f"SHAP beeswarm: {name}",
        )
        save_figure(fig, out / f"shap_{name}_beeswarm", synthetic)
    bands = band_importance(importance)
    bands.to_csv(out / "shap_bands.csv", index=False)
    if not bands.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (quantity, group) in enumerate(bands.groupby("sensor_quantity")):
            spectrum = group.groupby("wavelength_nm").mean_abs_shap.mean().sort_index()
            ax.plot(
                spectrum.index,
                spectrum.values,
                marker="o",
                label=quantity,
                color=PALETTE[i % len(PALETTE)],
            )
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Mean |SHAP| probability contribution",
            title="Raw band importance",
        )
        ax.legend()
        save_figure(fig, out / "shap_wavelength", synthetic)
    return importance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--background-size", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reports/explain"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig.model_validate(run["config"])
    source = Path(cfg.input_file)
    if source_hash(source) != run["input_sha256"]:
        raise ValueError("Feature table differs from the saved experiment")
    synthetic = prepare_report(
        [args.run / "oof.parquet"], args.out, args.allow_synthetic, require_run=True
    )
    if synthetic != run["synthetic"]:
        raise ValueError("Run and OOF synthetic markers disagree")
    oof = read_table(args.run / "oof.parquet")
    selected = select_test_plants(oof, args.seed, args.fold, args.per_class)
    raw = read_table(source)
    data = _prepare(raw, cfg)
    test_ids = set(oof.loc[(oof.seed == args.seed) & (oof.fold == args.fold), "plant_id"])
    train = data.loc[~data.plant_id.isin(test_ids)]
    if cfg.split == "external_test":
        if (
            cfg.external_file is None
            or source_hash(Path(cfg.external_file)) != run["external_sha256"]
        ):
            raise ValueError("External test table differs from the saved experiment")
        test_data = _prepare(read_table(Path(cfg.external_file)), cfg)
    else:
        test_data = data
    test = test_data.set_index("plant_id").loc[selected.plant_id].reset_index()
    names = [name for cols in cfg.features.values() for name in cols]
    fitted = joblib.load(args.run / "models.joblib")[f"seed{args.seed}_fold{args.fold}"]
    predictor = _FlatPredictor(fitted, [len(cols) for cols in cfg.features.values()])
    explanation = explain_tabular(
        predictor,
        train[names],
        test[names],
        train.plant_id,
        test.plant_id,
        background_size=args.background_size,
    )
    classes = list(fitted.classes_)
    _figures(explanation, classes, args.out, synthetic)
    selected[["plant_id", "class", "seed", "fold"]].to_csv(
        args.out / "explained_plants.csv", index=False
    )
    write_summary(
        args.out / "explain_summary.md",
        "Held-out SHAP explanations",
        f"{len(selected)} plants from seed {args.seed}, fold {args.fold}; "
        f"{len(train)} disjoint training plants supplied the background. "
        "Permutation SHAP explains saved probability predictions; it does not establish causality.",
        synthetic,
    )


if __name__ == "__main__":
    main()
