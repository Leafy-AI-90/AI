"""Plant random-intercept mixed models for preregistered physiology features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import PALETTE, prepare_report, read_table, save_figure, write_summary

from leafy_ai.stats import holm


def evaluate(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit feature ~ class × day + (1 | plant) without filling missing values."""
    needed = {"plant_id", "day", "class", *features}
    if needed - set(frame):
        raise ValueError(f"Missing columns {sorted(needed - set(frame))}")
    if frame.duplicated(["plant_id", "day"]).any():
        raise ValueError("Aggregate technical repeats to one plant-day first")
    rows, missing = [], []
    for feature in features:
        data = frame[["plant_id", "day", "class", feature]].copy()
        count = int(data[feature].isna().sum())
        missing.append(
            {
                "feature": feature,
                "missing_plant_days": count,
                "observed_plant_days": len(data) - count,
            }
        )
        data = data.dropna().rename(columns={"class": "treatment", feature: "value"})
        if data.plant_id.nunique() < 3 or data.treatment.nunique() < 2:
            raise ValueError(f"Insufficient measured plants/classes for {feature}")
        fitted = smf.mixedlm("value ~ C(treatment) * day", data, groups=data["plant_id"]).fit(
            reml=False, method="lbfgs"
        )
        for term, estimate in fitted.params.items():
            rows.append(
                {
                    "feature": feature,
                    "term": term,
                    "estimate": estimate,
                    "se": fitted.bse[term],
                    "p": fitted.pvalues[term],
                    "n_plants": data.plant_id.nunique(),
                    "n_plant_days": len(data),
                    "converged": bool(fitted.converged),
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    tests = table.term.str.contains("treatment")
    table.loc[tests, "p_holm"] = holm(table.loc[tests, "p"].to_numpy())
    return table, pd.DataFrame(missing)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-table", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/mixed_models"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report([args.features_table], args.out, args.allow_synthetic)
    source = read_table(args.features_table)
    table, missing = evaluate(source, args.feature)
    table.to_csv(args.out / "mixed_models.csv", index=False)
    missing.to_csv(args.out / "mixed_models_missing.csv", index=False)
    fig, axes = plt.subplots(
        len(args.feature), 1, figsize=(7, 4 * len(args.feature)), squeeze=False
    )
    for feature, ax in zip(args.feature, axes[:, 0], strict=True):
        summary = source.groupby(["class", "day"], observed=True)[feature].mean().reset_index()
        for i, (name, group) in enumerate(summary.groupby("class")):
            ax.plot(
                group.day, group[feature], marker="o", color=PALETTE[i % len(PALETTE)], label=name
            )
        ax.set(
            xlabel="Day from treatment", ylabel=feature, title=f"Observed plant means: {feature}"
        )
        ax.legend()
    save_figure(fig, args.out / "mixed_models", synthetic)
    write_summary(
        args.out / "mixed_models_summary.md",
        "Mixed models",
        "Plant random intercept with fixed class × day effects, maximum likelihood. "
        "Holm adjustment covers class and interaction terms across nominated features. "
        "Missing observations are counted and never imputed; figures show observed means.",
        synthetic,
    )


if __name__ == "__main__":
    main()
