"""Corrected paired repeated-CV comparison; never ANOVA on fold accuracies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary

from leafy_ai.stats import nadeau_bengio


def evaluate(frame: pd.DataFrame, first: str, second: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require identical repeat/fold/test-plant hashes before paired inference."""
    needed = {"model", "repeat", "fold", "score", "test_plant_hash", "test_fraction"}
    if needed - set(frame):
        raise ValueError(f"Fold table missing {sorted(needed - set(frame))}")
    data = frame.loc[frame.model.isin([first, second])]
    if data.duplicated(["model", "repeat", "fold"]).any():
        raise ValueError("Duplicate model/repeat/fold score")
    a = data.loc[data.model == first].set_index(["repeat", "fold"]).sort_index()
    b = data.loc[data.model == second].set_index(["repeat", "fold"]).sort_index()
    if not a.index.equals(b.index) or not a.test_plant_hash.equals(b.test_plant_hash):
        raise ValueError("Models must use the same held-out plant sets")
    if not a.test_fraction.equals(b.test_fraction) or a.test_fraction.nunique() != 1:
        raise ValueError("Common test fraction required for corrected t test")
    differences = a.score - b.score
    test = nadeau_bengio(differences.to_numpy(), float(a.test_fraction.iloc[0]))
    table = pd.DataFrame(
        [
            {
                **test,
                "first": first,
                "second": second,
                "n_fold_scores": len(differences),
                "test_fraction": a.test_fraction.iloc[0],
            }
        ]
    )
    folds = differences.rename("difference").reset_index()
    return table, folds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-scores", type=Path, required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/repeated_cv"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report([args.fold_scores], args.out, args.allow_synthetic)
    table, folds = evaluate(read_table(args.fold_scores), args.first, args.second)
    table.to_csv(args.out / "repeated_cv.csv", index=False)
    folds.to_csv(args.out / "repeated_cv_folds.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(folds)), folds.difference, color=PALETTE[0])
    ax.axhline(0, color=PALETTE[1], linestyle="--")
    ax.set(xlabel="Paired repeat/fold", ylabel="Macro F1 difference", title="Corrected repeated CV")
    save_figure(fig, args.out / "repeated_cv", synthetic)
    row = table.iloc[0]
    write_summary(
        args.out / "repeated_cv_summary.md",
        "Repeated CV",
        f"Nadeau–Bengio corrected resampled t test: mean paired difference "
        f"{row.mean_difference:.3f}; p={row.p:.4g}. Fold scores share held-out plant hashes. "
        "Treat dependent fold scores as descriptive; primary endpoint remains held-out plants/cycle.",
        synthetic,
    )


if __name__ == "__main__":
    main()
