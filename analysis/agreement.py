"""Blind symptom-rater Cohen kappa, resampling plants for confidence limits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import prepare_report, read_table, save_figure, write_summary
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def evaluate(
    frame: pd.DataFrame, day: int | None = None, reps: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One blinded pair per plant at a fixed day; no proxy-derived labels."""
    needed = {"plant_id", "day", "rater1_score", "rater2_score"}
    if needed - set(frame):
        raise ValueError(f"Missing columns {sorted(needed - set(frame))}")
    if day is None:
        if frame.day.nunique() != 1:
            raise ValueError("Select a fixed day for agreement")
        day = int(frame.day.iloc[0])
    data = frame.loc[frame.day == day].copy()
    if data.plant_id.duplicated().any():
        raise ValueError("One rater pair per plant at selected day required")
    missing = int(data[["rater1_score", "rater2_score"]].isna().any(axis=1).sum())
    data = data.dropna(subset=["rater1_score", "rater2_score"])
    if len(data) < 2 or reps < 100:
        raise ValueError("Insufficient paired plant scores or bootstrap reps")
    labels = sorted(set(data.rater1_score) | set(data.rater2_score))
    kappa = cohen_kappa_score(data.rater1_score, data.rater2_score, labels=labels)
    rng = np.random.default_rng(23)
    draws = []
    for _ in range(reps):
        sample = data.iloc[rng.choice(len(data), len(data), replace=True)]
        value = cohen_kappa_score(sample.rater1_score, sample.rater2_score, labels=labels)
        if np.isfinite(value):
            draws.append(value)
    low, high = np.quantile(draws, [0.025, 0.975])
    table = pd.DataFrame(
        [
            {
                "day": day,
                "n_plants": len(data),
                "missing_pairs": missing,
                "cohen_kappa": kappa,
                "ci_low": low,
                "ci_high": high,
                "valid_bootstraps": len(draws),
            }
        ]
    )
    matrix = pd.DataFrame(
        confusion_matrix(data.rater1_score, data.rater2_score, labels=labels),
        index=labels,
        columns=labels,
    )
    return table, matrix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--day", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/agreement"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report([args.scores], args.out, args.allow_synthetic)
    table, matrix = evaluate(read_table(args.scores), args.day, args.reps)
    table.to_csv(args.out / "agreement.csv", index=False)
    matrix.to_csv(args.out / "agreement_matrix.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set(
        xticks=range(len(matrix)),
        yticks=range(len(matrix)),
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
        xlabel="Rater 2 score",
        ylabel="Rater 1 score",
    )
    save_figure(fig, args.out / "agreement", synthetic)
    row = table.iloc[0]
    write_summary(
        args.out / "agreement_summary.md",
        "Rater agreement",
        f"Cohen κ={row.cohen_kappa:.3f} (plant bootstrap 95% CI "
        f"{row.ci_low:.3f} to {row.ci_high:.3f}), {row.n_plants} paired plants. "
        f"{row.missing_pairs} incomplete pairs excluded and reported.",
        synthetic,
    )


if __name__ == "__main__":
    main()
