"""H3: plant-level key-pair confusion with and without added modalities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary
from sklearn.metrics import confusion_matrix

from leafy_ai.stats import confusion_pair, pair_oof, predictions


def evaluate(
    augmented: pd.DataFrame,
    baseline: pd.DataFrame,
    pairs: list[tuple[str, str]],
    seed: int | None = None,
    reps: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Report relative key-pair reduction with paired plant bootstrap CI."""
    if reps < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    a, b, classes = pair_oof(augmented, baseline, seed)
    rng = np.random.default_rng(23)
    groups = [np.flatnonzero(a["class"].to_numpy() == c) for c in classes]
    rows = []
    for left, right in pairs:
        if left not in classes or right not in classes:
            raise ValueError("Confusion pair must use study classes")
        before = confusion_pair(b, classes, left, right)
        after = confusion_pair(a, classes, left, right)
        changes = []
        for _ in range(reps):
            chosen = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            old = confusion_pair(b.iloc[chosen], classes, left, right)
            if old > 0:
                new = confusion_pair(a.iloc[chosen], classes, left, right)
                changes.append((old - new) / old)
        low, high = np.quantile(changes, [0.025, 0.975]) if changes else (np.nan, np.nan)
        rows.append(
            {
                "pair": f"{left}:{right}",
                "baseline_rate": before,
                "augmented_rate": after,
                "relative_reduction": (before - after) / before if before else np.nan,
                "ci_low": low,
                "ci_high": high,
                "valid_bootstraps": len(changes),
                "n_plants": len(a),
            }
        )
    truth = a["class"].to_numpy()
    baseline_cm = pd.DataFrame(
        confusion_matrix(truth, predictions(b, classes), labels=classes),
        index=classes,
        columns=classes,
    )
    augmented_cm = pd.DataFrame(
        confusion_matrix(truth, predictions(a, classes), labels=classes),
        index=classes,
        columns=classes,
    )
    return pd.DataFrame(rows), baseline_cm, augmented_cm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True, help="Class names joined by :")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/h3"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    pairs = [tuple(value.split(":")) for value in args.pair]
    if any(len(pair) != 2 for pair in pairs):
        parser.error("Each --pair must be class_a:class_b")
    synthetic = prepare_report(
        [args.augmented, args.baseline], args.out, args.allow_synthetic, require_run=True
    )
    table, baseline_cm, augmented_cm = evaluate(
        read_table(args.augmented), read_table(args.baseline), pairs, args.seed, args.reps
    )
    table.to_csv(args.out / "h3_confusion.csv", index=False)
    baseline_cm.to_csv(args.out / "h3_baseline_matrix.csv")
    augmented_cm.to_csv(args.out / "h3_augmented_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, cm, title in zip(
        axes, (baseline_cm, augmented_cm), ("Baseline", "Augmented"), strict=True
    ):
        ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0)
        for (row, column), value in np.ndenumerate(cm.to_numpy()):
            ax.text(column, row, str(value), ha="center", va="center", color=PALETTE[1])
        ax.set(
            xticks=range(len(cm)),
            yticks=range(len(cm)),
            xticklabels=cm.columns,
            yticklabels=cm.index,
            xlabel="Predicted class",
            ylabel="True class",
            title=title,
        )
        ax.tick_params(axis="x", rotation=45)
    save_figure(fig, args.out / "h3_confusion", synthetic)
    status = " and ".join(
        f"{row.pair}: {row.relative_reduction:.1%} reduction"
        if np.isfinite(row.relative_reduction)
        else f"{row.pair}: undefined (zero baseline errors)"
        for row in table.itertuples()
    )
    write_summary(
        args.out / "h3_summary.md",
        "H3 confusion",
        f"Paired held-out plants; {status}. Target: ≥30% relative reduction. "
        "Intervals resample whole plants within class.",
        synthetic,
    )


if __name__ == "__main__":
    main()
