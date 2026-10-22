"""H5: root-disease leaf-only versus leaf-plus-root held-out macro F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary

from leafy_ai.stats import classification_metrics, pair_oof, plant_bootstrap


def evaluate(
    leaf: pd.DataFrame, root: pd.DataFrame, seed: int | None = None, reps: int = 2000
) -> pd.DataFrame:
    """Ratio and paired plant bootstrap CI; root-disease class required."""
    a, b, classes = pair_oof(leaf, root, seed)
    if "root_disease" not in classes:
        raise ValueError("H5 is defined only for the root_disease preset")

    def ratio(x: pd.DataFrame, y: pd.DataFrame) -> float:
        denominator = classification_metrics(y, classes)["macro_f1"]
        return (
            classification_metrics(x, classes)["macro_f1"] / denominator if denominator else np.nan
        )

    point, low, high = plant_bootstrap(a, b, classes, ratio, reps=reps)
    return pd.DataFrame(
        [
            {
                "n_plants": len(a),
                "leaf_macro_f1": classification_metrics(a, classes)["macro_f1"],
                "leaf_root_macro_f1": classification_metrics(b, classes)["macro_f1"],
                "ratio": point,
                "ci_low": low,
                "ci_high": high,
                "target_met": bool(point >= 0.9),
            }
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf", type=Path, required=True)
    parser.add_argument("--leaf-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/h5"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report(
        [args.leaf, args.leaf_root], args.out, args.allow_synthetic, require_run=True
    )
    table = evaluate(read_table(args.leaf), read_table(args.leaf_root), args.seed, args.reps)
    table.to_csv(args.out / "h5_root_vs_leaf.csv", index=False)
    row = table.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Leaf only", "Leaf + root"], [row.leaf_macro_f1, row.leaf_root_macro_f1], color=PALETTE[:2]
    )
    ax.set(ylabel="Held-out plant macro F1", ylim=(0, 1), title="H5: root image contribution")
    save_figure(fig, args.out / "h5_root_vs_leaf", synthetic)
    write_summary(
        args.out / "h5_summary.md",
        "H5 root versus leaf",
        f"Leaf-only / leaf-plus-root macro F1 = {row.ratio:.3f} "
        f"(paired plant bootstrap 95% CI {row.ci_low:.3f} to {row.ci_high:.3f}). "
        f"The ≥0.90 target is {'met' if row.target_met else 'not met'}.",
        synthetic,
    )


if __name__ == "__main__":
    main()
