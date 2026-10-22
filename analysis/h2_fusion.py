"""H2: compare preselected fused and best single-modality held-out runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary

from leafy_ai.stats import (
    classification_metrics,
    delong_paired,
    holm,
    mcnemar,
    pair_oof,
    plant_bootstrap,
    predictions,
)


def evaluate(
    fused: pd.DataFrame, single: pd.DataFrame, seed: int | None = None, reps: int = 2000
) -> pd.DataFrame:
    """Return one paired, plant-level H2 table."""
    a, b, classes = pair_oof(fused, single, seed)
    ma, mb = classification_metrics(a, classes), classification_metrics(b, classes)
    point, low, high = plant_bootstrap(
        a,
        b,
        classes,
        lambda x, y: classification_metrics(x, classes)["macro_f1"]
        - classification_metrics(y, classes)["macro_f1"],
        reps=reps,
    )
    correct_a = predictions(a, classes) == a["class"].to_numpy()
    correct_b = predictions(b, classes) == b["class"].to_numpy()
    mc = mcnemar(correct_a, correct_b)
    rows = []
    for metric in ma:
        rows.append(
            {
                "metric": metric,
                "fused": ma[metric],
                "single": mb[metric],
                "difference": ma[metric] - mb[metric],
                "ci_low": low if metric == "macro_f1" else np.nan,
                "ci_high": high if metric == "macro_f1" else np.nan,
                "p": mc["p"] if metric == "macro_f1" else np.nan,
                "test": f"McNemar {mc['method']} on paired correctness"
                if metric == "macro_f1"
                else "",
            }
        )
    for name in classes:
        binary = a["class"].to_numpy() == name
        try:
            result = delong_paired(binary, b[f"p_{name}"], a[f"p_{name}"])
        except ValueError:
            continue
        rows.append(
            {
                "metric": f"delong_{name}",
                "fused": result["auc_second"],
                "single": result["auc_first"],
                "difference": result["difference"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p": result["p"],
                "test": "paired DeLong one-vs-rest AUROC",
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.metric.str.startswith("delong_")
    table["p_holm"] = np.nan
    if eligible.any():
        table.loc[eligible, "p_holm"] = holm(table.loc[eligible, "p"])
    table.attrs["mcnemar"] = mc
    table.attrs["n_plants"] = len(a)
    table.attrs["f1_difference"] = point
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument(
        "--single", type=Path, required=True, help="Best single modality, chosen independently"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/h2"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report(
        [args.fused, args.single], args.out, args.allow_synthetic, require_run=True
    )
    table = evaluate(read_table(args.fused), read_table(args.single), args.seed, args.reps)
    table.to_csv(args.out / "h2_fusion.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = table.set_index("metric").loc[["macro_f1", "balanced_accuracy", "macro_auroc"]]
    positions = np.arange(len(metrics))
    ax.bar(positions - 0.2, metrics.fused, 0.4, label="Fused", color=PALETTE[0])
    ax.bar(positions + 0.2, metrics.single, 0.4, label="Single modality", color=PALETTE[1])
    ax.set(xticks=positions, xticklabels=metrics.index, ylabel="Held-out plant score", ylim=(0, 1))
    ax.legend()
    save_figure(fig, args.out / "h2_fusion", synthetic)
    row = metrics.loc["macro_f1"]
    passed = row.fused >= 0.80 and row.difference >= 0.06 and row.p < 0.05
    write_summary(
        args.out / "h2_summary.md",
        "H2 fusion",
        f"{table.attrs['n_plants']} paired plants. Fused macro F1 {row.fused:.3f}; "
        f"difference versus independently selected single modality {row.difference:.3f} "
        f"(plant bootstrap 95% CI {row.ci_low:.3f} to {row.ci_high:.3f}; "
        f"McNemar p={row.p:.4g}). Prespecified targets {'met' if passed else 'not met'}. "
        "McNemar tests paired accuracy, not macro F1. One-vs-rest DeLong p-values are Holm adjusted.",
        synthetic,
    )


if __name__ == "__main__":
    main()
