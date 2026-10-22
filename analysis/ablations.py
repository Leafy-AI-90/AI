"""Paired held-out ablations: band budget, modality, background, positions, pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary

from leafy_ai.stats import classification_metrics, pair_oof, plant_bootstrap

KINDS = {"bands", "modality", "background", "positions", "pretraining"}


def evaluate(
    runs: dict[str, pd.DataFrame], reference: str, seed: int | None, reps: int
) -> pd.DataFrame:
    """Compare each named ablation with one preselected reference on paired plants."""
    if reference not in runs or len(runs) < 2:
        raise ValueError("Reference and at least one ablation run required")
    rows = []
    for label, frame in runs.items():
        a, b, classes = pair_oof(frame, runs[reference], seed)
        score = classification_metrics(a, classes)["macro_f1"]
        baseline = classification_metrics(b, classes)["macro_f1"]
        if label == reference:
            low = high = 0.0
        else:
            _, low, high = plant_bootstrap(
                a,
                b,
                classes,
                lambda x, y, labels=classes: classification_metrics(x, labels)["macro_f1"]
                - classification_metrics(y, labels)["macro_f1"],
                reps=reps,
            )
        rows.append(
            {
                "run": label,
                "n_plants": len(a),
                "macro_f1": score,
                "reference_macro_f1": baseline,
                "difference": score - baseline,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--run", action="append", required=True, help="label=path/to/oof.parquet")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/ablations"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    paths = {}
    for item in args.run:
        if "=" not in item:
            parser.error("--run must be label=path")
        label, path = item.split("=", 1)
        if label in paths:
            parser.error("Duplicate run label")
        paths[label] = Path(path)
    if args.kind == "bands" and not {"3", "6", "12", "18", "22"} <= set(paths):
        parser.error("Band budget curve requires runs labelled 3, 6, 12, 18, 22")
    synthetic = prepare_report(
        list(paths.values()), args.out, args.allow_synthetic, require_run=True
    )
    table = evaluate(
        {name: read_table(path) for name, path in paths.items()},
        args.reference,
        args.seed,
        args.reps,
    )
    table.insert(0, "kind", args.kind)
    table.to_csv(args.out / "ablations.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    if args.kind == "bands":
        ordered = table.assign(bands=table.run.astype(int)).sort_values("bands")
        ax.plot(ordered.bands, ordered.macro_f1, marker="o", color=PALETTE[0])
        ax.set(xlabel="Selected band count", xticks=[3, 6, 12, 18, 22])
    else:
        ax.bar(table.run, table.macro_f1, color=PALETTE[0])
        ax.tick_params(axis="x", rotation=35)
        ax.set(xlabel="Ablation")
    ax.set(ylabel="Held-out plant macro F1", ylim=(0, 1), title=f"{args.kind.title()} ablation")
    save_figure(fig, args.out / "ablations", synthetic)
    write_summary(
        args.out / "ablations_summary.md",
        "Ablations",
        f"{args.kind} compared on identical held-out plants against {args.reference}. "
        "Difference intervals are paired plant bootstrap estimates. "
        "Band selection must be fit within each training fold of the source runs.",
        synthetic,
    )


if __name__ == "__main__":
    main()
