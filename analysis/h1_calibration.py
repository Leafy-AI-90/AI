"""H1: held-out-plant chlorophyll calibration table and diagnostic figures.

Example: python analysis/h1_calibration.py --preset root_disease \
    --features data/interim/features.parquet --ground-truth data/interim/ground_truth.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from leafy_ai.calibration import evaluate_h1, prepare_h1
from leafy_ai.config import load_config


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input tables must be CSV or Parquet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _watermark(fig: plt.Figure, synthetic: bool) -> None:
    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC — NOT A BIOLOGICAL RESULT",
            ha="center",
            va="center",
            fontsize=22,
            alpha=0.25,
            rotation=30,
        )


def figures(oof: pd.DataFrame, vip: pd.DataFrame, out: Path, synthetic: bool) -> None:
    """Export 300 dpi PNG and vector SVG with measured units."""
    measured = oof["chl_total_ug_cm2"]
    predicted = oof["pred_plsr"]
    lo = min(measured.min(), predicted.min())
    hi = max(measured.max(), predicted.max())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(measured, predicted, color="#0072B2", s=25)
    ax.plot([lo, hi], [lo, hi], color="#D55E00", linestyle="--")
    ax.set(
        xlabel="Extracted chlorophyll (µg/cm²)",
        ylabel="Held-out PLSR prediction (µg/cm²)",
        title="H1: predicted vs measured",
    )
    _watermark(fig, synthetic)
    _save(fig, out / "predicted_vs_measured")

    difference = predicted - measured
    average = (predicted + measured) / 2
    bias = difference.mean()
    sd = difference.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(average, difference, color="#009E73", s=25)
    for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        ax.axhline(value, color="#D55E00", linestyle=style)
    ax.set(
        xlabel="Mean of measured and predicted (µg/cm²)",
        ylabel="Predicted − measured (µg/cm²)",
        title="H1: Bland–Altman agreement",
    )
    _watermark(fig, synthetic)
    _save(fig, out / "bland_altman")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(vip)), vip["vip_mean"], color="#0072B2")
    ax.set_xticks(range(len(vip)), vip["feature"], rotation=90)
    ax.set(
        xlabel="Wavelength feature",
        ylabel="Mean fold VIP (unitless)",
        title="H1: PLSR variable importance",
    )
    _watermark(fig, synthetic)
    _save(fig, out / "vip_bands")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="xcc")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/h1"))
    parser.add_argument(
        "--day", type=int, help="Pre-specified calibration day; preset endpoint by default"
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit visibly watermarked software-test figures",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.preset)
    feature_data, truth_data = _read(args.features), _read(args.ground_truth)
    synthetic = any(
        "synthetic" in df and df["synthetic"].astype(bool).any()
        for df in (feature_data, truth_data)
    )
    if synthetic and not args.allow_synthetic:
        raise ValueError("Synthetic data cannot produce H1 figures without --allow-synthetic")
    repo_root = Path(__file__).resolve().parents[1]
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
        ).strip()
    )
    if dirty and not synthetic:
        raise ValueError("Commit source changes before producing an H1 biological report")
    data, selection = prepare_h1(feature_data, truth_data, cfg, args.day)
    metrics, oof, vip = evaluate_h1(data, cfg)
    selection["outer_pls_components"] = metrics.attrs["outer_pls_components"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "h1_calibration.csv", index=False)
    oof.to_parquet(out / "h1_oof.parquet", index=False)
    vip.to_csv(out / "h1_vip.csv", index=False)
    figures(oof, vip, out, synthetic)
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    provenance = {
        "config": cfg.model_dump(mode="json"),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "features_sha256": _sha256(args.features),
        "ground_truth_sha256": _sha256(args.ground_truth),
        "synthetic": synthetic,
        "day": cfg.design.endpoint_day if args.day is None else args.day,
    }
    (out / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    pls = metrics.set_index("model").loc["plsr"]
    outcome = "passes" if pls.r2 >= 0.85 and pls.rpd >= 2 else "does not meet"
    summary = (
        f"# H1 calibration ({cfg.preset})\n\n"
        f"Day: {cfg.design.endpoint_day if args.day is None else args.day}; "
        f"solvent: {cfg.calibration.solvent}; unit: plant.\n\n"
        f"Selection: {selection}. Missing values were excluded and counted, never filled.\n\n"
        f"PLSR {outcome} the prespecified R² ≥ 0.85 and RPD ≥ 2.0 targets "
        f"(R²={pls.r2:.3f}, RPD={pls.rpd:.3f}).\n\n"
        "Tables: `h1_calibration.csv`, `h1_vip.csv`; plant-level predictions: `h1_oof.parquet`.\n"
    )
    if synthetic:
        summary = "**SYNTHETIC SOFTWARE TEST — NOT A BIOLOGICAL RESULT**\n\n" + summary
    (out / "h1_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
