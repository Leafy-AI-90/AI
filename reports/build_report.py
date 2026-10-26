"""Assemble provenance-checked analysis tables and figures into a Quarto site.

Example: python reports/build_report.py --analysis h1=reports/h1 --analysis h2=reports/h2
Run `quarto render reports/paper` to create a static site suitable for GitHub Pages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from leafy_ai.io import sha256_file

CAPTIONS = {
    "predicted_vs_measured": "Held-out predicted versus extracted chlorophyll (µg/cm²).",
    "bland_altman": "Bland–Altman agreement for chlorophyll (µg/cm²).",
    "vip_bands": "PLSR variable importance by sensor band.",
    "h2_fusion": "Held-out plant classification across modalities.",
    "h3_confusion": "Plant-level confusion matrices before and after added features.",
    "h4_lead_time": "Fraction of plants without a sustained alert by day.",
    "h5_root_vs_leaf": "Leaf-only versus leaf-and-root image macro F1.",
    "ablations": "Held-out plant ablation comparison.",
    "repeated_cv": "Paired repeated-fold score differences.",
    "mixed_models": "Observed physiology by treatment and day.",
    "agreement": "Blinded symptom-rater agreement.",
    "power": "McNemar paired sample-size sensitivity.",
    "shap_global": "Global SHAP probability contribution on held-out plants.",
    "shap_wavelength": "SHAP importance by sensor wavelength (nm).",
    "gradcam_inside_leaf": "Fraction of Grad-CAM intensity inside the leaf mask.",
}


def _watermark(png: Path, svg: Path, target: Path) -> None:
    """Add an unmistakable software-only mark to both exported formats."""
    image = plt.imread(png)
    fig, ax = plt.subplots(figsize=(max(4, image.shape[1] / 300), max(3, image.shape[0] / 300)))
    ax.imshow(image)
    ax.axis("off")
    fig.text(
        0.5,
        0.5,
        "SYNTHETIC — NOT A BIOLOGICAL RESULT",
        ha="center",
        va="center",
        fontsize=20,
        color="#D55E00",
        alpha=0.65,
        rotation=30,
    )
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    source = svg.read_text(encoding="utf-8")
    label = '<text x="50%" y="50%" text-anchor="middle" fill="#D55E00" fill-opacity="0.85" font-size="28">SYNTHETIC — NOT A BIOLOGICAL RESULT</text>'
    if "</svg>" not in source:
        raise ValueError(f"Malformed SVG: {svg}")
    target.with_suffix(".svg").write_text(
        source.replace("</svg>", label + "\n</svg>"), encoding="utf-8"
    )


def _validate_source(directory: Path, allow_synthetic: bool) -> bool:
    """Check report provenance and current input hashes before publication."""
    provenance_path = directory / "run.json"
    if not provenance_path.is_file():
        raise ValueError(f"Missing run.json in {directory}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    synthetic = provenance.get("synthetic") is True
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic reports require --allow-synthetic")
    if not synthetic and provenance.get("working_tree_dirty"):
        raise ValueError("Real report was made from uncommitted code")
    for file, expected in provenance.get("inputs", {}).items():
        path = Path(file)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Analysis input hash mismatch: {path}")
    for stem in ("features", "ground_truth"):
        expected = provenance.get(f"{stem}_sha256")
        if expected:
            path = Path(provenance.get(f"{stem}_path", ""))
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Analysis input hash mismatch: {stem}")
    if not synthetic and not provenance.get("git_commit"):
        raise ValueError("Real report lacks a code revision")
    return synthetic


def build(analyses: dict[str, Path], out: Path, allow_synthetic: bool = False) -> Path:
    """Create a fresh, reproducible Quarto project without overwriting reports."""
    if not analyses:
        raise ValueError("At least one analysis report directory is required")
    root = Path(__file__).resolve().parents[1]
    sources = {name: _validate_source(path, allow_synthetic) for name, path in analyses.items()}
    if len(set(sources.values())) != 1:
        raise ValueError("Do not mix synthetic and real analysis reports")
    synthetic = next(iter(sources.values()))
    if (
        not synthetic
        and subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("Commit source changes before assembling a real-data paper")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.mkdir(parents=True)
    template = root / "reports" / "templates"
    shutil.copy2(template / "_quarto.yml", out / "_quarto.yml")
    shutil.copy2(template / "styles.css", out / "styles.css")
    lines = [
        "---",
        'title: "Leafy AI research report"',
        "---",
        "",
        "Generated from committed analysis code and hashed input artifacts.",
        "",
    ]
    if synthetic:
        lines += [
            "::: {.synthetic-banner}",
            "**SYNTHETIC SOFTWARE TEST — NOT A BIOLOGICAL RESULT**",
            ":::",
            "",
        ]
    manifest = {"synthetic": synthetic, "analyses": {}}
    for name, source_dir in analyses.items():
        target_dir = out / name
        target_dir.mkdir()
        manifest["analyses"][name] = {
            "source": str(source_dir),
            "run_sha256": sha256_file(source_dir / "run.json"),
        }
        lines += [f"## {name.upper()}", ""]
        for summary in sorted(source_dir.glob("*_summary.md")):
            lines += [summary.read_text(encoding="utf-8"), ""]
        for table in sorted(source_dir.rglob("*.csv")):
            relative = table.relative_to(source_dir)
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(table, destination)
            dataframe = pd.read_csv(table)
            lines += [
                f"### {table.stem.replace('_', ' ').title()}",
                "",
                f"[Download CSV]({name}/{relative.as_posix()})",
                "",
                dataframe.to_html(index=False, escape=True, classes="table table-striped"),
                "",
            ]
        for png in sorted(source_dir.rglob("*.png")):
            svg = png.with_suffix(".svg")
            if not svg.is_file():
                raise ValueError(f"Figure needs PNG and SVG: {png}")
            relative = png.relative_to(source_dir)
            target = (target_dir / relative).with_suffix("")
            target.parent.mkdir(parents=True, exist_ok=True)
            if synthetic:
                _watermark(png, svg, target)
            else:
                shutil.copy2(png, target.with_suffix(".png"))
                shutil.copy2(svg, target.with_suffix(".svg"))
            caption = CAPTIONS.get(png.stem, png.stem.replace("_", " ").capitalize())
            lines += [
                f'![{caption}]({name}/{relative.as_posix()}){{fig-cap="{caption}"}}',
                "",
                f"[Download SVG]({name}/{relative.with_suffix('.svg').as_posix()})",
                "",
            ]
    (out / "index.qmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", required=True, help="name=report-directory")
    parser.add_argument("--out", type=Path, default=Path("reports/paper"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    analyses = {}
    for entry in args.analysis:
        if "=" not in entry:
            parser.error("--analysis must be name=report-directory")
        name, path = entry.split("=", 1)
        if name in analyses or not name.replace("_", "").isalnum():
            parser.error("Analysis names must be unique letters/digits/underscores")
        analyses[name] = Path(path)
    print(build(analyses, args.out, args.allow_synthetic))


if __name__ == "__main__":
    main()
