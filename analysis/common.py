"""Shared provenance and output rules for hypothesis scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table without changing missing values."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Expected CSV or Parquet input")


def source_hash(path: Path) -> str:
    """Hash an input artifact for report provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_report(
    inputs: list[Path], out: Path, allow_synthetic: bool, *, require_run: bool = False
) -> bool:
    """Refuse unmarked synthetic or dirty real-data publication outputs."""
    synthetic_flags = []
    for path in inputs:
        table = read_table(path)
        synthetic_flags.append(bool("synthetic" in table and table.synthetic.fillna(False).any()))
        if path.name.startswith("SYNTH_") and not synthetic_flags[-1]:
            raise ValueError("SYNTH_ input must carry synthetic=True")
    synthetic = any(synthetic_flags)
    if synthetic and not all(synthetic_flags):
        raise ValueError("Do not mix real and synthetic input tables")
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic data need --allow-synthetic and watermark")
    root = Path(__file__).resolve().parents[1]
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    if dirty and not synthetic:
        raise ValueError("Commit code before producing a real-data report")
    if require_run and not synthetic:
        for path in inputs:
            if not (path.parent / "run.json").is_file():
                raise ValueError("Real OOF input must have its run.json provenance")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_dirty": dirty,
        "synthetic": synthetic,
        "inputs": {str(path): source_hash(path) for path in inputs},
    }
    (out / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return synthetic


def save_figure(fig: plt.Figure, path: Path, synthetic: bool) -> None:
    """Write 300 dpi PNG and SVG; mark software-only plots visibly."""
    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC — NOT A BIOLOGICAL RESULT",
            ha="center",
            va="center",
            fontsize=20,
            alpha=0.3,
            rotation=30,
        )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_summary(path: Path, title: str, text: str, synthetic: bool) -> None:
    """Write an explicit software-only banner where appropriate."""
    banner = "**SYNTHETIC SOFTWARE TEST — NOT A BIOLOGICAL RESULT**\n\n" if synthetic else ""
    path.write_text(f"{banner}# {title}\n\n{text.strip()}\n", encoding="utf-8")
