"""Paper assembly keeps synthetic results labelled and source hashes intact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports"))

from build_report import build  # noqa: E402

from leafy_ai.io import sha256_file  # noqa: E402


def test_synthetic_paper_requires_flag_and_marks_both_figure_formats(tmp_path: Path) -> None:
    source = tmp_path / "SYNTH_predictions.csv"
    pd.DataFrame({"synthetic": [True], "score": [0.5]}).to_csv(source, index=False)
    analysis = tmp_path / "h2"
    analysis.mkdir()
    (analysis / "run.json").write_text(
        json.dumps(
            {
                "git_commit": "test-commit",
                "synthetic": True,
                "working_tree_dirty": True,
                "inputs": {str(source): sha256_file(source)},
            }
        )
    )
    (analysis / "h2_fusion.csv").write_text("metric,value\nmacro_f1,0.5\n")
    (analysis / "h2_summary.md").write_text("Synthetic software check only.\n")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(analysis / "h2_fusion.png", dpi=300)
    fig.savefig(analysis / "h2_fusion.svg")
    (analysis / "plants").mkdir()
    fig.savefig(analysis / "plants" / "plant1.png", dpi=300)
    fig.savefig(analysis / "plants" / "plant1.svg")
    plt.close(fig)
    out = tmp_path / "paper"
    with pytest.raises(ValueError, match="Synthetic"):
        build({"h2": analysis}, out)
    build({"h2": analysis}, out, allow_synthetic=True)
    assert (out / "index.qmd").is_file()
    assert "SYNTHETIC SOFTWARE TEST" in (out / "index.qmd").read_text()
    assert "SYNTHETIC — NOT A BIOLOGICAL RESULT" in (out / "h2/h2_fusion.svg").read_text(
        encoding="utf-8"
    )
    assert (out / "h2/h2_fusion.png").is_file()
    assert (out / "h2/plants/plant1.png").is_file()
    with pytest.raises(FileExistsError):
        build({"h2": analysis}, out, allow_synthetic=True)
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        build({"h2": analysis}, tmp_path / "paper2", allow_synthetic=True)
