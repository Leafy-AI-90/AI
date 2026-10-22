"""Synthetic reports are explicit software checks and include no paper claim."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from ablations import main as ablation_main  # noqa: E402
from agreement import main as agreement_main  # noqa: E402
from h2_fusion import main as h2_main  # noqa: E402
from h3_confusion import main as h3_main  # noqa: E402
from h4_lead_time import main as h4_main  # noqa: E402
from h5_root_vs_leaf import main as h5_main  # noqa: E402
from mixed_models import main as mixed_main  # noqa: E402
from power import main as power_main  # noqa: E402
from repeated_cv import main as cv_main  # noqa: E402


def _oof(tmp_path: Path) -> tuple[Path, Path]:
    classes = ("control", "n_deficiency", "root_disease")
    rows = []
    for i in range(18):
        label = classes[i % 3]
        probability = np.full(3, 0.1)
        probability[i % 3] = 0.8
        rows.append(
            {
                "plant_id": f"c{i % 3}-p{i}",
                "cycle": f"c{i % 3}",
                "class": label,
                "day": 10,
                "seed": 42,
                "synthetic": True,
                **{f"p_{name}": probability[j] for j, name in enumerate(classes)},
            }
        )
    good = pd.DataFrame(rows)
    weak = good.copy()
    weak.loc[weak.index[:3], [f"p_{name}" for name in classes]] = [0.8, 0.1, 0.1]
    a, b = tmp_path / "SYNTH_fused.parquet", tmp_path / "SYNTH_single.parquet"
    good.to_parquet(a, index=False)
    weak.to_parquet(b, index=False)
    return a, b


def test_hypothesis_reports_and_synthetic_gate(tmp_path: Path) -> None:
    fused, single = _oof(tmp_path)
    out = tmp_path / "h2"
    with pytest.raises(ValueError, match="Synthetic"):
        h2_main(
            ["--fused", str(fused), "--single", str(single), "--reps", "100", "--out", str(out)]
        )
    h2_main(
        [
            "--fused",
            str(fused),
            "--single",
            str(single),
            "--reps",
            "100",
            "--out",
            str(out),
            "--allow-synthetic",
        ]
    )
    assert (out / "h2_fusion.csv").is_file() and (out / "h2_fusion.svg").is_file()
    assert "SYNTHETIC SOFTWARE TEST" in (out / "h2_summary.md").read_text()
    h3_main(
        [
            "--augmented",
            str(fused),
            "--baseline",
            str(single),
            "--pair",
            "n_deficiency:control",
            "--reps",
            "100",
            "--out",
            str(tmp_path / "h3"),
            "--allow-synthetic",
        ]
    )
    h5_main(
        [
            "--leaf",
            str(single),
            "--leaf-root",
            str(fused),
            "--reps",
            "100",
            "--out",
            str(tmp_path / "h5"),
            "--allow-synthetic",
        ]
    )
    assert (tmp_path / "h3/h3_confusion.csv").is_file()
    assert (tmp_path / "h5/h5_root_vs_leaf.csv").is_file()


def test_daily_and_supporting_reports(tmp_path: Path) -> None:
    fused, single = _oof(tmp_path)
    ablation_main(
        [
            "--kind",
            "modality",
            "--run",
            f"fused={fused}",
            "--run",
            f"spectral={single}",
            "--reference",
            "fused",
            "--reps",
            "100",
            "--out",
            str(tmp_path / "ablations"),
            "--allow-synthetic",
        ]
    )
    daily, scores = [], []
    for i in range(9):
        label = "root_disease" if i % 2 else "n_deficiency"
        for day in range(5):
            daily.append(
                {
                    "plant_id": f"p{i}",
                    "cycle": "c0",
                    "class": label,
                    "day": day,
                    "seed": 42,
                    "p_control": 0.8 if day == 0 else 0.2,
                    "synthetic": True,
                }
            )
            scores.append(
                {"plant_id": f"p{i}", "day": day, "blind_score": int(day >= 4), "synthetic": True}
            )
    daily_path, score_path = tmp_path / "SYNTH_daily.parquet", tmp_path / "SYNTH_scores.parquet"
    pd.DataFrame(daily).to_parquet(daily_path, index=False)
    pd.DataFrame(scores).to_parquet(score_path, index=False)
    h4_main(
        [
            "--daily-oof",
            str(daily_path),
            "--blind-scores",
            str(score_path),
            "--reps",
            "100",
            "--out",
            str(tmp_path / "h4"),
            "--allow-synthetic",
        ]
    )
    assert (tmp_path / "h4/h4_lead_time.csv").is_file()
    raters = pd.DataFrame(
        {
            "plant_id": [f"p{i}" for i in range(12)],
            "day": 10,
            "rater1_score": [i % 3 for i in range(12)],
            "rater2_score": [(i + int(i % 4 == 0)) % 3 for i in range(12)],
            "synthetic": True,
        }
    )
    rater_path = tmp_path / "SYNTH_raters.csv"
    raters.to_csv(rater_path, index=False)
    agreement_main(
        [
            "--scores",
            str(rater_path),
            "--reps",
            "100",
            "--out",
            str(tmp_path / "agreement"),
            "--allow-synthetic",
        ]
    )
    fold_rows = [
        {
            "model": model,
            "repeat": 0,
            "fold": fold,
            "score": 0.6 + 0.1 * (model == "fused") + fold * 0.01,
            "test_plant_hash": f"hash{fold}",
            "test_fraction": 0.2,
            "synthetic": True,
        }
        for model in ("fused", "single")
        for fold in range(5)
    ]
    folds = tmp_path / "SYNTH_folds.csv"
    pd.DataFrame(fold_rows).to_csv(folds, index=False)
    cv_main(
        [
            "--fold-scores",
            str(folds),
            "--first",
            "fused",
            "--second",
            "single",
            "--out",
            str(tmp_path / "cv"),
            "--allow-synthetic",
        ]
    )
    power_main(["--p12", "0.05", "--p21", "0.15", "--out", str(tmp_path / "power")])
    rng = np.random.default_rng(44)
    physiology = []
    for plant in range(30):
        offset = rng.normal(scale=0.3)
        treatment = "control" if plant < 15 else "root_disease"
        for day in range(4):
            physiology.append(
                {
                    "plant_id": f"m{plant}",
                    "class": treatment,
                    "day": day,
                    "thermal_dt": offset
                    + day * (0.2 if plant >= 15 else 0.02)
                    + rng.normal(scale=0.1),
                    "synthetic": True,
                }
            )
    physiology_path = tmp_path / "SYNTH_physiology.csv"
    pd.DataFrame(physiology).to_csv(physiology_path, index=False)
    mixed_main(
        [
            "--features-table",
            str(physiology_path),
            "--feature",
            "thermal_dt",
            "--out",
            str(tmp_path / "mixed"),
            "--allow-synthetic",
        ]
    )
    assert all(
        (tmp_path / name / f"{stem}.csv").is_file()
        for name, stem in [
            ("ablations", "ablations"),
            ("agreement", "agreement"),
            ("cv", "repeated_cv"),
            ("power", "power"),
            ("mixed", "mixed_models"),
        ]
    )
