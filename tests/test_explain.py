"""Held-out explanation checks, including disjoint SHAP background and leaf masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from explain_tabular import main as explain_main  # noqa: E402

from leafy_ai.experiment import ExperimentConfig, run_experiment  # noqa: E402
from leafy_ai.explain import (
    attention_inside_mask,
    band_importance,
    explain_tabular,
    gradcam_map,
    select_test_plants,
    shap_importance,
)


def test_shap_background_cannot_include_test_plant() -> None:
    frame = pd.DataFrame({"r_410": [0.1, 0.2], "a_660": [0.3, 0.4]})
    with pytest.raises(ValueError, match="disjoint"):
        explain_tabular(None, frame, frame.iloc[[0]], ["a", "b"], ["a"])


def test_permutation_shap_returns_per_class_band_scores() -> None:
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(24, 2)), columns=["r_410", "a_660"])
    y = np.where(x.r_410 + x.a_660 > 0, "stress", "control")
    model = LogisticRegression().fit(x.iloc[:20], y[:20])
    explanation = explain_tabular(
        model,
        x.iloc[:20],
        x.iloc[20:],
        [f"train{i}" for i in range(20)],
        [f"test{i}" for i in range(4)],
        background_size=5,
    )
    importance = shap_importance(explanation, model.classes_)
    assert len(importance) == 4
    bands = band_importance(importance)
    assert set(bands.wavelength_nm) == {410, 660}


def test_gradcam_heat_is_aligned_with_leaf_mask() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=1, bias=False),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2, 2, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[-1].weight.fill_(1)
    image = torch.ones((1, 3, 8, 8), requires_grad=False)
    heat = gradcam_map(model, image, model[0], 0)
    mask = np.zeros((8, 8), bool)
    mask[:, :4] = True
    assert heat.shape == mask.shape
    assert attention_inside_mask(heat, mask) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="aligned"):
        attention_inside_mask(heat, mask[:4])


def test_test_plant_sample_is_unique_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "plant_id": [f"p{i}" for i in range(30)],
            "class": ["a"] * 15 + ["b"] * 15,
            "seed": 42,
            "fold": 0,
        }
    )
    sample = select_test_plants(frame, 42, 0, per_class=4)
    assert len(sample) == 8 and sample.plant_id.is_unique
    with pytest.raises(ValueError, match="one held-out"):
        select_test_plants(pd.concat([frame, frame.iloc[[0]]]), 42, 0)


def test_saved_synthetic_fold_produces_watermarked_shap_report(tmp_path: Path) -> None:
    classes = ("control", "n_deficiency", "fe_deficiency", "xcc", "n_deficiency_xcc")
    rows = []
    for cycle in range(3):
        for index, label in enumerate(classes):
            for plant in range(3):
                rows.append(
                    {
                        "plant_id": f"c{cycle}_{index}_{plant}",
                        "cycle": f"c{cycle}",
                        "class": label,
                        "day": 8,
                        "synthetic": True,
                        "r_410": index + plant * 0.1 + cycle * 0.01,
                        "a_660": index * 0.4 + plant * 0.02,
                    }
                )
    source = tmp_path / "SYNTH_features.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    cfg = ExperimentConfig(
        name="shap_check",
        model="logistic",
        split="leave_one_cycle_out",
        input_file=source,
        features={"spectral": ("r_410", "a_660")},
        seeds=(17,),
        inner_folds=2,
        search_trials=0,
        synthetic=True,
    )
    run = run_experiment(cfg, tmp_path / "runs", quick=True)
    out = tmp_path / "shap"
    explain_main(
        [
            "--run",
            str(run),
            "--seed",
            "17",
            "--fold",
            "0",
            "--per-class",
            "1",
            "--background-size",
            "5",
            "--out",
            str(out),
            "--allow-synthetic",
        ]
    )
    assert (out / "shap_global.png").is_file()
    assert (out / "shap_wavelength.svg").is_file()
    assert "SYNTHETIC SOFTWARE TEST" in (out / "explain_summary.md").read_text()
