"""Paired plant inference, leakage guards, and hypothesis-script calculations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from h2_fusion import evaluate as h2  # noqa: E402
from h3_confusion import evaluate as h3  # noqa: E402
from h4_lead_time import evaluate as h4  # noqa: E402
from h5_root_vs_leaf import evaluate as h5  # noqa: E402
from power import connor_pairs  # noqa: E402
from repeated_cv import evaluate as repeated_cv  # noqa: E402

from leafy_ai.stats import (  # noqa: E402
    classification_metrics,
    delong_paired,
    holm,
    mcnemar,
    nadeau_bengio,
    pair_oof,
    plant_bootstrap,
    validate_oof,
)

CLASSES = ("control", "n_deficiency", "root_disease")


def oof(n: int = 15) -> pd.DataFrame:
    labels = np.resize(np.array(CLASSES), n)
    p = np.full((n, 3), 0.1)
    for i, label in enumerate(labels):
        p[i, CLASSES.index(label)] = 0.8
    frame = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES])
    frame["plant_id"] = [f"c{i % 3}-p{i}" for i in range(n)]
    frame["cycle"] = [f"c{i % 3}" for i in range(n)]
    frame["class"] = labels
    frame["day"] = 10
    frame["seed"] = 42
    return frame


def test_duplicate_plant_or_seed_rejected() -> None:
    data = oof()
    with pytest.raises(ValueError, match="one nonmissing"):
        validate_oof(pd.concat([data, data.iloc[[0]]]))
    with pytest.raises(ValueError, match="Select one seed"):
        validate_oof(pd.concat([data, data.assign(seed=43)]))


def test_paired_labels_and_plants_must_match() -> None:
    data = oof()
    with pytest.raises(ValueError, match="identical"):
        pair_oof(data, data.iloc[1:])
    wrong = data.copy()
    wrong.loc[0, "class"] = "n_deficiency"
    with pytest.raises(ValueError, match="disagree"):
        pair_oof(data, wrong)


def test_mcnemar_exact_and_asymptotic() -> None:
    result = mcnemar(np.array([True] * 8), np.array([False] * 8))
    assert result["method"] == "exact binomial"
    assert result["p"] == pytest.approx(2 / 2**8)
    assert mcnemar(np.array([True] * 25), np.array([False] * 25))["method"].startswith("continuity")


def test_delong_direction_and_holm() -> None:
    y = np.array([1, 1, 1, 0, 0, 0], bool)
    poor = np.array([0.4, 0.2, 0.8, 0.6, 0.3, 0.7])
    good = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    result = delong_paired(y, poor, good)
    assert result["difference"] > 0
    assert result["auc_second"] == 1
    assert np.allclose(holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_bootstrap_resamples_plants_and_preserves_classes() -> None:
    data = oof()
    score, low, high = plant_bootstrap(
        data,
        data,
        CLASSES,
        lambda a, b: classification_metrics(a, CLASSES)["macro_f1"]
        - classification_metrics(b, CLASSES)["macro_f1"],
        reps=100,
    )
    assert (score, low, high) == (0, 0, 0)
    with pytest.raises(ValueError, match="plant_id"):
        plant_bootstrap(pd.concat([data, data.iloc[[0]]]), data, CLASSES, lambda a, b: 0, reps=100)


def test_h2_and_h3_use_paired_plants() -> None:
    good = oof()
    poor = good.copy()
    wrong = poor.index[:3]
    poor.loc[wrong, [f"p_{c}" for c in CLASSES]] = [0.8, 0.1, 0.1]
    table = h2(good, poor, reps=100)
    assert table.set_index("metric").loc["macro_f1", "difference"] > 0
    comparison, before, after = h3(good, poor, [("n_deficiency", "control")], reps=100)
    assert comparison.loc[0, "relative_reduction"] == 1
    assert before.to_numpy().sum() == after.to_numpy().sum() == len(good)


def test_h4_positive_lead_and_no_imputed_events() -> None:
    rows, scores = [], []
    for plant, label in [("a", "root_disease"), ("b", "n_deficiency")]:
        for day in range(5):
            rows.append(
                {
                    "plant_id": plant,
                    "day": day,
                    "class": label,
                    "seed": 42,
                    "p_control": 0.8 if day == 0 else 0.2,
                }
            )
            scores.append({"plant_id": plant, "day": day, "blind_score": int(day >= 4)})
    table, plants, curve = h4(pd.DataFrame(rows), pd.DataFrame(scores), reps=100)
    assert set(table.median_lead_days) == {3}
    assert set(plants.lead_days) == {3}
    assert not curve.empty


def test_h5_ratio_and_preset_guard() -> None:
    data = oof()
    table = h5(data, data, reps=100)
    assert table.loc[0, "ratio"] == 1
    no_root = data.loc[data["class"] != "root_disease"].drop(columns=["p_root_disease"]).copy()
    no_root[["p_control", "p_n_deficiency"]] /= 0.9
    with pytest.raises(ValueError, match="root_disease"):
        h5(no_root, no_root, reps=100)


def test_corrected_cv_requires_same_plant_hash() -> None:
    rows = []
    for model in ("a", "b"):
        for fold in range(5):
            rows.append(
                {
                    "model": model,
                    "repeat": 0,
                    "fold": fold,
                    "score": 0.8 + (model == "a") * 0.05 + fold * 0.005,
                    "test_plant_hash": f"fold{fold}",
                    "test_fraction": 0.2,
                }
            )
    frame = pd.DataFrame(rows)
    table, differences = repeated_cv(frame, "a", "b")
    assert len(differences) == 5 and table.loc[0, "mean_difference"] == pytest.approx(0.05)
    frame.loc[frame.model == "b", "test_plant_hash"] = "wrong"
    with pytest.raises(ValueError, match="same held-out"):
        repeated_cv(frame, "a", "b")


def test_connor_planning_monotonic_effect() -> None:
    assert connor_pairs(0.05, 0.15) < connor_pairs(0.08, 0.12)
    assert np.isinf(connor_pairs(0.1, 0.1))
    with pytest.raises(ValueError):
        connor_pairs(0.6, 0.6)


def test_nadeau_bengio_no_zero_variance_claim() -> None:
    result = nadeau_bengio([0.1, 0.1, 0.1], 0.2)
    assert result["p"] == 1
