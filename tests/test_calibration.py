"""Hand-calculated extraction checks and independent-plant H1 software checks."""

import numpy as np
import pandas as pd
import pytest

from leafy_ai.calibration import (
    calibration_metrics,
    evaluate_h1,
    h1_columns,
    lichtenthaler,
    prepare_h1,
)
from leafy_ai.config import load_config


@pytest.mark.parametrize(
    ("solvent", "a", "b"),
    [
        ("acetone_80", 12.25 * 0.8 - 2.79 * 0.3, 21.50 * 0.3 - 5.10 * 0.8),
        ("ethanol_95", 13.36 * 0.8 - 5.19 * 0.3, 27.43 * 0.3 - 8.12 * 0.8),
    ],
)
def test_lichtenthaler_by_hand(solvent, a, b):
    result = lichtenthaler(
        0.8,
        0.3,
        solvent=solvent,
        extraction_volume_ml=10,
        dilution_factor=2,
        leaf_area_cm2=5,
        path_length_cm=1,
    )
    assert result.a == pytest.approx(4 * a)
    assert result.b == pytest.approx(4 * b)
    assert result.total == pytest.approx(4 * (a + b))


@pytest.mark.parametrize(
    "bad", [{"leaf_area_cm2": 0}, {"solvent": "methanol"}, {"absorbance_high": float("nan")}]
)
def test_lichtenthaler_rejects_invalid_inputs(bad):
    params = dict(
        absorbance_high=0.8,
        absorbance_low=0.3,
        solvent="acetone_80",
        extraction_volume_ml=10,
        dilution_factor=1,
        leaf_area_cm2=5,
    )
    with pytest.raises(ValueError):
        lichtenthaler(**(params | bad))


def _tables(n=30):
    cfg = load_config("xcc")
    rng = np.random.default_rng(7)
    columns = h1_columns(cfg)
    x = rng.uniform(0.1, 0.9, size=(n, len(columns)))
    plants = [f"C1_cal_{i:02d}" for i in range(n)]
    features = pd.DataFrame(x, columns=columns)
    features.insert(0, "plant_id", plants)
    features["day"] = cfg.design.endpoint_day
    features["class"] = "calibration"
    features["m660"] = x[:, 0] - x[:, 1]
    features["ci_red_edge"] = x[:, 2]
    truth = pd.DataFrame(
        {
            "plant_id": plants,
            "day": cfg.design.endpoint_day,
            "chl_total_ug_cm2": 20 + 8 * x[:, 0] - 3 * x[:, 1],
            "extraction_solvent": cfg.calibration.solvent,
        }
    )
    return cfg, features, truth


def test_h1_nested_predictions_one_per_plant():
    cfg, features, truth = _tables()
    data, counts = prepare_h1(features, truth, cfg)
    metrics, oof, importance = evaluate_h1(data, cfg)
    assert counts["usable_plants"] == 30
    assert len(oof) == oof.plant_id.nunique() == 30
    assert set(metrics.model) == {"plsr", "m660", "ci_red_edge"}
    assert len(importance) == len(h1_columns(cfg))
    assert np.isfinite(oof.filter(like="pred_").to_numpy()).all()


def test_plsr_never_predicts_a_plant_seen_during_fit(monkeypatch):
    import leafy_ai.calibration as calibration

    original = calibration.PLSRegression

    class GuardedPLS(original):
        def fit(self, x, y):
            self.training_rows = set(np.asarray(x)[:, 0])
            return super().fit(x, y)

        def predict(self, x):
            assert self.training_rows.isdisjoint(set(np.asarray(x)[:, 0]))
            return super().predict(x)

    monkeypatch.setattr(calibration, "PLSRegression", GuardedPLS)
    cfg, features, truth = _tables()
    data, _ = prepare_h1(features, truth, cfg)
    evaluate_h1(data, cfg)


def test_h1_reports_missing_truth_without_imputing():
    cfg, features, truth = _tables()
    truth.loc[0, "chl_total_ug_cm2"] = np.nan
    data, counts = prepare_h1(features, truth, cfg)
    assert counts["missing_extracted_chlorophyll"] == 1
    assert counts["usable_plants"] == 29
    assert features.loc[0, "plant_id"] not in set(data.plant_id)


def test_h1_rejects_repeated_plant_or_wrong_class():
    cfg, features, truth = _tables()
    with pytest.raises(ValueError, match="duplicate plant/day"):
        prepare_h1(pd.concat([features, features.iloc[[0]]]), truth, cfg)
    features["class"] = "control"
    with pytest.raises(ValueError, match="Too few measured"):
        prepare_h1(features, truth, cfg)


def test_agreement_metrics_by_hand():
    result = calibration_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    assert result["ba_bias_ug_cm2"] == pytest.approx(0.5)
    assert result["rmse_ug_cm2"] == pytest.approx(0.5)
    assert result["rpd"] == pytest.approx(2.0)


def test_h1_script_refuses_unmarked_synthetic_figures(tmp_path):
    import importlib.util
    from pathlib import Path

    cfg, features, truth = _tables()
    features["synthetic"] = True
    truth["synthetic"] = True
    feature_path, truth_path = tmp_path / "features.csv", tmp_path / "ground_truth.csv"
    features.to_csv(feature_path, index=False)
    truth.to_csv(truth_path, index=False)
    script = Path(__file__).resolve().parents[1] / "analysis/h1_calibration.py"
    spec = importlib.util.spec_from_file_location("h1_calibration", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = [
        "--preset",
        cfg.preset,
        "--features",
        str(feature_path),
        "--ground-truth",
        str(truth_path),
        "--out",
        str(tmp_path / "report"),
    ]
    with pytest.raises(ValueError, match="Synthetic data"):
        module.main(args)
    module.main([*args, "--allow-synthetic"])
    report = tmp_path / "report"
    assert (report / "h1_calibration.csv").exists()
    assert (report / "h1_oof.parquet").exists()
    assert "SYNTHETIC" in (report / "bland_altman.svg").read_text(encoding="utf-8")
