"""Plant-level split and fold-local fitting checks for Phase 5."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch import nn

from leafy_ai.experiment import (
    ExperimentConfig,
    _check_embedding_manifest,
    _expand_features,
    _prepare,
    load_experiment,
    run_experiment,
)
from leafy_ai.fusion_training import F3Classifier
from leafy_ai.image_training import ImageTraining, fit_image_fold, validate_images


def _study(tmp_path):
    classes = ("control", "n_deficiency", "fe_deficiency", "xcc", "n_deficiency_xcc")
    rows = []
    for cycle in range(3):
        for label_idx, label in enumerate(classes):
            for plant in range(4):
                plant_id = f"c{cycle}_{label}_{plant}"
                for day in (-1, 8):
                    rows.append(
                        {
                            "plant_id": plant_id,
                            "cycle": f"cycle_{cycle}",
                            "class": label,
                            "day": day,
                            "m660": label_idx + 0.1 * plant + 0.01 * cycle,
                            "r_680": label_idx * 0.2 + 0.01 * plant,
                            "synthetic": True,
                        }
                    )
    path = tmp_path / "SYNTH_features.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    cfg = ExperimentConfig(
        name="software_check",
        model="logistic",
        split="leave_one_cycle_out",
        input_file=path,
        features={"spectral": ("m660", "r_680")},
        seeds=(13,),
        inner_folds=2,
        search_trials=2,
        synthetic=True,
    )
    return pd.DataFrame(rows), cfg


def test_experiment_yaml() -> None:
    cfg = load_experiment("configs/experiments/xcc_spectral.yaml")
    assert cfg.study == "xcc"
    assert cfg.split == "leave_one_cycle_out"


def test_rejects_repeated_endpoint_and_conflicting_labels(tmp_path) -> None:
    data, cfg = _study(tmp_path)
    repeated = pd.concat([data, data.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="One row per"):
        _prepare(repeated, cfg)
    data.loc[data.index[-1], "class"] = "control"
    with pytest.raises(ValueError, match="inconsistent treatment"):
        _prepare(data, cfg)


def test_missing_features_are_not_imputed(tmp_path) -> None:
    data, cfg = _study(tmp_path)
    data.loc[data.index[1], "m660"] = np.nan
    with pytest.raises(ValueError, match="Missing features"):
        _prepare(data, cfg)


def test_cycle_oof_is_one_prediction_per_plant(tmp_path, monkeypatch) -> None:
    data, cfg = _study(tmp_path)
    monkeypatch.setattr("leafy_ai.experiment._git_state", lambda: ("test-commit", False))
    import leafy_ai.experiment as experiment

    original_tune = experiment._tune
    training_cycles = []

    def audited_tune(config, views, labels, plants, seed, quick):
        training_cycles.append({str(plant).split("_")[0] for plant in plants})
        return original_tune(config, views, labels, plants, seed, quick)

    monkeypatch.setattr(experiment, "_tune", audited_tune)
    out = run_experiment(cfg, tmp_path / "runs", quick=True)
    oof = pd.read_parquet(out / "oof.parquet")
    assert len(oof) == data.plant_id.nunique()
    assert oof.plant_id.is_unique
    assert oof.synthetic.all()
    assert set(oof["class"]) == set(data["class"])
    assert np.allclose(oof.filter(regex="^p_").sum(axis=1), 1)
    run = json.loads((out / "run.json").read_text())
    assert run["synthetic"] and run["quick"]
    assert run["git_commit"] == "test-commit"
    assert len(run["tuning"]) == 3
    assert len(training_cycles) == 3
    assert all(len(cycles) == 2 for cycles in training_cycles)
    assert set.union(*training_cycles) == {"c0", "c1", "c2"}


def test_external_test_rejects_shared_plant_or_cycle(tmp_path, monkeypatch) -> None:
    _, cfg = _study(tmp_path)
    cfg = cfg.model_copy(update={"split": "external_test", "external_file": cfg.input_file})
    monkeypatch.setattr("leafy_ai.experiment._git_state", lambda: ("test-commit", False))
    with pytest.raises(ValueError, match="shares plant IDs"):
        run_experiment(cfg, tmp_path / "runs")


def test_image_fold_aggregates_by_plant_and_rejects_overlap(tmp_path, monkeypatch) -> None:
    rows = []
    for label in ("control", "xcc"):
        for plant in range(6):
            for image_idx in range(2):
                file = tmp_path / f"{label}_{plant}_{image_idx}.png"
                Image.new("RGB", (32, 32), (60 + 50 * (label == "xcc"), 100, 45)).save(file)
                rows.append(
                    {
                        "plant_id": f"{label}_{plant}",
                        "cycle": "a" if plant < 3 else "b",
                        "class": label,
                        "day": 8,
                        "file_path": str(file),
                    }
                )
    data = validate_images(pd.DataFrame(rows), ("control", "xcc"), 8)
    train = data[data.cycle == "a"]
    test = data[data.cycle == "b"]
    settings = ImageTraining(size=32, epochs=1, batch_size=4, patience=1)

    def tiny_model(_settings, labels, _seed):
        return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, len(labels)))

    monkeypatch.setattr("leafy_ai.image_training._new_model", tiny_model)
    model, probabilities, epochs = fit_image_fold(
        train, test, ("control", "xcc"), settings, seed=9, inner_folds=2, quick=True
    )
    assert isinstance(model, nn.Module)
    assert epochs == 1
    assert len(probabilities) == test.plant_id.nunique()
    assert np.allclose(probabilities[["p_control", "p_xcc"]].sum(axis=1), 1)
    with pytest.raises(ValueError, match="share plant IDs"):
        fit_image_fold(train, train, ("control", "xcc"), settings, 9, 2, True)


def test_f3_trains_on_inner_plant_folds_only() -> None:
    labels = np.repeat(np.arange(5), 4)
    plants = np.arange(len(labels))
    rng = np.random.default_rng(2)
    image = rng.normal(size=(len(labels), 8))
    spectra = rng.normal(size=(len(labels), 3))
    classifier = F3Classifier(seed=2, inner_folds=2, quick=True).fit(
        [image, spectra], labels, plants
    )
    probabilities = classifier.predict_proba([image[:2], spectra[:2]])
    assert probabilities.shape == (2, 5)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_real_image_features_require_matching_frozen_manifest(tmp_path) -> None:
    source = tmp_path / "features.parquet"
    source.write_bytes(b"not actually a parquet file")
    cfg = ExperimentConfig(
        name="b2",
        model="dinov2",
        split="leave_one_cycle_out",
        input_file=source,
        features={"image": ("image_*",)},
    )
    with pytest.raises(ValueError, match="verified embedding manifest"):
        _check_embedding_manifest(cfg, source)
    expanded = _expand_features(cfg, pd.Index(["image_001", "image_000", "class"]))
    assert expanded.features["image"] == ("image_000", "image_001")


def test_band_rfe_is_fit_only_on_outer_training_plants(tmp_path, monkeypatch) -> None:
    data, base = _study(tmp_path)
    rng = np.random.default_rng(11)
    band_names = [f"r_{410 + i:03d}" for i in range(18)] + ["a_660", "a_710", "a_940", "a_970"]
    for name in band_names:
        data[name] = rng.normal(size=len(data))
    original = tmp_path / "SYNTH_bands_original.parquet"
    data.to_parquet(original, index=False)
    changed = data.copy()
    changed.loc[changed.cycle == "cycle_0", band_names] += 1000 * rng.normal(
        size=(sum(changed.cycle == "cycle_0"), len(band_names))
    )
    perturbed = tmp_path / "SYNTH_bands_perturbed.parquet"
    changed.to_parquet(perturbed, index=False)
    monkeypatch.setattr("leafy_ai.experiment._git_state", lambda: ("test-commit", False))
    selected = []
    for path in (original, perturbed):
        cfg = ExperimentConfig(
            name=path.stem,
            model="logistic",
            split="leave_one_cycle_out",
            input_file=path,
            features={"spectral": tuple(band_names)},
            band_budget=3,
            seeds=(13,),
            inner_folds=2,
            search_trials=0,
            synthetic=True,
        )
        out = run_experiment(cfg, tmp_path / "runs", quick=True)
        selections = json.loads((out / "run.json").read_text())["tuning"]
        assert all(len(fold["selected_bands"]) == 3 for fold in selections)
        selected.append(selections[0]["selected_bands"])
    assert selected[0] == selected[1]


@pytest.mark.parametrize("model", ["f1_late", "f2_intermediate", "f3_cross_attention"])
def test_fusion_end_to_end_on_synthetic_plants(tmp_path, monkeypatch, model) -> None:
    data, base = _study(tmp_path)
    rng = np.random.default_rng(7)
    for i in range(8):
        data[f"image_{i:03d}"] = rng.normal(size=len(data))
    file = tmp_path / f"SYNTH_{model}.parquet"
    data.to_parquet(file, index=False)
    cfg = base.model_copy(
        update={
            "name": model,
            "model": model,
            "input_file": file,
            "features": {"image": ("image_*",), "spectral": ("m660", "r_680")},
            "search_trials": 0,
        }
    )
    monkeypatch.setattr("leafy_ai.experiment._git_state", lambda: ("test-commit", False))
    out = run_experiment(cfg, tmp_path / "runs", quick=True)
    oof = pd.read_parquet(out / "oof.parquet")
    assert oof.plant_id.is_unique
    assert len(oof) == data.plant_id.nunique()
