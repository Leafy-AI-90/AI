"""Plant-level, fold-local classification experiments.

The input is a validated feature table with one row per plant and day. This
module never infers treatment labels from optical measurements, fills missing
biology, or treats repeat readings as independent test cases.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from .config import load_config
from .fusion_training import F3Classifier
from .image_training import ImageTraining, fit_image_fold, validate_images
from .io import sha256_file
from .models import GroupedLateFusion, intermediate_fusion, spectral_model
from .validation import cycle_folds, grouped_folds
from .vision import CATALOG, DINOV2_SHA256


class ExperimentConfig(BaseModel):
    """Frozen, explicit experiment plan loaded from YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study: str = "xcc"
    name: str
    model: Literal[
        "logistic",
        "pls_da",
        "random_forest",
        "xgboost",
        "thermal_logistic",
        "f1_late",
        "f2_intermediate",
        "f3_cross_attention",
        "proposed",
        "dinov2",
        "efficientnet_b0",
        "phone",
        "phone_image",
    ]
    split: Literal["nested_grouped", "leave_one_cycle_out", "external_test"]
    features: dict[str, tuple[str, ...]]
    seeds: tuple[int, ...] = (42,)
    outer_folds: int = Field(default=5, ge=2)
    inner_folds: int = Field(default=3, ge=2)
    search_trials: int = Field(default=8, ge=0, le=100)
    endpoint_day: int | None = None
    input_file: Path
    external_file: Path | None = None
    synthetic: bool = False
    image_training: dict[str, Any] = Field(default_factory=dict)
    embedding_manifest: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.model not in ("efficientnet_b0", "phone_image") and (
            not self.features or any(not v for v in self.features.values())
        ):
            raise ValueError("Each modality must list explicit feature columns")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("Seeds must be nonempty and unique")
        if self.split == "external_test" and self.external_file is None:
            raise ValueError("external_test requires external_file")
        if (
            self.model in ("f1_late", "f2_intermediate", "f3_cross_attention", "proposed", "phone")
            and len(self.features) < 2
        ):
            raise ValueError("Fusion requires at least two modalities")
        if (
            self.model in ("f2_intermediate", "f3_cross_attention", "proposed", "phone")
            and next(iter(self.features)) != "image"
        ):
            raise ValueError("F2/F3/proposed require image features as the first modality")
        if self.model == "thermal_logistic" and set(self.features) != {"thermal"}:
            raise ValueError("Thermal baseline uses the thermal modality only")
        if "backbone" in self.image_training:
            raise ValueError("Image backbone is selected by model, not image_training")
        columns = [column for values in self.features.values() for column in values]
        if len(columns) != len(set(columns)):
            raise ValueError("A feature column cannot appear in more than one modality")
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Parse an experiment YAML without accepting unknown keys."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def _expand_features(cfg: ExperimentConfig, columns: pd.Index) -> ExperimentConfig:
    """Resolve explicit glob selectors against the saved table, then freeze names."""
    resolved = {}
    for modality, selectors in cfg.features.items():
        found = []
        for selector in selectors:
            matches = sorted(column for column in columns if fnmatch(column, selector))
            if not matches:
                raise ValueError(f"Feature selector {selector!r} matched no columns")
            found.extend(matches)
        if len(found) != len(set(found)):
            raise ValueError(f"Duplicate features in {modality} modality")
        resolved[modality] = tuple(found)
    all_columns = [column for values in resolved.values() for column in values]
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("A resolved feature appears in multiple modalities")
    return cfg.model_copy(update={"features": resolved})


def _check_embedding_manifest(cfg: ExperimentConfig, source: Path) -> None:
    if "image" not in cfg.features or cfg.synthetic:
        return
    if cfg.embedding_manifest is None:
        raise ValueError("Real image embeddings require a verified embedding manifest")
    manifest = json.loads(cfg.embedding_manifest.read_text(encoding="utf-8"))
    if cfg.model == "phone":
        pinned = json.loads(
            (Path(__file__).resolve().parents[2] / "models/manifest.json").read_text()
        )
        expected_hash = pinned["models"]["mobilenet_v3_large"]["sha256"]
    else:
        expected_hash = DINOV2_SHA256

    if (
        manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("checkpoint_sha256") != expected_hash
        or manifest.get("frozen") is not True
    ):
        raise ValueError(
            "Image embedding manifest does not verify this table and frozen checkpoint"
        )


def _prepare(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    study = load_config(cfg.study)
    if df.groupby("plant_id", dropna=False)["class"].nunique(dropna=False).gt(1).any():
        raise ValueError("A plant has inconsistent treatment labels across days")
    if df.groupby("plant_id", dropna=False)["cycle"].nunique(dropna=False).gt(1).any():
        raise ValueError("A plant appears in multiple cycles")
    required = [
        "plant_id",
        "cycle",
        "class",
        "day",
        *dict.fromkeys(col for cols in cfg.features.values() for col in cols),
    ]
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f"Missing experiment columns: {absent}")
    day = study.design.endpoint_day if cfg.endpoint_day is None else cfg.endpoint_day
    if not study.design.days.first < day <= study.design.days.last:
        raise ValueError("Endpoint day is outside the study schedule")
    data = df.loc[df["day"] == day, required + (["synthetic"] if "synthetic" in df else [])].copy()
    if data.empty:
        raise ValueError(f"No plants measured at endpoint day {day}")
    if data["plant_id"].duplicated().any():
        raise ValueError("One row per globally unique plant is required at the endpoint")
    if data[["plant_id", "cycle", "class"]].isna().any().any():
        raise ValueError("Plant identifiers, cycles and treatment labels cannot be missing")
    unknown = sorted(set(data["class"]) - set(study.classes))
    if unknown:
        raise ValueError(f"Treatment labels outside study preset: {unknown}")
    if (
        data[[c for c in required if c not in ("plant_id", "cycle", "class", "day")]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing features at endpoint; report them in QC and choose a complete cohort"
        )
    if not np.isfinite(
        data[[c for c in required if c not in ("plant_id", "cycle", "class", "day")]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Experiment features must be finite")
    if "synthetic" in data and data["synthetic"].astype(bool).any() != cfg.synthetic:
        raise ValueError("Synthetic marker and experiment configuration disagree")
    if cfg.synthetic and "synthetic" not in data:
        raise ValueError("Synthetic experiments require a synthetic=True source column")
    return data.sort_values("plant_id", ignore_index=True)


def _views(df: pd.DataFrame, cfg: ExperimentConfig) -> list[np.ndarray]:
    return [df[list(cols)].to_numpy(dtype=float) for cols in cfg.features.values()]


def _estimator(cfg: ExperimentConfig, seed: int, n_train: int):
    if cfg.model == "f1_late":
        return GroupedLateFusion(
            [spectral_model("logistic", seed) for _ in cfg.features],
            n_splits=cfg.inner_folds,
            seed=seed,
        )
    if cfg.model in ("f2_intermediate", "proposed", "phone"):
        first = len(next(iter(cfg.features.values())))
        total = sum(map(len, cfg.features.values()))
        return intermediate_fusion(
            first, total, pca_components=min(32, first, n_train - 1), seed=seed
        )
    name = "logistic" if cfg.model in ("thermal_logistic", "dinov2") else cfg.model
    return spectral_model(name, seed)


def _fit(model, views: list[np.ndarray], labels: np.ndarray, plants: np.ndarray):
    if isinstance(model, (GroupedLateFusion, F3Classifier)):
        return model.fit(views, labels, plants)
    return model.fit(np.column_stack(views), labels)


def _predict(model, views: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = views if isinstance(model, (GroupedLateFusion, F3Classifier)) else np.column_stack(views)
    return model.predict_proba(x), np.asarray(model.classes_)


def _tune(
    cfg: ExperimentConfig,
    views: list[np.ndarray],
    y: np.ndarray,
    plants: np.ndarray,
    seed: int,
    quick: bool,
):
    """Optuna objective sees training plants only; all estimators are fresh per fold."""
    if cfg.model == "f3_cross_attention":
        return F3Classifier(seed=seed, inner_folds=cfg.inner_folds, quick=quick), {}
    base = _estimator(cfg, seed, len(y))
    if cfg.search_trials == 0 or cfg.model not in (
        "logistic",
        "thermal_logistic",
        "dinov2",
        "random_forest",
        "xgboost",
    ):
        return base, {}
    import optuna

    folds = list(grouped_folds(y, plants, cfg.inner_folds, seed))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if cfg.model in ("logistic", "thermal_logistic", "dinov2"):
            params = {"logisticregression__C": trial.suggest_float("C", 0.01, 100, log=True)}
        elif cfg.model == "random_forest":
            params = {"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8)}
        else:
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            }
        scores = []
        for tr, va in folds:
            model = clone(base).set_params(**params)
            train_views = [v[tr] for v in views]
            valid_views = [v[va] for v in views]
            fitted = _fit(model, train_views, y[tr], plants[tr])
            p, classes = _predict(fitted, valid_views)
            scores.append(
                f1_score(y[va], classes[p.argmax(axis=1)], average="macro", zero_division=0)
            )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=min(cfg.search_trials, 2) if quick else cfg.search_trials)
    best = study.best_params
    if cfg.model in ("logistic", "thermal_logistic", "dinov2"):
        params = {"logisticregression__C": best["C"]}
    else:
        params = best
    return base.set_params(**params), best


def _metrics(y: np.ndarray, p: np.ndarray, classes: np.ndarray) -> dict:
    predicted = classes[p.argmax(axis=1)]
    result = {
        "n_plants": len(y),
        "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "per_class_recall": dict(
            zip(
                classes.tolist(),
                map(
                    float, recall_score(y, predicted, labels=classes, average=None, zero_division=0)
                ),
                strict=True,
            )
        ),
    }
    if set(classes) == set(y):
        result["macro_auroc"] = float(
            np.mean([roc_auc_score(y == c, p[:, i]) for i, c in enumerate(classes)])
        )
    return result


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return commit, dirty


def run_experiment(
    cfg: ExperimentConfig, out_root: str | Path = "runs", quick: bool = False
) -> Path:
    """Save plant OOF probabilities, metrics, trained models and full provenance."""
    commit, dirty = _git_state()
    if dirty and not cfg.synthetic:
        raise ValueError("Real-data runs require a clean committed code revision")
    if cfg.model in ("efficientnet_b0", "phone_image"):
        return _run_image_experiment(cfg, out_root, quick, commit, dirty)
    source = Path(cfg.input_file)
    raw = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    cfg = _expand_features(cfg, raw.columns)
    _check_embedding_manifest(cfg, source)
    data = _prepare(raw, cfg)
    external = None
    if cfg.external_file:
        path = Path(cfg.external_file)
        external = _prepare(
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path), cfg
        )
        if set(data.plant_id) & set(external.plant_id):
            raise ValueError("External test shares plant IDs with training")
        if set(data.cycle) & set(external.cycle):
            raise ValueError("External test shares cycles with training")
    study = load_config(cfg.study)
    classes = np.array(study.classes)
    run_dir = Path(out_root) / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    outcomes = []
    models = {}
    tuning = []
    for seed in cfg.seeds[:1] if quick else cfg.seeds:
        y = data["class"].to_numpy()
        plants = data["plant_id"].to_numpy()
        views = _views(data, cfg)
        if cfg.split == "nested_grouped":
            splits = list(grouped_folds(y, plants, cfg.outer_folds, seed))
        elif cfg.split == "leave_one_cycle_out":
            splits = list(cycle_folds(plants, data["cycle"].to_numpy()))
        else:
            splits = [(np.arange(len(data)), np.arange(len(external)))]
        for fold, (tr, te) in enumerate(splits):
            train_views = [v[tr] for v in views]
            model, params = _tune(cfg, train_views, y[tr], plants[tr], seed, quick)
            fitted = _fit(model, train_views, y[tr], plants[tr])
            if cfg.split == "external_test":
                test = external
                test_views = _views(test, cfg)
            else:
                test = data.iloc[te]
                test_views = [v[te] for v in views]
            p, model_classes = _predict(fitted, test_views)
            if set(model_classes) != set(classes):
                raise ValueError("Training fold lacks a study class")
            p = p[:, [np.flatnonzero(model_classes == c)[0] for c in classes]]
            record = test[["plant_id", "cycle", "class", "day"]].copy()
            record["seed"] = seed
            record["fold"] = fold
            for i, c in enumerate(classes):
                record[f"p_{c}"] = p[:, i]
            record["predicted"] = classes[p.argmax(axis=1)]
            outcomes.append(record)
            tuning.append({"seed": seed, "fold": fold, "best_params": params})
            models[f"seed{seed}_fold{fold}"] = fitted
    oof = pd.concat(outcomes, ignore_index=True)
    if cfg.split != "external_test" and oof.groupby(["seed", "plant_id"]).size().ne(1).any():
        raise ValueError("Each plant must receive exactly one held-out prediction per seed")
    metrics = {
        str(seed): _metrics(
            part["class"].to_numpy(), part[[f"p_{c}" for c in classes]].to_numpy(), classes
        )
        for seed, part in oof.groupby("seed")
    }
    oof.to_parquet(run_dir / "oof.parquet", index=False)
    joblib.dump(models, run_dir / "models.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "config": cfg.model_dump(mode="json"),
        "git_commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("leafy-ai", "numpy", "pandas", "scikit-learn", "xgboost", "optuna")
        },
        "input_sha256": sha256_file(source),
        "external_sha256": sha256_file(cfg.external_file) if cfg.external_file else None,
        "tuning": tuning,
        "quick": quick,
        "synthetic": cfg.synthetic,
    }
    (run_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _run_image_experiment(
    cfg: ExperimentConfig, out_root: str | Path, quick: bool, commit: str, dirty: bool
) -> Path:
    """B1/phone RGB branch with fresh initialization for every outer fold."""
    source = Path(cfg.input_file)
    raw = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    study = load_config(cfg.study)
    day = study.design.endpoint_day if cfg.endpoint_day is None else cfg.endpoint_day
    data = validate_images(raw, study.classes, day)
    if cfg.synthetic and ("synthetic" not in data or not data["synthetic"].all()):
        raise ValueError("Synthetic image runs require synthetic=True on every row")
    if not cfg.synthetic and "synthetic" in data and data["synthetic"].any():
        raise ValueError("Synthetic images cannot enter a real-data run")
    external = None
    if cfg.external_file:
        path = Path(cfg.external_file)
        raw_external = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        external = validate_images(raw_external, study.classes, day)
        if set(data.plant_id) & set(external.plant_id):
            raise ValueError("External test shares plant IDs with training")
        if set(data.cycle) & set(external.cycle):
            raise ValueError("External test shares cycles with training")
    backbone = "mobilenet_v3_large" if cfg.model == "phone_image" else "efficientnet_b0"
    settings = ImageTraining(backbone=backbone, **cfg.image_training)
    if quick:
        settings = ImageTraining(**{**settings.__dict__, "epochs": 1, "size": 64})
    plants = data.drop_duplicates("plant_id").reset_index(drop=True)
    y, ids, cycles = [plants[column].to_numpy() for column in ("class", "plant_id", "cycle")]
    run_dir = Path(out_root) / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "models").mkdir()
    outcomes, selected_epochs = [], []
    for seed in cfg.seeds[:1] if quick else cfg.seeds:
        if cfg.split == "nested_grouped":
            splits = grouped_folds(y, ids, cfg.outer_folds, seed)
        elif cfg.split == "leave_one_cycle_out":
            splits = cycle_folds(ids, cycles)
        else:
            splits = [(np.arange(len(plants)), np.arange(len(external.plant_id.unique())))]
        for fold, (tr, te) in enumerate(splits):
            train = data[data.plant_id.isin(ids[tr])]
            test = external if external is not None else data[data.plant_id.isin(ids[te])]
            model, probabilities, epochs = fit_image_fold(
                train, test, study.classes, settings, seed, cfg.inner_folds, quick
            )
            torch_path = run_dir / "models" / f"seed{seed}_fold{fold}.pt"
            import torch

            torch.save(model.cpu().state_dict(), torch_path)
            record = test.drop_duplicates("plant_id")[["plant_id", "cycle", "class", "day"]]
            record = record.merge(probabilities, on="plant_id", validate="one_to_one")
            record["seed"] = seed
            record["fold"] = fold
            record["predicted"] = np.array(study.classes)[
                record[[f"p_{c}" for c in study.classes]].to_numpy().argmax(axis=1)
            ]
            outcomes.append(record)
            selected_epochs.append({"seed": seed, "fold": fold, "epochs": epochs})
    oof = pd.concat(outcomes, ignore_index=True)
    if cfg.split != "external_test" and oof.groupby(["seed", "plant_id"]).size().ne(1).any():
        raise ValueError("Each plant must receive one held-out prediction per seed")
    oof.to_parquet(run_dir / "oof.parquet", index=False)
    classes = np.array(study.classes)
    metrics = {
        str(seed): _metrics(
            part["class"].to_numpy(), part[[f"p_{c}" for c in classes]].to_numpy(), classes
        )
        for seed, part in oof.groupby("seed")
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "config": cfg.model_dump(mode="json"),
        "git_commit": commit,
        "dirty": dirty,
        "input_sha256": sha256_file(source),
        "external_sha256": sha256_file(cfg.external_file) if cfg.external_file else None,
        "weights_sha256": sha256_file(
            Path(settings.weights_dir) / CATALOG[backbone][1].url.rsplit("/", 1)[1]
        ),
        "selected_epochs": selected_epochs,
        "quick": quick,
        "synthetic": cfg.synthetic,
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("leafy-ai", "numpy", "pandas", "torch", "torchvision")
        },
    }
    (run_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return run_dir
