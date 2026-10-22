"""Fold-local RGB fine-tuning with plant-level validation and prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .validation import grouped_folds
from .vision import image_transform, load_backbone


@dataclass(frozen=True)
class ImageTraining:
    """Training settings; `quick` reduces epochs and pixels for CPU checks."""

    backbone: str = "efficientnet_b0"
    weights_dir: str = "models/weights"
    epochs: int = 30
    patience: int = 5
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    size: int = 384
    device: str = "cpu"


class _Images(Dataset):
    def __init__(self, rows: pd.DataFrame, labels: tuple[str, ...], training: bool, size: int):
        self.rows = rows.reset_index(drop=True)
        self.labels = {name: index for index, name in enumerate(labels)}
        self.transform = image_transform(training=training, size=size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["file_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[row["class"]], row["plant_id"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_images(rows: pd.DataFrame, labels: tuple[str, ...], day: int) -> pd.DataFrame:
    """Check labels and files; preserve multiple images under one plant ID."""
    needed = {"plant_id", "cycle", "class", "day", "file_path"}
    if not needed <= set(rows):
        raise ValueError(f"Image index is missing {sorted(needed - set(rows))}")
    if rows.groupby("plant_id")["class"].nunique().gt(1).any():
        raise ValueError("A plant has conflicting image treatment labels")
    if rows.groupby("plant_id")["cycle"].nunique().gt(1).any():
        raise ValueError("A plant appears in multiple cycles")
    out = rows.loc[rows["day"] == day].copy()
    if out.empty:
        raise ValueError(f"No images at endpoint day {day}")
    if set(out["class"]) - set(labels):
        raise ValueError("Image labels must come from study treatment records")
    if out[list(needed)].isna().any().any():
        raise ValueError("Missing image metadata at endpoint")
    missing = [str(path) for path in out["file_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image files: {missing[:5]}")
    return out.sort_values(["plant_id", "file_path"], ignore_index=True)


def _loader(
    rows: pd.DataFrame, labels: tuple[str, ...], settings: ImageTraining, training: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        _Images(rows, labels, training, settings.size),
        batch_size=settings.batch_size,
        shuffle=training,
        num_workers=0,
        generator=generator,
    )


def _class_weights(rows: pd.DataFrame, labels: tuple[str, ...], device: str) -> torch.Tensor:
    counts = rows.drop_duplicates("plant_id").groupby("class").size()
    if set(counts.index) != set(labels):
        raise ValueError("Training fold lacks a study class")
    return torch.tensor(
        [len(rows.plant_id.unique()) / (len(labels) * counts[c]) for c in labels],
        dtype=torch.float32,
        device=device,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    model.train()
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def predict_plant_proba(
    model: nn.Module, rows: pd.DataFrame, labels: tuple[str, ...], settings: ImageTraining
) -> pd.DataFrame:
    """Mean image probabilities within plant; one final row per plant."""
    model.eval()
    outputs = []
    for images, _, plants in _loader(rows, labels, settings, False, 0):
        probabilities = model(images.to(settings.device)).softmax(dim=1).cpu().numpy()
        for plant, vector in zip(plants, probabilities, strict=True):
            outputs.append(
                {
                    "plant_id": plant,
                    **{
                        f"p_{label}": float(value)
                        for label, value in zip(labels, vector, strict=True)
                    },
                }
            )
    return pd.DataFrame(outputs).groupby("plant_id", as_index=False).mean()


def _new_model(settings: ImageTraining, labels: tuple[str, ...], seed: int) -> nn.Module:
    _seed_everything(seed)
    return load_backbone(settings.backbone, settings.weights_dir, len(labels)).to(settings.device)


def fit_image_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: tuple[str, ...],
    settings: ImageTraining,
    seed: int,
    inner_folds: int = 3,
    quick: bool = False,
) -> tuple[nn.Module, pd.DataFrame, int]:
    """Choose epochs using an inner plant fold, refit, then predict outer plants."""
    if set(train.plant_id) & set(test.plant_id):
        raise ValueError("Image train/test share plant IDs")
    if not all(Path(path).is_file() for path in pd.concat([train, test])["file_path"]):
        raise FileNotFoundError("Image fold contains a missing file")
    train_index, valid_index = next(
        grouped_folds(train["class"].to_numpy(), train["plant_id"].to_numpy(), inner_folds, seed)
    )
    inner_train, valid = train.iloc[train_index], train.iloc[valid_index]
    weights = _class_weights(inner_train, labels, settings.device)
    model = _new_model(settings, labels, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    epochs = min(settings.epochs, 1) if quick else settings.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = _loader(inner_train, labels, settings, True, seed)
    best_score, best_epoch, stale = -1.0, 1, 0
    for epoch in range(1, epochs + 1):
        _epoch(model, train_loader, nn.CrossEntropyLoss(weight=weights), optimizer, settings.device)
        scheduler.step()
        valid_p = predict_plant_proba(model, valid, labels, settings)
        truth = (
            valid.drop_duplicates("plant_id").set_index("plant_id").loc[valid_p.plant_id, "class"]
        )
        scores = valid_p[[f"p_{c}" for c in labels]].to_numpy()
        predicted = np.array(labels)[scores.argmax(axis=1)]
        score = f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    # A new model is fitted to every outer training plant for the selected epoch budget.
    final = _new_model(settings, labels, seed)
    optimizer = torch.optim.AdamW(
        final.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_epoch)
    train_loader = _loader(train, labels, settings, True, seed)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train, labels, settings.device))
    for _ in range(best_epoch):
        _epoch(final, train_loader, loss_fn, optimizer, settings.device)
        scheduler.step()
    return final, predict_plant_proba(final, test, labels, settings), best_epoch
