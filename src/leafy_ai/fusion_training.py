"""Fold-local F3 gated cross-attention comparison on frozen image embeddings."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .validation import grouped_folds
from .vision import GatedCrossAttention


class F3Classifier:
    """Cross-attend one frozen image vector to measured spectral bands.

    The image encoder must be frozen and SHA verified before this classifier is
    called. This is a comparison head, not a patch-level attention model.
    """

    def __init__(
        self,
        seed: int = 42,
        inner_folds: int = 3,
        epochs: int = 30,
        patience: int = 5,
        quick: bool = False,
    ):
        self.seed = seed
        self.inner_folds = inner_folds
        self.epochs = 1 if quick else epochs
        self.patience = patience

    def _new(self, image_dim: int, spectral_dim: int, n_classes: int) -> nn.Module:
        torch.manual_seed(self.seed)
        return GatedCrossAttention(
            image_dim=image_dim, spectral_dim=spectral_dim, classes=n_classes
        )

    @staticmethod
    def _tensor(views: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(views) < 2:
            raise ValueError("F3 needs image and at least one measured modality")
        return (
            torch.tensor(views[0], dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.column_stack(views[1:]), dtype=torch.float32),
        )

    @staticmethod
    def _epoch(
        model: nn.Module,
        image: torch.Tensor,
        spectral: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        model.train()
        for batch in torch.randperm(len(labels)).split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(image[batch], spectral[batch]), labels[batch], weight=weights
            )
            loss.backward()
            optimizer.step()

    def fit(self, views: list[np.ndarray], labels: np.ndarray, plants: np.ndarray):
        labels = np.asarray(labels)
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        train, valid = next(grouped_folds(labels, plants, self.inner_folds, self.seed))
        counts = np.bincount(encoded[train], minlength=len(self.classes_))
        if np.any(counts == 0):
            raise ValueError("Inner F3 training fold lacks a class")
        weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32)
        spectral_dim = sum(view.shape[1] for view in views[1:])
        model = self._new(views[0].shape[1], spectral_dim, len(counts))
        inner_image, inner_spectral = self._tensor([view[train] for view in views])
        inner_labels = torch.tensor(encoded[train], dtype=torch.long)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        best, best_epoch, stale = -1.0, 1, 0
        for epoch in range(1, self.epochs + 1):
            self._epoch(model, inner_image, inner_spectral, inner_labels, weights, optimizer)
            scheduler.step()
            p = self._predict_model(model, [view[valid] for view in views])
            score = f1_score(
                labels[valid],
                self.classes_[p.argmax(axis=1)],
                labels=self.classes_,
                average="macro",
                zero_division=0,
            )
            if score > best:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_ = self._new(views[0].shape[1], spectral_dim, len(counts))
        all_counts = np.bincount(encoded, minlength=len(self.classes_))
        all_weights = torch.tensor(
            len(encoded) / (len(all_counts) * all_counts), dtype=torch.float32
        )
        image, spectral = self._tensor(views)
        targets = torch.tensor(encoded, dtype=torch.long)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_epoch)
        for _ in range(best_epoch):
            self._epoch(self.model_, image, spectral, targets, all_weights, optimizer)
            scheduler.step()
        self.selected_epochs_ = best_epoch
        return self

    @staticmethod
    @torch.inference_mode()
    def _predict_model(model: nn.Module, views: list[np.ndarray]) -> np.ndarray:
        model.eval()
        image, spectral = F3Classifier._tensor(views)
        return model(image, spectral).softmax(dim=1).numpy()

    def predict_proba(self, views: list[np.ndarray]) -> np.ndarray:
        return self._predict_model(self.model_, views)
