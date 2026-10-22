"""Optional image pretraining on licensed, group-audited archive train splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from leafy_ai.datasets import ArchiveDataset
from leafy_ai.io import sha256_file
from leafy_ai.vision import CATALOG, image_transform, load_backbone


def _license_gate(root: Path, dataset: str) -> dict:
    """Require a local human-reviewed receipt for uncertain PlantVillage terms."""
    if dataset == "plantdoc":
        path = root / "data/raw/plantdoc/LICENSE.txt"
        if not path.is_file() or "Attribution 4.0 International" not in path.read_text(
            encoding="utf-8", errors="ignore"
        ):
            raise ValueError("PlantDoc CC-BY-4.0 license text is missing or unexpected")
        return {"license": "CC-BY-4.0", "file": str(path), "sha256": sha256_file(path)}
    if dataset == "plantvillage":
        path = root / "data/raw/plantvillage/license-review.json"
        if not path.is_file():
            raise ValueError(
                "PlantVillage needs a reviewed local license-review.json before pretraining"
            )
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not all(receipt.get(key) for key in ("license", "source_url", "reviewed_by")):
            raise ValueError("PlantVillage license review lacks license, source URL or reviewer")
        return receipt
    raise ValueError("Only PlantVillage and PlantDoc have implemented archive loaders")


@torch.inference_mode()
def _validate(model: nn.Module, loader: DataLoader, device: str, n_classes: int) -> float:
    model.eval()
    truth, predicted = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return float(
        f1_score(truth, predicted, labels=np.arange(n_classes), average="macro", zero_division=0)
    )


def pretrain(
    root: Path,
    dataset: str,
    output: Path,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
) -> Path:
    """Train on author archive train split; select epoch on audited validation split."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    license_info = _license_gate(root, dataset)
    torch.manual_seed(seed)
    train = ArchiveDataset(root, dataset, "train", image_transform(training=True))
    valid = ArchiveDataset(root, dataset, "validation", image_transform())
    if train.classes != valid.classes:
        raise ValueError("Archive train and validation class vocabularies differ")
    model = load_backbone("efficientnet_b0", root / "models/weights", len(train.classes)).to(device)
    counts = Counter(row["label"] for row in train.rows)
    weights = torch.tensor(
        [len(train) / (len(train.classes) * counts[label]) for label in train.classes],
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0)
    best_score = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(images.to(device)), labels.to(device), weight=weights
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        score = _validate(model, valid_loader, device, len(train.classes))
        if score > best_score:
            best_score = score
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    details = {
        "dataset": dataset,
        "classes": train.classes,
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "license": license_info,
        "split_manifest_sha256": sha256_file(root / f"data/processed/{dataset}.csv"),
        "imagenet_checkpoint_sha256": sha256_file(
            root / "models/weights" / CATALOG["efficientnet_b0"][1].url.rsplit("/", 1)[1]
        ),
        "model_sha256": sha256_file(output),
        "note": "Upstream archive validation only; no Choy Sum performance claim",
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    train.close()
    valid.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("plantvillage", "plantdoc"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        pretrain(
            args.root,
            args.dataset,
            args.output,
            args.epochs,
            args.batch_size,
            args.seed,
            args.device,
        )
    )


if __name__ == "__main__":
    main()
