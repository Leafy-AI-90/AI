"""Decode real training batches and backpropagate; does not train/evaluate a model."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from leafy_ai.datasets import ArchiveDataset
from leafy_ai.vision import image_transform, load_backbone

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
torch.manual_seed(42)
report = {
    "purpose": "Real-data loading/backpropagation smoke check, not completed training",
    "datasets": {},
}
for name in ("plantvillage", "plantdoc"):
    data = ArchiveDataset(ROOT, name, "train", image_transform(size=384))
    loader = DataLoader(data, batch_size=2, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    model = load_backbone("efficientnet_b0", ROOT / "models/weights", num_classes=len(data.classes))
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.eval()
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    assert torch.isfinite(loss) and model.classifier[1].weight.grad is not None
    report["datasets"][name] = {
        "batch": list(images.shape),
        "classes": len(data.classes),
        "decode_and_backward": "passed",
    }
    data.close()
(ROOT / "docs/data-verification.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
