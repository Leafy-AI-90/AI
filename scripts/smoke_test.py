"""Real checkpoint inference and synthetic model-fit checks; no research metrics."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from leafy_ai.models import (
    GroupedLateFusion,
    chlorophyll_model,
    intermediate_fusion,
    spectral_model,
)
from leafy_ai.vision import CATALOG, GatedCrossAttention, embeddings, image_transform, load_backbone


def main(skip_checkpoints=False):
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    report = {
        "data": "synthetic software checks only; no plant performance claims",
        "models": {},
        "versions": {},
    }
    for package in (
        "torch",
        "torchvision",
        "scikit-learn",
        "xgboost",
        "timm",
        "shap",
        "statsmodels",
        "grad-cam",
    ):
        report["versions"][package] = importlib.metadata.version(package)
    # Import optional explanation packages as part of installation verification.
    import pytorch_grad_cam  # noqa: F401

    batch = image_transform()(Image.new("RGB", (400, 400), (60, 130, 45))).unsqueeze(0)
    for name, (_, _, dimension) in ({} if skip_checkpoints else CATALOG).items():
        model = load_backbone(name, root / "models/weights").eval()
        with torch.inference_mode():
            logits, vector = model(batch), embeddings(model, batch)
        assert logits.shape == (1, 1000) and vector.shape == (1, dimension)
        assert torch.isfinite(logits).all() and torch.isfinite(vector).all()
        report["models"][name] = {
            "checkpoint_forward": "passed",
            "input": list(batch.shape),
            "embedding": list(vector.shape),
        }
        del model
    x, y = rng.normal(size=(100, 21)), np.tile(np.arange(5), 20)
    for name in ("logistic", "pls_da", "random_forest", "svm", "mlp", "xgboost"):
        model = spectral_model(name).fit(x, y)
        assert model.predict(x[:3]).shape == (3,)
        report["models"][name] = "synthetic fit/predict passed"
    calibration = chlorophyll_model().fit(x, rng.normal(size=100))
    assert np.isfinite(calibration.predict(x[:3])).all()
    fused = rng.normal(size=(100, 88))
    fusion = intermediate_fusion(64, 88).fit(fused, y)
    assert fusion.predict(fused[:3]).shape == (3,)
    stacked = GroupedLateFusion([spectral_model("logistic"), spectral_model("random_forest")]).fit(
        [x, x[:, :5]], y, np.arange(100)
    )
    assert np.allclose(stacked.predict_proba([x[:3], x[:3, :5]]).sum(1), 1)
    attention = GatedCrossAttention()
    logits = attention(torch.randn(2, 4, 1280), torch.randn(2, 21))
    logits.sum().backward()
    assert logits.shape == (2, 5) and attention.image_projection.weight.grad is not None
    report["models"].update(
        {
            "plsr": "synthetic fit passed",
            "F1": "grouped OOF fit passed",
            "F2": "synthetic fit passed",
            "F3": "forward/backward passed (untrained)",
        }
    )
    if skip_checkpoints:
        # Partial run: never overwrite the full checkpoint verification record.
        report["models"]["checkpoints"] = "skipped (--skip-checkpoints)"
    else:
        (root / "docs/verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Skip ImageNet checkpoint forward passes (no downloaded weights needed)",
    )
    main(parser.parse_args().skip_checkpoints)
