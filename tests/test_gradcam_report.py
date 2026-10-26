"""Grad-CAM CLI uses held-out image plants and their registered masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import gradcam as script  # noqa: E402

from leafy_ai.io import sha256_file  # noqa: E402


class TinyImageModel(nn.Module):
    """Spatial image model with deterministic positive saliency."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 5)
        with torch.no_grad():
            self.features[0].weight.zero_()
            self.features[0].bias.fill_(1)
            self.head.weight.fill_(1)
            self.head.bias.zero_()

    def forward(self, image):
        return self.head(self.pool(self.features(image)).flatten(1))


def test_gradcam_report_samples_unique_plants_and_measures_leaf_fraction(
    tmp_path: Path, monkeypatch
) -> None:
    classes = ("control", "n_deficiency", "fe_deficiency", "xcc", "n_deficiency_xcc")
    image_rows, oof_rows = [], []
    for i, name in enumerate(classes):
        image = tmp_path / f"SYNTH_image{i}.png"
        mask = tmp_path / f"SYNTH_mask{i}.png"
        Image.new("RGB", (16, 16), (100, 200, 50)).save(image)
        leaf = np.zeros((16, 16), dtype=np.uint8)
        leaf[:, :8] = 255
        Image.fromarray(leaf).save(mask)
        image_rows.append(
            {
                "plant_id": f"p{i}",
                "class": name,
                "day": 8,
                "view": "leaf close up",
                "file_path": str(image),
                "mask_path": str(mask),
                "synthetic": True,
            }
        )
        probability = np.full(5, 0.025)
        probability[i] = 0.9
        oof_rows.append(
            {
                "plant_id": f"p{i}",
                "class": name,
                "cycle": "c0",
                "day": 8,
                "seed": 17,
                "fold": 0,
                "predicted": name,
                "synthetic": True,
                **{f"p_{label}": probability[j] for j, label in enumerate(classes)},
            }
        )
    image_index = tmp_path / "SYNTH_images.parquet"
    pd.DataFrame(image_rows).to_parquet(image_index, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(oof_rows).to_parquet(run / "oof.parquet", index=False)
    (run / "run.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": "efficientnet_b0",
                    "study": "xcc",
                    "input_file": str(image_index),
                    "image_training": {},
                },
                "input_sha256": sha256_file(image_index),
                "synthetic": True,
                "quick": True,
            }
        )
    )
    monkeypatch.setattr(script, "_load_model", lambda *_args: (TinyImageModel(), 16))
    out = tmp_path / "gradcam"
    script.main(
        [
            "--run",
            str(run),
            "--images",
            str(image_index),
            "--seed",
            "17",
            "--per-class",
            "1",
            "--out",
            str(out),
            "--allow-synthetic",
        ]
    )
    table = pd.read_csv(out / "gradcam.csv")
    assert len(table) == 5 and table.plant_id.is_unique
    assert np.allclose(table.attention_inside_leaf, 0.5)
    assert len(list((out / "plants").glob("*.svg"))) == 5
    wrong_day = pd.DataFrame(image_rows).assign(day=9)
    with pytest.raises(ValueError, match="saved endpoint"):
        script._image_rows(pd.DataFrame(oof_rows), wrong_day, pd.DataFrame(image_rows))
