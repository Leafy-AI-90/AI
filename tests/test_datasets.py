import csv
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from leafy_ai.datasets import ArchiveDataset

spec = importlib.util.spec_from_file_location(
    "prepare_datasets", Path(__file__).resolve().parents[1] / "scripts/prepare_datasets.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_duplicate_training_image_cannot_cross_test_boundary():
    rows = [
        dict(group_id="same", label="healthy", upstream_split=s, image_valid="true")
        for s in ("train", "test")
    ]
    rows += [
        dict(group_id=str(i), label="healthy", upstream_split="train", image_valid="true")
        for i in range(10)
    ]
    result = prepare.assign_splits(rows)
    assert all(row["split"] == "test" for row in result[:2])
    assert {r["split"] for r in result[2:]} == {"train", "validation"}


def test_conflicting_labels_and_corrupt_images_are_quarantined():
    rows = [
        dict(group_id="same", label=label, upstream_split="train", image_valid="true")
        for label in ("healthy", "diseased")
    ]
    rows.append(
        dict(group_id="broken", label="healthy", upstream_split="train", image_valid="false")
    )
    assert all(row["split"] == "quarantine" for row in prepare.assign_splits(rows))


def test_archive_loader_uses_explicit_split(tmp_path):
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    raw = tmp_path / "data/raw/plantdoc"
    raw.mkdir(parents=True)
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    with zipfile.ZipFile(raw / "images.zip", "w") as z:
        z.writestr("train/leaf.png", image.getvalue())
    rows = [
        dict(
            archive="data/raw/plantdoc/images.zip",
            member="train/leaf.png",
            label="healthy",
            split="train",
        )
    ]
    with (folder / "plantdoc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = ArchiveDataset(tmp_path, "plantdoc", "train")
    assert data[0][0].size == (8, 8)
    assert data[0][1] == 0
    data.close()
    with pytest.raises(ValueError):
        ArchiveDataset(tmp_path, "plantdoc", "test")
