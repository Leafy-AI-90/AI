"""Create verified, frozen DINOv2 or MobileNet plant/day embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from leafy_ai.io import sha256_file, write_immutable
from leafy_ai.vision import (
    CATALOG,
    DINOV2_SHA256,
    dinov2_transform,
    embeddings,
    image_transform,
    load_backbone,
    load_dinov2,
    sha256,
)


def extract(
    feature_file: Path,
    image_file: Path,
    out: Path,
    weights_dir: Path,
    views: tuple[str, ...] = ("top", "leaf"),
    device: str = "cpu",
    backbone: str = "dinov2_vits14",
) -> Path:
    """Average deterministic frozen image embeddings per plant/day and join features."""
    features = (
        pd.read_parquet(feature_file)
        if feature_file.suffix == ".parquet"
        else pd.read_csv(feature_file)
    )
    images = (
        pd.read_parquet(image_file) if image_file.suffix == ".parquet" else pd.read_csv(image_file)
    )
    needed = {"plant_id", "day", "view", "file_path"}
    if not needed <= set(images):
        raise ValueError(f"Image index missing {sorted(needed - set(images))}")
    if features.duplicated(["plant_id", "day"]).any():
        raise ValueError("Feature table needs one row per plant/day")
    images = images[images["view"].isin(views)].copy()
    if images.empty:
        raise ValueError("No images for requested views")
    if images[["plant_id", "day", "file_path"]].isna().any().any():
        raise ValueError("Image paths and plant/day keys cannot be missing")
    keys = features[["plant_id", "day"]]
    if len(
        images.merge(keys, on=["plant_id", "day"], how="left", indicator=True).query(
            "_merge == 'left_only'"
        )
    ):
        raise ValueError("Images include plant/day keys absent from feature table")
    if backbone == "dinov2_vits14":
        transform = dinov2_transform()
        model = load_dinov2(weights_dir).to(device)
        dimension = 384
        checkpoint_hash = DINOV2_SHA256
    elif backbone == "mobilenet_v3_large":
        transform = image_transform(training=False, size=384)
        model = load_backbone(backbone, weights_dir).eval().to(device)
        dimension = CATALOG[backbone][2]
        checkpoint_hash = sha256(weights_dir / CATALOG[backbone][1].url.rsplit("/", 1)[1])
    else:
        raise ValueError("Choose dinov2_vits14 or mobilenet_v3_large")
    records = []
    with torch.inference_mode():
        for row in images.itertuples(index=False):
            with Image.open(row.file_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            vector = model(tensor) if backbone == "dinov2_vits14" else embeddings(model, tensor)
            embedding = vector.squeeze(0).cpu().numpy()
            if embedding.shape != (dimension,) or not np.isfinite(embedding).all():
                raise ValueError("Backbone did not return a finite embedding of expected size")
            records.append((row.plant_id, row.day, embedding))
    embedded = (
        pd.DataFrame(
            {
                "plant_id": [r[0] for r in records],
                "day": [r[1] for r in records],
                **{f"image_{i:03d}": [float(r[2][i]) for r in records] for i in range(dimension)},
            }
        )
        .groupby(["plant_id", "day"], as_index=False)
        .mean()
    )
    merged = features.merge(embedded, on=["plant_id", "day"], how="left", validate="one_to_one")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_immutable(merged, out)
    manifest = {
        "source_sha256": sha256_file(out),
        "features_sha256": sha256_file(feature_file),
        "images_sha256": sha256_file(image_file),
        "checkpoint_sha256": checkpoint_hash,
        "backbone": backbone,
        "frozen": True,
        "views": list(views),
        "aggregation": "mean of image embeddings per plant/day",
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/weights"))
    parser.add_argument("--view", action="append", dest="views")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backbone", choices=("dinov2_vits14", "mobilenet_v3_large"), default="dinov2_vits14"
    )
    args = parser.parse_args()
    print(
        extract(
            args.features,
            args.images,
            args.output,
            args.weights,
            tuple(args.views) if args.views else ("top", "leaf"),
            args.device,
            args.backbone,
        )
    )


if __name__ == "__main__":
    main()
