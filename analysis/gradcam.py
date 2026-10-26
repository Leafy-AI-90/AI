"""Grad-CAM for up to 20 random held-out image plants per class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common import PALETTE, prepare_report, read_table, save_figure, source_hash, write_summary
from PIL import Image

from leafy_ai.config import load_config
from leafy_ai.explain import attention_inside_mask, gradcam_map
from leafy_ai.image_training import ImageTraining
from leafy_ai.stats import validate_oof
from leafy_ai.vision import CATALOG, image_transform, load_backbone


def _select(oof: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen = []
    for _, group in oof.groupby("class", sort=True):
        chosen.extend(rng.choice(group.index.to_numpy(), min(per_class, len(group)), replace=False))
    return oof.loc[chosen].sort_values(["class", "plant_id"]).reset_index(drop=True)


def _image_rows(
    selected: pd.DataFrame, image_index: pd.DataFrame, run_images: pd.DataFrame
) -> pd.DataFrame:
    needed = {"plant_id", "class", "day", "file_path", "mask_path"}
    if needed - set(image_index):
        raise ValueError("Image index needs plant_id, class, day, file_path, mask_path")
    keys = ["plant_id", "class", "day", "file_path"]
    if set(keys) - set(run_images):
        raise ValueError("Saved experiment image index lacks endpoint image keys")
    eligible = image_index.merge(
        run_images[keys].drop_duplicates(), on=keys, how="inner", validate="many_to_one"
    )
    rows = selected.merge(
        eligible, on=["plant_id", "class", "day"], how="inner", validate="one_to_many"
    )
    if "view" in rows:
        rows = rows.loc[rows["view"] != "root"]
    rows = rows.sort_values(["plant_id", "file_path"]).drop_duplicates("plant_id")
    if set(rows.plant_id) != set(selected.plant_id):
        raise ValueError("Every selected plant needs a saved endpoint leaf image and mask")
    for column in ("file_path", "mask_path"):
        if rows[column].isna().any() or not all(Path(path).is_file() for path in rows[column]):
            raise FileNotFoundError(f"Missing {column} for selected test plants")
    return rows


def _load_model(run_dir: Path, run: dict, fold: int, seed: int, classes: list[str], device: str):
    backbone = (
        "mobilenet_v3_large" if run["config"]["model"] == "phone_image" else "efficientnet_b0"
    )
    settings = ImageTraining(backbone=backbone, **run["config"].get("image_training", {}))
    if run["quick"]:
        settings = ImageTraining(**{**settings.__dict__, "size": 64, "epochs": 1})
    weights_file = Path(settings.weights_dir) / CATALOG[backbone][1].url.rsplit("/", 1)[1]
    if source_hash(weights_file) != run["weights_sha256"]:
        raise ValueError("Image checkpoint provenance hash mismatch")
    model = load_backbone(backbone, settings.weights_dir, len(classes)).to(device)
    checkpoint = run_dir / "models" / f"seed{seed}_fold{fold}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, settings.size


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--images", type=Path, required=True, help="Image index with leaf mask_path"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=23)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("reports/gradcam"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if run["config"]["model"] not in ("efficientnet_b0", "phone_image"):
        raise ValueError("Grad-CAM requires a saved image-only fold model")
    if source_hash(Path(run["config"]["input_file"])) != run["input_sha256"]:
        raise ValueError("Image index differs from the saved experiment")
    synthetic = prepare_report(
        [args.run / "oof.parquet", args.images], args.out, args.allow_synthetic, require_run=False
    )
    if synthetic != run["synthetic"]:
        raise ValueError("Run and input synthetic markers disagree")
    study = load_config(run["config"]["study"])
    oof, _ = validate_oof(read_table(args.run / "oof.parquet"), args.seed)
    chosen = _select(oof, args.per_class, args.sample_seed)
    rows = _image_rows(
        chosen, read_table(args.images), read_table(Path(run["config"]["input_file"]))
    )
    (args.out / "plants").mkdir(exist_ok=True)
    reports = []
    models = {}
    for row in rows.to_dict(orient="records"):
        fold = int(row["fold"])
        if fold not in models:
            models[fold] = _load_model(
                args.run, run, fold, args.seed, list(study.classes), args.device
            )
        model, size = models[fold]
        with Image.open(row["file_path"]) as source:
            rgb = source.convert("RGB")
            tensor = image_transform(training=False, size=size)(rgb).unsqueeze(0).to(args.device)
            display = np.asarray(rgb.resize((size, size)), dtype=float) / 255.0
        with Image.open(row["mask_path"]) as source:
            mask = (
                np.asarray(source.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127
            )
        predicted = str(row["predicted"])
        target = list(study.classes).index(predicted)
        heat = gradcam_map(model, tensor, model.features[-1], target)
        inside = attention_inside_mask(heat, mask)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(display)
        ax.imshow(heat, cmap="viridis", alpha=0.45, vmin=0, vmax=1)
        ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
        ax.set(title=f"{row['plant_id']}: {predicted}", xticks=[], yticks=[])
        path = args.out / "plants" / str(row["plant_id"])
        save_figure(fig, path, synthetic)
        reports.append(
            {
                "plant_id": row["plant_id"],
                "class": row["class"],
                "predicted": predicted,
                "seed": args.seed,
                "fold": fold,
                "attention_inside_leaf": inside,
                "mask_path": row["mask_path"],
                "figure": str(path.with_suffix(".png")),
            }
        )
    table = pd.DataFrame(reports)
    table.to_csv(args.out / "gradcam.csv", index=False)
    summary = table.groupby("class").attention_inside_leaf.agg(["count", "mean", "median"])
    summary.to_csv(args.out / "gradcam_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary.index, summary["mean"], color=PALETTE[: len(summary)])
    ax.set(
        ylabel="Mean fraction of Grad-CAM inside leaf mask",
        ylim=(0, 1),
        title="Held-out attention inside leaf",
    )
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, args.out / "gradcam_inside_leaf", synthetic)
    write_summary(
        args.out / "gradcam_summary.md",
        "Grad-CAM leaf-mask audit",
        f"{len(table)} held-out plants sampled (up to {args.per_class} per class). "
        "The reported fraction measures localization only; Grad-CAM is not a causal explanation.",
        synthetic,
    )


if __name__ == "__main__":
    main()
