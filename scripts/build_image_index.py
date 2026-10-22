"""Join validated image records to treatment labels and cycle metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from leafy_ai.io import write_immutable


def build(plants_file: Path, images_file: Path, output: Path) -> Path:
    plants = pd.read_parquet(plants_file)
    images = pd.read_parquet(images_file)
    if plants.plant_id.duplicated().any():
        raise ValueError("Plant metadata contains duplicate IDs")
    if "class" in images or "cycle" in images:
        raise ValueError("Image labels must come from the plants treatment table")
    index = images.merge(
        plants[["plant_id", "class", "cycle"]], on="plant_id", how="left", validate="many_to_one"
    )
    if index[["class", "cycle"]].isna().any().any():
        raise ValueError("An image refers to a plant without treatment metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_immutable(index, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plants", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build(args.plants, args.images, args.output))


if __name__ == "__main__":
    main()
