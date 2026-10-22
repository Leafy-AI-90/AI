"""Colour-correct and segment every image listed in an interim images table.

Writes corrected PNG, leaf mask and background-removed variant under --out, and
``images_processed.parquet`` beside the interim tables. Images without a
detected colour checker stay uncorrected with ``corrected_path`` empty.

Example:
    python scripts/process_images.py --interim data/interim/cycle1 --images-root data/raw/cycle1 --out data/interim/cycle1/images
"""

import argparse
from pathlib import Path

import pandas as pd

from leafy_ai.colour_correction import correct_image
from leafy_ai.io import write_immutable
from leafy_ai.segmentation import segment_image


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--interim", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path, help="Base for images.file_path")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    images = pd.read_parquet(args.interim / "images.parquet")
    rows = []
    for row in images.itertuples(index=False):
        source = args.images_root / row.file_path
        corrected, fit = correct_image(source, args.out / "corrected")
        mask, nobg, fraction = segment_image(corrected or source, args.out / "masks")
        rows.append(
            dict(
                image_id=row.image_id,
                corrected_path=str(corrected) if corrected else None,
                ccm_rmse=fit.rmse if fit else None,
                mask_path=str(mask),
                nobg_path=str(nobg),
                leaf_fraction=fraction,
            )
        )
    table = pd.DataFrame(rows)
    write_immutable(table, args.interim / "images_processed.parquet")
    print(
        f"Processed {len(table)} images; {table['corrected_path'].isna().sum()} without colour checker"
    )


if __name__ == "__main__":
    main()
