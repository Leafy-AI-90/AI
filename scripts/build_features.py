"""Build plant/day feature table from interim Parquet written by scripts/ingest.py.

Example:
    python scripts/build_features.py --preset xcc --interim data/interim/cycle1
"""

import argparse
from pathlib import Path

import pandas as pd

from leafy_ai.config import load_config
from leafy_ai.feature_table import build_features
from leafy_ai.io import write_immutable
from leafy_ai.schemas import required_tables


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="xcc", help="Preset name or YAML path")
    parser.add_argument("--interim", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = load_config(args.preset)
    tables = {t: pd.read_parquet(args.interim / f"{t}.parquet") for t in required_tables(cfg)}
    readings = pd.read_parquet(args.interim / "clip_readings_qc.parquet")
    table = build_features(tables, cfg, readings)
    digest = write_immutable(table, args.interim / "features.parquet")
    print(f"Wrote {len(table)} plant/days x {table.shape[1]} columns; sha256 {digest[:12]}")


if __name__ == "__main__":
    main()
