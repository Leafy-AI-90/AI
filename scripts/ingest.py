"""Validate raw exports, run QC, write immutable interim Parquet and QC report.

Example:
    python scripts/ingest.py --preset xcc --raw data/raw/cycle1 --out data/interim/cycle1
"""

import argparse
from pathlib import Path

from leafy_ai.config import load_config
from leafy_ai.io import load_raw, write_immutable, write_interim
from leafy_ai.qc import run_qc, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="xcc", help="Preset name or YAML path")
    parser.add_argument("--raw", required=True, type=Path, help="Directory with raw table exports")
    parser.add_argument("--out", required=True, type=Path, help="New interim output directory")
    parser.add_argument(
        "--report", type=Path, help="QC report directory (default reports/qc/<out name>)"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.preset)
    data = load_raw(args.raw, cfg)
    manifest = write_interim(data, args.out, cfg)
    result = run_qc(data.tables, cfg)
    write_immutable(result.readings, args.out / "clip_readings_qc.parquet")
    write_immutable(result.images, args.out / "images_qc.parquet")
    report = write_report(result, args.report or Path("reports/qc") / args.out.name, cfg)
    print(f"Wrote {len(manifest['outputs'])} tables to {args.out}; QC report {report}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
