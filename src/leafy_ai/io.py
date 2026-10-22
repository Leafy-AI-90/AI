"""Load raw exports, validate against schemas, join, and write immutable Parquet.

Raw exports from the phone app and ESP32 logger are expected as one CSV or
Parquet file per table, named after the table (``plants.csv``,
``clip_readings.parquet``, ...) with columns matching :mod:`leafy_ai.schemas`.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .config import StudyConfig
from .schemas import TABLES, required_tables, validate

SUFFIXES = (".csv", ".parquet")


@dataclass
class RawData:
    tables: dict[str, pd.DataFrame]
    sources: dict[str, dict] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def find_table(raw_dir: str | Path, name: str) -> Path | None:
    """Return the single file for ``name``; error if both CSV and Parquet exist."""
    found = [
        Path(raw_dir) / f"{name}{s}" for s in SUFFIXES if (Path(raw_dir) / f"{name}{s}").is_file()
    ]
    if len(found) > 1:
        raise ValueError(f"Ambiguous {name}: both CSV and Parquet in {raw_dir}")
    return found[0] if found else None


def load_raw(raw_dir: str | Path, cfg: StudyConfig) -> RawData:
    """Read and validate every table the preset requires. Raises on any schema failure."""
    data = RawData({})
    for name in TABLES:
        path = find_table(raw_dir, name)
        if name not in required_tables(cfg):
            if path is not None:
                raise ValueError(f"{path.name} supplied but preset {cfg.preset!r} disables it")
            continue
        if path is None:
            raise FileNotFoundError(f"Missing required table {name} in {raw_dir}")
        data.tables[name] = validate(name, read_table(path), cfg)
        data.sources[name] = {"file": path.name, "sha256": sha256_file(path)}
    check_integrity(data.tables)
    return data


def check_integrity(tables: dict[str, pd.DataFrame]) -> None:
    """Every plant_id must exist in plants; every clip session must have references."""
    known = set(tables["plants"]["plant_id"])
    problems = []
    for name, df in tables.items():
        if name not in ("plants", "references") and "plant_id" in df:
            orphans = sorted(set(df["plant_id"]) - known)
            if orphans:
                problems.append(f"{name}: unknown plant_id {orphans[:5]}")
    if "clip_readings" in tables and "references" in tables:
        missing = sorted(
            set(tables["clip_readings"]["session_id"]) - set(tables["references"]["session_id"])
        )
        if missing:
            problems.append(f"clip_readings: sessions without references {missing[:5]}")
    if problems:
        raise ValueError("Referential integrity failed: " + "; ".join(problems))


def join_plant_day(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per measured plant/day: plant metadata, ground truth and thermal.

    Thermal must hold at most one frame summary per plant/day; aggregation of
    multiple frames is a feature decision for Phase 3, not done silently here.
    """
    keys = ["plant_id", "day"]
    index = tables["clip_readings"][keys].drop_duplicates()
    joined = index.merge(tables["plants"], on="plant_id", how="left", validate="many_to_one")
    joined = joined.merge(tables["ground_truth"], on=keys, how="left", validate="one_to_one")
    if "thermal" in tables:
        thermal = tables["thermal"].drop(columns="timestamp")
        if thermal.duplicated(keys).any():
            raise ValueError("Multiple thermal frames per plant/day; aggregate explicitly first")
        joined = joined.merge(thermal, on=keys, how="left", validate="one_to_one")
    return joined.sort_values(keys, ignore_index=True)


def write_immutable(df: pd.DataFrame, path: Path) -> str:
    """Write Parquet once. Identical content is a no-op; different content raises."""
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / path.name
        df.to_parquet(staged, index=False)
        digest = sha256_file(staged)
        if path.exists():
            if sha256_file(path) != digest:
                raise FileExistsError(
                    f"{path} exists with different content; interim data is immutable"
                )
            return digest
        path.write_bytes(staged.read_bytes())
    path.chmod(0o444)
    return digest


def write_interim(data: RawData, out_dir: str | Path, cfg: StudyConfig) -> dict:
    """Write validated tables plus plant/day join to ``out_dir`` with a hash manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = dict(data.tables) | {"plant_day": join_plant_day(data.tables)}
    files = {}
    for name, df in outputs.items():
        path = out / f"{name}.parquet"
        files[name] = {"file": path.name, "sha256": write_immutable(df, path), "rows": len(df)}
    manifest = {
        "preset": cfg.preset,
        "leafy_ai_version": version("leafy-ai"),
        "sources": data.sources,
        "outputs": files,
    }
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "created_utc"} != manifest:
            raise FileExistsError(f"{path} describes different data; use a new interim directory")
        return previous
    manifest["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
