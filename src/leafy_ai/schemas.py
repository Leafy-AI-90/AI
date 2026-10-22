"""Pandera schemas for raw study tables; columns follow the active preset.

Schemas check structure and categorical domains only. Physical plausibility
(saturation, negative counts, drift) belongs to :mod:`leafy_ai.qc`, which flags
rows instead of rejecting them. Biological measurements are nullable: missing
values stay missing and are reported, never filled.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from .config import StudyConfig

TABLES = ("plants", "clip_readings", "references", "thermal", "images", "ground_truth")
SOLVENTS = ("acetone_80", "ethanol_95")
CALIBRATION_CLASS = "calibration"


def band_columns(cfg: StudyConfig, prefix: str = "as") -> list[str]:
    return [f"{prefix}_{band}" for band in cfg.spectral.bands_nm]


def led_columns(cfg: StudyConfig, prefix: str = "led") -> list[str]:
    return [f"{prefix}{led}_{state}" for led in cfg.transmission.leds_nm for state in ("on", "off")]


def views(cfg: StudyConfig) -> tuple[str, ...]:
    return ("top", "leaf", "root") if cfg.sensors.root_images else ("top", "leaf")


def _num(nullable: bool = False, *checks: pa.Check) -> pa.Column:
    return pa.Column(float, list(checks), nullable=nullable, coerce=True)


def _day(cfg: StudyConfig) -> pa.Column:
    days = cfg.design.days
    return pa.Column(int, pa.Check.in_range(days.first, days.last), coerce=True)


def _id() -> pa.Column:
    return pa.Column(str, pa.Check.str_length(min_value=1), coerce=True)


def _time() -> pa.Column:
    return pa.Column("datetime64[ns]", coerce=True)


def _cycle_prefix(df: pd.DataFrame) -> pd.Series:
    # plant_id must start with C<cycle>_ so IDs stay globally unique across cycles.
    prefix = "C" + df["cycle"].astype(str) + "_"
    return pd.Series(
        [pid.startswith(pre) for pid, pre in zip(df["plant_id"], prefix, strict=True)],
        index=df.index,
    )


def _reason_given(df: pd.DataFrame) -> pd.Series:
    reason = df["exclusion_reason"].fillna("").str.strip()
    return ~df["exclusion_flag"] | (reason != "")


def plants(cfg: StudyConfig) -> pa.DataFrameSchema:
    classes = [*cfg.classes, CALIBRATION_CLASS]
    return pa.DataFrameSchema(
        {
            "plant_id": pa.Column(str, unique=True, coerce=True),
            "cycle": pa.Column(int, pa.Check.ge(1), coerce=True),
            "class": pa.Column(str, pa.Check.isin(classes), coerce=True),
            "rack": _id(),
            "shelf": _id(),
            "position": _id(),
            "sow_date": pa.Column("datetime64[ns]", coerce=True),
            "transplant_date": pa.Column("datetime64[ns]", coerce=True),
            "cultivar": _id(),
            "seed_lot": _id(),
            "exclusion_flag": pa.Column(bool, coerce=True),
            "exclusion_reason": pa.Column(str, nullable=True),
        },
        checks=[
            pa.Check(_cycle_prefix, error="plant_id must start with C<cycle>_"),
            pa.Check(_reason_given, error="exclusion_reason required when excluded"),
            pa.Check(
                lambda df: df["transplant_date"] >= df["sow_date"],
                error="transplant_date must not precede sow_date",
            ),
        ],
        strict=True,
        name="plants",
    )


def clip_readings(cfg: StudyConfig) -> pa.DataFrameSchema:
    columns = {
        "plant_id": _id(),
        "day": _day(cfg),
        "timestamp": _time(),
        "leaf_position": pa.Column(str, pa.Check.isin(cfg.leaf_positions), coerce=True),
        "repeat": pa.Column(int, pa.Check.in_range(1, cfg.repeats), coerce=True),
        "integration_time_ms": _num(False, pa.Check.gt(0)),
        "gain": _num(False, pa.Check.gt(0)),
        "sensor_temp_c": _num(),
        "session_id": _id(),
    }
    columns |= {c: _num() for c in band_columns(cfg) + band_columns(cfg, "dark")}
    columns |= {c: _num() for c in led_columns(cfg)}
    return pa.DataFrameSchema(
        columns,
        unique=["plant_id", "day", "leaf_position", "repeat"],
        strict=True,
        name="clip_readings",
    )


def references(cfg: StudyConfig) -> pa.DataFrameSchema:
    columns = {"session_id": _id(), "timestamp": _time()}
    columns |= {c: _num() for c in band_columns(cfg, "white") + band_columns(cfg, "dark")}
    columns |= {c: _num() for c in led_columns(cfg, "empty")}
    columns |= {c: _num(False, pa.Check.in_range(0, 1)) for c in band_columns(cfg, "tile_r")}
    return pa.DataFrameSchema(
        columns, unique=["session_id", "timestamp"], strict=True, name="references"
    )


def thermal(cfg: StudyConfig) -> pa.DataFrameSchema:
    temps = ("leaf_mean_c", "leaf_median_c", "leaf_p10_c", "leaf_p90_c")
    columns = {"plant_id": _id(), "day": _day(cfg), "timestamp": _time()}
    columns |= {c: _num() for c in temps}
    columns |= {
        "leaf_std_c": _num(False, pa.Check.ge(0)),
        "air_temp_c": _num(),
        "rh_pct": _num(False, pa.Check.in_range(0, 100)),
        "wet_ref_c": _num(True),
        "dry_ref_c": _num(True),
        "frame_file": _id(),
    }
    return pa.DataFrameSchema(
        columns, unique=["plant_id", "day", "timestamp"], strict=True, name="thermal"
    )


def images(cfg: StudyConfig) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "image_id": pa.Column(str, unique=True, coerce=True),
            "plant_id": _id(),
            "day": _day(cfg),
            "view": pa.Column(str, pa.Check.isin(views(cfg)), coerce=True),
            "file_path": _id(),
            "phone_model": _id(),
            "exposure_s": _num(False, pa.Check.gt(0)),
            "iso": _num(False, pa.Check.gt(0)),
            "white_balance_k": _num(False, pa.Check.gt(0)),
            "colour_checker_detected": pa.Column(bool, coerce=True),
        },
        strict=True,
        name="images",
    )


def ground_truth(cfg: StudyConfig) -> pa.DataFrameSchema:
    nonneg = pa.Check.ge(0)
    score = pa.Check.in_range(0, 5)
    return pa.DataFrameSchema(
        {
            "plant_id": _id(),
            "day": _day(cfg),
            "chl_a_ug_cm2": _num(True, nonneg),
            "chl_b_ug_cm2": _num(True, nonneg),
            "chl_total_ug_cm2": _num(True, nonneg),
            "extraction_solvent": pa.Column(str, pa.Check.isin(SOLVENTS), nullable=True),
            "tissue_n_pct": _num(True, pa.Check.in_range(0, 100)),
            "sap_nitrate_ppm": _num(True, nonneg),
            "rwc_pct": _num(True, pa.Check.in_range(0, 120)),
            "root_browning": _num(True, score),
            "pathogen_load": _num(True, nonneg),
            "pathogen_confirmed": pa.Column("boolean", nullable=True, coerce=True),
            "rater1_score": _num(True, score),
            "rater2_score": _num(True, score),
        },
        checks=[
            pa.Check(
                lambda df: df["chl_total_ug_cm2"].isna() | df["extraction_solvent"].notna(),
                error="extraction_solvent required with chlorophyll values",
            )
        ],
        unique=["plant_id", "day"],
        strict=True,
        name="ground_truth",
    )


BUILDERS = {
    "plants": plants,
    "clip_readings": clip_readings,
    "references": references,
    "thermal": thermal,
    "images": images,
    "ground_truth": ground_truth,
}


def required_tables(cfg: StudyConfig) -> tuple[str, ...]:
    """Tables a preset must supply; thermal only when the thermal sensor is enabled."""
    return tuple(t for t in TABLES if t != "thermal" or cfg.sensors.thermal)


def schema(table: str, cfg: StudyConfig) -> pa.DataFrameSchema:
    if table not in BUILDERS:
        raise ValueError(f"Unknown table {table!r}; choose from {TABLES}")
    return BUILDERS[table](cfg)


def validate(table: str, df: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Validate and coerce one table; raises ``SchemaErrors`` listing every failure."""
    return schema(table, cfg).validate(df.reset_index(drop=True), lazy=True)
