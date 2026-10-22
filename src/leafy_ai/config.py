"""Study design presets loaded from YAML and validated with pydantic.

Presets live in ``configs/presets/<name>.yaml`` at the repository root. Set
``LEAFY_AI_CONFIG_DIR`` to a directory containing ``presets/`` to override.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

AS7265X_BANDS = (
    410, 435, 460, 485, 510, 535, 560, 585, 610,
    645, 680, 705, 730, 760, 810, 860, 900, 940,
)  # fmt: skip
TRANSMISSION_LEDS = (660, 710, 940, 970)
REQUIRED_LEDS = (660, 710, 940)
LEAF_POSITIONS = ("oldest", "youngest", "middle", "lesion", "off_lesion")
PRESETS = ("root_disease", "xcc", "salt_planb")
DEFAULT_PRESET = "xcc"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Days(_Frozen):
    first: int = -1
    last: int = 14


class Design(_Frozen):
    plants_per_class: int = Field(gt=0)
    cycles: int = Field(ge=2)
    calibration_plants: int = Field(ge=0)
    days: Days = Days()
    endpoint_day: int


class Spectral(_Frozen):
    sensor: Literal["AS7265x"] = "AS7265x"
    bands_nm: tuple[int, ...] = AS7265X_BANDS


class Transmission(_Frozen):
    leds_nm: tuple[int, ...] = REQUIRED_LEDS


class Sensors(_Frozen):
    rgb: bool = True
    spectral: bool = True
    transmission: bool = True
    thermal: bool = False
    root_images: bool = False


QC_FLAGS = (
    "saturated",
    "negative_counts",
    "white_drift",
    "temperature_drift",
    "repeat_cv",
    "missing_position",
    "no_colour_checker",
)


class QC(_Frozen):
    """Pre-registered QC thresholds. Flagged rows are excluded only for ``exclude_flags``."""

    saturation_count: float = Field(default=65535, gt=0)
    white_drift_max: float = Field(default=0.02, gt=0)
    temperature_drift_max_c: float = Field(default=2.0, gt=0)
    repeat_cv_max: float = Field(default=0.05, gt=0)
    exclude_flags: tuple[str, ...] = ("saturated", "negative_counts")

    @model_validator(mode="after")
    def _check(self) -> QC:
        unknown = set(self.exclude_flags) - set(QC_FLAGS)
        if unknown:
            raise ValueError(f"Unknown QC flags {sorted(unknown)}; choose from {QC_FLAGS}")
        return self


class StudyConfig(_Frozen):
    """One study design. Every field is explicit in the preset YAML."""

    schema_version: Literal[1] = 1
    preset: str
    description: str = ""
    crop: str = "choy_sum"
    classes: tuple[str, ...]
    design: Design
    leaf_positions: tuple[str, ...]
    repeats: int = Field(gt=0)
    spectral: Spectral = Spectral()
    transmission: Transmission = Transmission()
    sensors: Sensors = Sensors()
    key_confusion_pairs: tuple[tuple[str, str], ...] = ()
    qc: QC = QC()

    @model_validator(mode="after")
    def _check(self) -> StudyConfig:
        classes = self.classes
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("Need at least two unique classes")
        if "control" not in classes:
            raise ValueError("A 'control' class is required")
        bands = self.spectral.bands_nm
        if not bands or list(bands) != sorted(set(bands)) or not set(bands) <= set(AS7265X_BANDS):
            raise ValueError("bands_nm must be strictly increasing AS7265x bands")
        leds = self.transmission.leds_nm
        if list(leds) != sorted(set(leds)) or not set(leds) <= set(TRANSMISSION_LEDS):
            raise ValueError(f"leds_nm must be strictly increasing values from {TRANSMISSION_LEDS}")
        if not set(REQUIRED_LEDS) <= set(leds):
            raise ValueError(f"leds_nm must include {REQUIRED_LEDS}")
        positions = self.leaf_positions
        if len(positions) not in (3, 4) or len(set(positions)) != len(positions):
            raise ValueError("Use 3 or 4 unique leaf positions")
        if not set(positions) <= set(LEAF_POSITIONS):
            raise ValueError(f"leaf_positions must come from {LEAF_POSITIONS}")
        days = self.design.days
        if not days.first < self.design.endpoint_day <= days.last:
            raise ValueError("endpoint_day must fall after the first day and by the last day")
        for pair in self.key_confusion_pairs:
            if len(set(pair)) != 2 or not set(pair) <= set(classes):
                raise ValueError(f"Confusion pair {pair} must name two different classes")
        return self

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_classification_plants(self) -> int:
        return self.n_classes * self.design.plants_per_class * self.design.cycles


def config_dir() -> Path:
    """Directory holding ``presets/``; env var first, then repository ``configs/``."""
    override = os.environ.get("LEAFY_AI_CONFIG_DIR")
    path = Path(override) if override else Path(__file__).resolve().parents[2] / "configs"
    if not (path / "presets").is_dir():
        raise FileNotFoundError(
            f"No presets directory under {path}. Use an editable install from the "
            "repository or set LEAFY_AI_CONFIG_DIR."
        )
    return path


def _resolve(preset: str | Path) -> Path:
    if isinstance(preset, str) and preset in PRESETS:
        return config_dir() / "presets" / f"{preset}.yaml"
    path = Path(preset)
    if path.suffix not in (".yaml", ".yml") or not path.is_file():
        raise ValueError(f"Unknown preset {preset!r}; choose {PRESETS} or a YAML path")
    return path


@lru_cache(maxsize=16)
def _load(path: Path) -> StudyConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return StudyConfig.model_validate(data)


def load_config(preset: str | Path = DEFAULT_PRESET) -> StudyConfig:
    """Load and validate a preset name (``xcc``, ``root_disease``, ``salt_planb``) or YAML path."""
    return _load(_resolve(preset).resolve())
