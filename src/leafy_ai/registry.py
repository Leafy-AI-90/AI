"""Feature registry: name, unit, source sensor and equation for every feature column.

``feature_registry(cfg)`` lists exactly the columns :func:`leafy_ai.feature_table.build_features`
produces for a preset (tested). ``docs/features.md`` is generated from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import BANDS
from .config import StudyConfig
from .feature_table import SLOPE_WINDOW

SPECTRAL = "AS7265x"
TRANSMISSION = "Transmission LEDs"
THERMAL = "MLX90640 + air T/RH"


@dataclass(frozen=True)
class Feature:
    name: str
    unit: str
    sensor: str
    equation: str
    reference: str = ""


REFERENCES = {
    "reflectance": "Dark-corrected leaf / dark-corrected white tile x tile reflectance",
    "beer": "Beer-Lambert absorbance, log10(I0 / I)",
    "ndvi": "Rouse et al. (1974)",
    "ndre": "Gitelson & Merzlyak (1994)",
    "ci_re": "Gitelson et al. (2003), doi:10.1029/2002GL016450",
    "gndvi": "Gitelson et al. (1996), doi:10.1016/S0034-4257(96)00072-7",
    "pri": "Gamon et al. (1992), 531/570 nm; nearest AS7265x bands 535/560 nm",
    "water": "Liquid water band near 970 nm (cf. Penuelas et al. 1997 water index)",
    "cwsi": "Jones (1999), doi:10.1016/S0168-1923(99)00005-2",
    "vpd": "Allen et al. (1998) FAO-56, eq. 11 to 13",
}


def _spectral() -> list[Feature]:
    feats = [Feature(f"r_{b}", "1", SPECTRAL, f"R_{b}", REFERENCES["reflectance"]) for b in BANDS]
    feats += [
        Feature(f"dr_{a}_{b}", "1/nm", SPECTRAL, f"(R_{b} - R_{a}) / ({b} - {a})")
        for a, b in zip(BANDS[:-1], BANDS[1:], strict=True)
    ]
    feats += [
        Feature("ndvi", "1", SPECTRAL, "(R810 - R680) / (R810 + R680)", REFERENCES["ndvi"]),
        Feature("ndre", "1", SPECTRAL, "(R760 - R705) / (R760 + R705)", REFERENCES["ndre"]),
        Feature("ci_red_edge", "1", SPECTRAL, "R760 / R705 - 1", REFERENCES["ci_re"]),
        Feature("gndvi", "1", SPECTRAL, "(R810 - R560) / (R810 + R560)", REFERENCES["gndvi"]),
        Feature("pri_proxy", "1", SPECTRAL, "(R535 - R560) / (R535 + R560)", REFERENCES["pri"]),
    ]
    feats += [
        Feature(f"a_{n}", "1", TRANSMISSION, f"A{n} = log10(E{n} / S{n})", REFERENCES["beer"])
        for n in (660, 710, 940)
    ]
    feats += [
        Feature(
            "m660",
            "1",
            TRANSMISSION,
            "A660 - A940",
            "Chlorophyll absorbance, 940 nm path-length reference",
        ),
        Feature(
            "m710",
            "1",
            TRANSMISSION,
            "A710 - A940",
            "Red-edge absorbance, 940 nm path-length reference",
        ),
    ]
    return feats


def _positional(cfg: StudyConfig) -> list[Feature]:
    pos = set(cfg.leaf_positions)
    feats = []
    if {"oldest", "youngest"} <= pos:
        feats.append(
            Feature(
                "pos_m710_age",
                "1",
                TRANSMISSION,
                "M710_oldest - M710_youngest",
                "N mobile (old leaves) vs Fe immobile (young leaves)",
            )
        )
        if "middle" in pos:
            feats.append(
                Feature(
                    "pos_m710_middle",
                    "1",
                    TRANSMISSION,
                    "M710_middle - (M710_oldest + M710_youngest) / 2",
                )
            )
    if {"lesion", "off_lesion"} <= pos:
        feats.append(
            Feature(
                "pos_m710_lesion",
                "1",
                TRANSMISSION,
                "M710_off_lesion - M710_lesion",
                "Local lesion vs systemic chlorosis",
            )
        )
    feats.append(
        Feature(
            "pos_m710_variance", "1", TRANSMISSION, "Population variance of M710 across positions"
        )
    )
    return feats


def _thermal() -> list[Feature]:
    return [
        Feature("thermal_dt", "degC", THERMAL, "T_leaf_mean - T_air"),
        Feature(
            "thermal_cwsi", "1", THERMAL, "(T_leaf - T_wet) / (T_dry - T_wet)", REFERENCES["cwsi"]
        ),
        Feature("thermal_spread", "degC", THERMAL, "T_leaf_p90 - T_leaf_p10"),
        Feature(
            "vpd_kpa",
            "kPa",
            "Air T/RH",
            "0.6108 exp(17.27 T / (T + 237.3)) (1 - RH / 100)",
            REFERENCES["vpd"],
        ),
    ]


def feature_registry(cfg: StudyConfig) -> list[Feature]:
    """Every feature column for a preset, in the order ``build_features`` emits them."""
    base = _spectral()
    if 970 in cfg.transmission.leds_nm:
        base.append(Feature("w970", "1", TRANSMISSION, "A970 - A940", REFERENCES["water"]))
    base += _positional(cfg)
    if cfg.sensors.thermal:
        base += _thermal()
    first = cfg.design.days.first
    delta = [
        Feature(
            f"{f.name}_delta",
            f.unit,
            f.sensor,
            f"{f.name}(day) - {f.name}(day {first})",
            "Change from baseline",
        )
        for f in base
    ]
    slope = [
        Feature(
            f"{f.name}_slope",
            f"{f.unit}/day" if f.unit != "1" else "1/day",
            f.sensor,
            f"OLS slope of {f.name} over last {SLOPE_WINDOW} measurement days up to this day",
        )
        for f in base
    ]
    return base + delta + slope


def feature_names(cfg: StudyConfig) -> list[str]:
    return [f.name for f in feature_registry(cfg)]
