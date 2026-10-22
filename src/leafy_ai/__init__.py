"""Leafy AI research foundations; no fitted plant-diagnostic weights included."""

from .config import AS7265X_BANDS, DEFAULT_PRESET, load_config

BANDS = AS7265X_BANDS


def __getattr__(name):
    # Backward compatible ``leafy_ai.CLASSES``: class list of the default preset.
    if name == "CLASSES":
        return load_config(DEFAULT_PRESET).classes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
