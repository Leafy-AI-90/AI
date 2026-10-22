from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import leafy_ai
from leafy_ai.config import AS7265X_BANDS, PRESETS, StudyConfig, config_dir, load_config

LEGACY_CLASSES = ("control", "n_deficiency", "fe_deficiency", "xcc", "n_deficiency_xcc")


def raw(preset):
    return yaml.safe_load((config_dir() / "presets" / f"{preset}.yaml").read_text())


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_loads_with_study_design(preset):
    cfg = load_config(preset)
    assert cfg.preset == preset
    assert cfg.n_classes == 5
    assert cfg.n_classification_plants == 360
    assert cfg.design.calibration_plants == 60
    assert cfg.spectral.bands_nm == AS7265X_BANDS
    assert cfg.repeats == 3


def test_default_classes_stay_backward_compatible():
    assert leafy_ai.CLASSES == LEGACY_CLASSES
    assert leafy_ai.BANDS == AS7265X_BANDS
    assert load_config().preset == "xcc"


def test_preset_specific_sensors_and_endpoints():
    root, xcc, salt = (load_config(p) for p in ("root_disease", "xcc", "salt_planb"))
    assert root.sensors.thermal and root.sensors.root_images
    assert 970 in root.transmission.leds_nm
    assert root.design.endpoint_day == 10 and xcc.design.endpoint_day == 8
    assert not xcc.sensors.thermal and not salt.sensors.thermal
    assert "salt" in salt.classes and "water_stress" in root.classes


def test_custom_yaml_path(tmp_path):
    data = raw("xcc") | {"preset": "custom", "leaf_positions": ["oldest", "middle", "youngest"]}
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_config(path).leaf_positions == ("oldest", "middle", "youngest")


@pytest.mark.parametrize(
    "change",
    [
        {"classes": ["n_deficiency", "xcc"]},
        {"classes": ["control", "control"]},
        {"leaf_positions": ["oldest", "youngest"]},
        {"leaf_positions": ["oldest", "youngest", "top"]},
        {"transmission": {"leds_nm": [660, 940]}},
        {"transmission": {"leds_nm": [660, 710, 940, 1050]}},
        {"spectral": {"sensor": "AS7265x", "bands_nm": [435, 410]}},
        {"spectral": {"sensor": "AS7265x", "bands_nm": [415]}},
        {"key_confusion_pairs": [["n_deficiency", "salt"]]},
        {"unknown_key": 1},
    ],
)
def test_invalid_configs_rejected(change):
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(raw("xcc") | change)


def test_endpoint_outside_measurement_days_rejected():
    data = raw("xcc")
    data["design"] = data["design"] | {"endpoint_day": 15}
    with pytest.raises(ValidationError):
        StudyConfig.model_validate(data)


def test_unknown_preset_and_missing_config_dir(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        load_config("not_a_preset")
    monkeypatch.setenv("LEAFY_AI_CONFIG_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        config_dir()


def test_config_dir_override(monkeypatch, tmp_path):
    (tmp_path / "presets").mkdir()
    monkeypatch.setenv("LEAFY_AI_CONFIG_DIR", str(tmp_path))
    assert config_dir() == Path(tmp_path)
