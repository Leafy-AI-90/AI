# Configs

`presets/` holds one YAML file per study design. `leafy_ai.config.load_config`
validates each file with pydantic and rejects unknown keys.

| Preset | Classes | Extra sensors | Endpoint day |
|---|---|---|---|
| `root_disease` | control, n_deficiency, water_stress, root_disease, n_deficiency_root_disease | thermal (MLX90640), 970 nm LED, root photos | 10 |
| `xcc` (default) | control, n_deficiency, fe_deficiency, xcc, n_deficiency_xcc | none | 8 |
| `salt_planb` | control, n_deficiency, fe_deficiency, salt, n_deficiency_salt | none | 8 |

```python
from leafy_ai.config import load_config

cfg = load_config("root_disease")      # preset name
cfg = load_config("my_variant.yaml")   # or a path to a custom YAML
cfg.classes, cfg.design.endpoint_day, cfg.sensors.thermal
```

Validation rules:

- At least two unique classes, including `control`
- `bands_nm` strictly increasing subset of AS7265x bands
- `leds_nm` strictly increasing, includes 660, 710 and 940 nm, optional 970 nm
- 3 or 4 unique leaf positions from oldest, youngest, middle, lesion, off_lesion
- `endpoint_day` after `days.first` and no later than `days.last`
- Each `key_confusion_pairs` entry names two different configured classes

Root disease and salt leaf positions are placeholders until design is fixed with
TLL mentor. Change YAML, not code. Experiment YAML files arrive in Phase 5.
