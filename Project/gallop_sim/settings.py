"""
settings.py - Loads and validates the project configuration.

config.py (in the project root) holds every tunable constant as an UPPERCASE
module-level name. This module copies those values into a SimpleNamespace,
applies any overrides (from the command line in main.py, or from tests), checks
them for consistency, and resolves output paths. Every other gallop_sim module
receives this namespace rather than importing config.py directly, so a run's
settings can be changed without editing the file and are saved with each run.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_MODELS = ("den_hartog", "kinematic")
VALID_WEATHER = ("none", "rain", "snow")
VALID_SKY = ("clear", "overcast")


def load_config(overrides=None):
    """Return a validated config namespace built from config.py with optional overrides applied.

    `overrides` maps UPPERCASE config names to new values; unknown names raise KeyError so
    typos are caught instead of silently ignored.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import config as config_module  # noqa: E402  (project-root module)

    values = {name: copy.deepcopy(getattr(config_module, name))
              for name in dir(config_module) if name.isupper()}
    for name, value in (overrides or {}).items():
        if name not in values:
            raise KeyError(f"Unknown config setting '{name}'")
        values[name] = value
    cfg = SimpleNamespace(**values)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Raise ValueError if config values are missing, out of range, or inconsistent with each other.

    Checking up front gives a clear message instead of a confusing failure halfway through a render.
    """
    n_modes = len(cfg.MODE_NUMBERS)
    if n_modes == 0:
        raise ValueError("MODE_NUMBERS must contain at least one mode")
    if any(int(n) != n or n < 1 for n in cfg.MODE_NUMBERS):
        raise ValueError("MODE_NUMBERS must be positive integers")
    if len(cfg.MODE_AERO_WEIGHTS) != n_modes:
        raise ValueError("MODE_AERO_WEIGHTS must have one entry per MODE_NUMBERS entry")
    if len(cfg.KINEMATIC_AMPLITUDES_M) != n_modes:
        raise ValueError("KINEMATIC_AMPLITUDES_M must have one entry per MODE_NUMBERS entry")
    if cfg.MOTION_MODEL not in VALID_MODELS:
        raise ValueError(f"MOTION_MODEL must be one of {VALID_MODELS}")
    if cfg.WEATHER not in VALID_WEATHER:
        raise ValueError(f"WEATHER must be one of {VALID_WEATHER}")
    if cfg.SKY_STYLE not in VALID_SKY:
        raise ValueError(f"SKY_STYLE must be one of {VALID_SKY}")
    if cfg.DURATION_S <= 0 or cfg.FPS <= 0:
        raise ValueError("DURATION_S and FPS must be positive")
    if cfg.ALERT_CLEAR_M > cfg.ALERT_THRESHOLD_M:
        raise ValueError("ALERT_CLEAR_M must not exceed ALERT_THRESHOLD_M")
    if not cfg.CONDUCTORS:
        raise ValueError("CONDUCTORS must list at least one conductor")
    names = [c["name"] for c in cfg.CONDUCTORS]
    if len(set(names)) != len(names):
        raise ValueError("Conductor names must be unique")
    for spec in cfg.CONDUCTORS:
        for key in ("name", "attach_y_m", "attach_z_m"):
            if key not in spec:
                raise ValueError(f"Conductor entry {spec} is missing '{key}'")
    for label, schedule in (("WIND_SCHEDULE", cfg.WIND_SCHEDULE), ("KINEMATIC_ENVELOPE", cfg.KINEMATIC_ENVELOPE)):
        times = [p[0] for p in schedule]
        if not schedule or times != sorted(times):
            raise ValueError(f"{label} must be a non-empty list of (time, value) pairs in time order")
    if not 0.0 <= cfg.FOG_DENSITY <= 1.0:
        raise ValueError("FOG_DENSITY must be between 0 and 1")


def resolve_output_dir(cfg):
    """Return the absolute output directory, interpreting relative paths against the project folder."""
    out = Path(cfg.OUTPUT_DIR)
    return out if out.is_absolute() else PROJECT_ROOT / out


def config_to_dict(cfg):
    """Return the config as a JSON-friendly dict (tuples become lists) for saving with each run."""
    def convert(value):
        """Recursively turn tuples into lists so the value serialises cleanly to JSON."""
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value
    return {name: convert(value) for name, value in sorted(vars(cfg).items())}
