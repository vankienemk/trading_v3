"""Project-wide configuration loading and access.

Configuration is split across YAML files under ``configs/``.  The loader merges
a baseline file with any number of overrides, deep-merging nested dicts so that
a partial override only changes the keys it mentions.  A single merged
``dict`` is exposed as ``cfg`` for convenience.
"""

from __future__ import annotations

import copy
import os
from typing import Any

_yaml: Any

try:  # pragma: no cover - depends on environment
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or merged."""


def _load_yaml(path: str) -> dict[str, Any]:
    if _yaml is None:
        raise ConfigError(
            "PyYAML is not installed and is required to read config files."
        )
    with open(path, encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    base: str = "baseline.yaml",
    overrides: list[str] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Load and merge configuration files from the ``configs/`` directory.

    With ``validate=True`` (default) the merged config is checked against the
    declarative spec in :mod:`src.config_schema` and a :class:`ConfigError` is
    raised for any missing key or invalid value.
    """
    merged: dict[str, Any] = {}
    names = [base, *list(overrides or [])]
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(CONFIG_DIR, name)
        if not os.path.exists(path):
            raise ConfigError(f"Config file not found: {path}")
        merged = deep_merge(merged, _load_yaml(path))
    merged["_config_dir"] = CONFIG_DIR
    merged["_root_dir"] = ROOT_DIR
    if validate:
        from .config_schema import validate_config

        validate_config(merged)
    return merged


def resolve_path(cfg: dict[str, Any], rel: str) -> str:
    """Resolve a config-relative path against the project root."""
    return os.path.join(cfg["_root_dir"], rel) if not os.path.isabs(rel) else rel


cfg: dict[str, Any] = {}


def set_global_config(config: dict[str, Any]) -> None:
    """Store the currently active merged config globally."""
    global cfg
    cfg = config
