"""Feature registry loader + machine-readable ``features.json`` schema.

The registry is the single contract for every feature the model consumes
(schema §7 of ``docs/SCHEMAS.md`` and INTERFACES.md §5).  Two physical views:

* the human/editable ``configs/features.yaml`` (the source of truth), and
* the machine-readable ``artifacts/feature_schemas/features.json``, generated
  from the YAML by :func:`write_feature_schema_json` so the two can never
  drift.

Every feature has a ``uses_future_data: false`` contract; the feature pipeline
(:mod:`src.features.feature_pipeline`) must output **exactly** the registered
names (plus ``event_id`` / ``event_time``).  Any divergence raises
:class:`FeatureRegistryError`.

Owned by Agent 4 (feature engineering).  Updates to the registry are a schema
change (rule 30.1): update this file + ``configs/features.yaml`` +
``docs/SCHEMAS.md`` and notify Agent 0.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import yaml

from src.config import ROOT_DIR

#: Path to the editable YAML registry (the source of truth).
FEATURE_REGISTRY_YAML = os.path.join(ROOT_DIR, "configs", "features.yaml")

#: Path the machine-readable JSON schema is written to (config-relative).
FEATURE_SCHEMA_JSON = os.path.join(
    ROOT_DIR, "artifacts", "feature_schemas", "features.json"
)

BANNED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        # Outcome / trade-simulation columns must never be model features.
        "outcome",
        "exit_time",
        "exit_price",
        "mfe_r",
        "mae_r",
        "bars_to_target",
        "bars_to_stop",
        "future_return",
    }
)


class FeatureRegistryError(ValueError):
    """Raised when the registry is malformed or the pipeline diverges from it."""


def _sanitize_range(value: Any, name: str) -> list[Any]:
    """Coerce an ``expected_range`` cell into a 2-element list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise FeatureRegistryError(
                f"feature {name!r}: expected_range must not be empty"
            )
        return list(value)
    # YAML unquoted scalars like `rolling | swing | equal` become a single str.
    if isinstance(value, str):
        return [v.strip() for v in value.split("|") if v.strip()]
    raise FeatureRegistryError(
        f"feature {name!r}: expected_range must be a list or 'a | b | c' string, "
        f"got {value!r}"
    )


def parse_feature_list(raw: list[Any]) -> list[dict[str, Any]]:
    """Validate and normalise a raw registry list into feature dicts.

    Each feature dict carries ``name, group, description, dtype, available_at,
    uses_future_data, expected_range``.  Raises :class:`FeatureRegistryError`
    on a duplicate name, a missing key, a banned feature name, or a non-false
    ``uses_future_data``.
    """
    if not isinstance(raw, list):
        raise FeatureRegistryError("'features' must be a list")

    required = (
        "name",
        "description",
        "dtype",
        "available_at",
        "uses_future_data",
        "expected_range",
    )
    seen: set[str] = set()
    features: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise FeatureRegistryError(f"features[{i}] must be a mapping")
        for key in required:
            if key not in item:
                raise FeatureRegistryError(
                    f"feature {item.get('name', f'[#{i}]')!r} is missing '{key}'"
                )
        name = str(item["name"])
        if name in BANNED_FEATURE_COLUMNS:
            raise FeatureRegistryError(
                f"feature {name!r} is a banned label/outcome column and cannot be "
                "a model feature"
            )
        if name in seen:
            raise FeatureRegistryError(f"duplicate feature name {name!r}")
        seen.add(name)
        if item["uses_future_data"] is not False:
            raise FeatureRegistryError(
                f"feature {name!r}: uses_future_data must be false (rule 30.4)"
            )
        if item["available_at"] not in ("event_time", "confirmation_time"):
            raise FeatureRegistryError(
                f"feature {name!r}: available_at must be 'event_time' or "
                f"'confirmation_time', got {item['available_at']!r}"
            )
        features.append(
            {
                "name": name,
                "group": str(item.get("group", "other")),
                "description": str(item["description"]),
                "dtype": str(item["dtype"]),
                "available_at": str(item["available_at"]),
                "uses_future_data": bool(item["uses_future_data"]),
                "expected_range": _sanitize_range(item["expected_range"], name),
            }
        )
    return features


def load_features() -> list[dict[str, Any]]:
    """Load and validate the feature registry from ``configs/features.yaml``.

    Delegates parsing to :func:`parse_feature_list`.  Returns an ordered list
    of feature dicts.  Raises :class:`FeatureRegistryError` on a malformed
    registry.
    """
    if not os.path.exists(FEATURE_REGISTRY_YAML):
        raise FeatureRegistryError(
            f"Feature registry not found: {FEATURE_REGISTRY_YAML}"
        )
    with open(FEATURE_REGISTRY_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "features" not in data:
        raise FeatureRegistryError(
            "Feature registry must be a mapping with a top-level 'features' list"
        )
    return parse_feature_list(data["features"])


def registered_feature_names(features: list[dict[str, Any]] | None = None) -> list[str]:
    """Ordered list of feature names in the registry."""
    feats = features if features is not None else load_features()
    return [f["name"] for f in feats]


def write_feature_schema_json(
    out_path: str | None = None, features: list[dict[str, Any]] | None = None
) -> str:
    """Write the machine-readable feature schema to ``out_path``.

    The JSON is generated from the in-memory registry (default: loaded from
    YAML) so the YAML stays the single source of truth.  Returns the output
    path that was written.
    """
    feats = features if features is not None else load_features()
    path = out_path if out_path is not None else FEATURE_SCHEMA_JSON
    payload = {
        "schema_version": 1,
        "source_registry": FEATURE_REGISTRY_YAML.replace(ROOT_DIR, "."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feats),
        "features": feats,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


__all__ = [
    "BANNED_FEATURE_COLUMNS",
    "FEATURE_REGISTRY_YAML",
    "FEATURE_SCHEMA_JSON",
    "FeatureRegistryError",
    "load_features",
    "parse_feature_list",
    "registered_feature_names",
    "write_feature_schema_json",
]
