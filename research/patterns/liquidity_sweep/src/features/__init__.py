"""Feature engineering (Agent 4) — causal event-level feature pipeline.

Public API (INTERFACES.md §5):

* :func:`src.features.feature_pipeline.build_event_features` — the feature
  matrix builder;
* :func:`src.features.registry.load_features` /
  :func:`src.features.registry.write_feature_schema_json` — the registry
  contract and its machine-readable ``features.json`` twin.
"""

from __future__ import annotations

from .feature_pipeline import FeaturePipelineError, build_event_features
from .registry import (
    FeatureRegistryError,
    load_features,
    parse_feature_list,
    registered_feature_names,
    write_feature_schema_json,
)
from .sessions import session_flags, session_specs

__all__ = [
    "FeaturePipelineError",
    "FeatureRegistryError",
    "build_event_features",
    "load_features",
    "parse_feature_list",
    "registered_feature_names",
    "session_flags",
    "session_specs",
    "write_feature_schema_json",
]
