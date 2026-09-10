"""
config_hash.py — Configuration hash convention (§6.2)

``config_hash = sha1(canonical_json(detector_config) + detector.version
                     + shared_indicators_version)[:12]``

* ``get_default_config()`` MUST contain the ``"version"`` key — a config
  whose semantics change must bump that version (never silently mutate).
* Every dataset row, every PatternEvent, and every model registry entry
  records its ``config_hash`` for absolute reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from research.core.contracts import BasePatternDetector, PatternEvent

#: Version of the shared indicator layer (ATR, pivot, trendline modules).
#: Bump only when a shared indicator's definition changes across the board.
SHARED_INDICATORS_VERSION = "v1.0"

_HASH_LEN = 12


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialization (§6.2): sorted keys, compact separators.

    This is the deterministic text that feeds the hash — floats, ints and
    strings must round-trip the same way regardless of insertion order.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(o: Any) -> Any:
    """Best-effort fallback for non-JSON-native config leaf values."""
    if hasattr(o, "isoformat"):  # datetime / Timestamp
        return o.isoformat()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"config value of type {type(o)!r} cannot be hashed")


def compute_config_hash(
    detector_config: dict[str, Any],
    detector_version: str | None = None,
    indicators_version: str = SHARED_INDICATORS_VERSION,
) -> str:
    """Compute the canonical config hash (§6.2).

    When *detector_version* is omitted it is read from
    ``detector_config["version"]`` — which MUST exist — so a semantic
    change forces a hash change through the version bump.
    """
    if detector_version is None:
        version = detector_config.get("version")
        if version is None:
            raise ValueError(
                "compute_config_hash: detector_config must contain 'version' "
                "key (spec §6.2) — config semantics must be versioned"
            )
        detector_version = str(version)
    payload = canonical_json(detector_config) + detector_version + indicators_version
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[: _HASH_LEN]


def config_hash_for_detector(detector: BasePatternDetector, config: dict[str, Any]) -> str:
    """Config hash for a detector instance + resolved config."""
    return compute_config_hash(
        config,
        detector_version=detector.version,
        indicators_version=SHARED_INDICATORS_VERSION,
    )


def set_event_config_hash(
    events: list[PatternEvent],
    config_hash: str,
    feature_schema_version: str,
) -> None:
    """Stamp lineage audit fields on a batch of events (§6.2)."""
    for ev in events:
        ev.config_hash = config_hash
        ev.feature_schema_version = feature_schema_version


if __name__ == "__main__":  # pragma: no cover
    # Minimal self-check
    cfg = {"sweep": {"min_penetration_atr": 0.05}, "version": "2.0"}
    print(compute_config_hash(cfg))