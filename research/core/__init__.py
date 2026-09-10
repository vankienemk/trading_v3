"""research.core — shared infrastructure for the multi-pattern engine.

Modules:
  * contracts        — v1.1 frozen core contracts (§2.1)
  * causal_checks    — runtime causality validator (§3.4)
  * config_hash      — canonical config hashing (§6.2)
  * pattern_synthesizer — ground-truth OHLC generator (§8) [Agent 8]
"""

from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    AVAILABLE_AT_ENTRY,
    AVAILABLE_AT_VALUES,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_LIVE,
    LIFECYCLE_RETIRED,
    LIFECYCLE_SHADOW,
    LIFECYCLE_STATES,
    LIFECYCLE_TRAINED,
    LIFECYCLE_VALIDATED,
    PATTERN_SHORT_NAMES,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
    order_comment_for,
)

__all__ = [
    "AVAILABLE_AT_CONFIRM",
    "AVAILABLE_AT_DETECT",
    "AVAILABLE_AT_ENTRY",
    "AVAILABLE_AT_VALUES",
    "LIFECYCLE_DEGRADED",
    "LIFECYCLE_LIVE",
    "LIFECYCLE_RETIRED",
    "LIFECYCLE_SHADOW",
    "LIFECYCLE_STATES",
    "LIFECYCLE_TRAINED",
    "LIFECYCLE_VALIDATED",
    "PATTERN_SHORT_NAMES",
    "BasePatternDetector",
    "PatternEvent",
    "PatternFeature",
    "order_comment_for",
]