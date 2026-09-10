"""
contracts.py — Core Contracts v1.1 (frozen)

Single source of truth for the multi-pattern plugin architecture
(REVERSAL_PATTERN_ENGINE_SPEC_v1.1.md §2.1, §3, §9.2).

Frozen on 2026-09-07 per spec §13 Agent 1.  Any change to these
dataclasses / the abstract base must go through a reviewed PR with
Bias-Auditor sign-off (spec §16 — freeze after Agent 1).

Guarantees enforced here:
  * PatternEvent carries the full v1.1 identity + lifecycle fields
    (lifecycle_state, confluence_group_id).
  * PatternFeature declares ``available_at`` and forbids future data;
    runtime validation lives in ``causal_checks.validate_causality``.
  * BasePatternDetector mandates a ``version`` inside the default config
    and exposes ``validate_causality`` + a short registration name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

# ---------------------------------------------------------------------------
# Feature availability semantics (§3.4) — the stamp at which a feature's
# underlying data is known to the market.
# ---------------------------------------------------------------------------
AVAILABLE_AT_DETECT = "detect"
AVAILABLE_AT_CONFIRM = "confirm"
AVAILABLE_AT_ENTRY = "entry"
AVAILABLE_AT_VALUES: tuple[str, str, str] = (
    AVAILABLE_AT_DETECT,
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_ENTRY,
)

# Lifecycle states (§5.1)
LIFECYCLE_TRAINED = "trained"
LIFECYCLE_VALIDATED = "validated"
LIFECYCLE_SHADOW = "shadow"
LIFECYCLE_LIVE = "live"
LIFECYCLE_DEGRADED = "degraded"
LIFECYCLE_RETIRED = "retired"
LIFECYCLE_STATES: tuple[str, ...] = (
    LIFECYCLE_TRAINED,
    LIFECYCLE_VALIDATED,
    LIFECYCLE_SHADOW,
    LIFECYCLE_LIVE,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_RETIRED,
)

# Direction vocabulary (event-side, retained for backwards compatibility
# with the sweep detector's bullish/bearish convention).
DIRECTION_BULLISH = "bullish"
DIRECTION_BEARISH = "bearish"


@dataclass
class PatternEvent:
    """One detected pattern instance — the unit of work of the engine.

    Spec §2.1.  All timing fields are strictly causal: ``detect_time`` is
    when the setup is recognised, ``confirm_time`` when confirmation fires,
    ``known_at`` = max(detect, confirm) is the moment the MARKET knows the
    event (features used for scoring must be available at or before it).
    """

    # --- Identity ---
    event_id: str
    pattern_name: str                    # "liquidity_sweep", "double_bottom", ...
    pattern_version: str
    symbol: str
    timeframe: str
    direction: str                       # "bullish" | "bearish"

    # --- Timing (causal) ---
    detect_time: pd.Timestamp            # setup recognition time
    confirm_time: pd.Timestamp | None = None  # confirmation (reclaim / neckline break)
    entry_time: pd.Timestamp | None = None    # planned entry time
    known_at: pd.Timestamp | None = None      # = max(detect, confirm) — set by validate_causality if None

    # --- Price levels ---
    entry_price: float = float("nan")
    stop_price: float = float("nan")
    target_price: float | None = None
    structure_levels: dict[str, float] = field(default_factory=dict)  # neckline, sweep_low, ...

    # --- Scores ---
    rule_score: float = 0.0              # rule-quality of the pattern (0-1)
    model_prob: float | None = None   # calibrated tier-2 probability

    # --- Lineage & audit (§6.2) ---
    config_hash: str = ""
    feature_schema_version: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)  # pattern-specific + discard_reason

    # --- Lifecycle / correlation (v1.1) ---
    lifecycle_state: str = LIFECYCLE_LIVE   # "shadow" | "live" → PendingSignal
    confluence_group_id: str | None = None   # assigned by CorrelationManager (§4)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def known_at_ts(self) -> pd.Timestamp:
        """The causal availability stamp of the event."""
        if self.known_at is not None:
            return self.known_at
        stamps = [self.detect_time] + (
            [self.confirm_time] if self.confirm_time is not None else []
        )
        return max(stamps)

    def validate_timing(self) -> None:
        """Assert the causal ordering §3.1 without feature checks.

        Raises ``ValueError`` on an incoherent timeline so a detector bug
        (detect after confirm, NaN timestamps...) is caught early.
        """
        if pd.isna(self.detect_time):
            raise ValueError(f"{self.event_id}: detect_time must be set")
        if self.confirm_time is not None and self.confirm_time < self.detect_time:
            raise ValueError(
                f"{self.event_id}: confirm_time {self.confirm_time} "
                f"< detect_time {self.detect_time}"
            )


@dataclass(frozen=True)
class PatternFeature:
    """Declaration of one feature a detector consumes for scoring.

    ``available_at`` gates at which moment the feature may first be used
    (§3.4 runtime check).  ``uses_future_data`` must always be False —
    both CI and the runtime validator enforce this.
    """

    name: str
    dtype: str                                # "float", "int", "bool", "str"
    available_at: str = AVAILABLE_AT_CONFIRM  # detect | confirm | entry
    uses_future_data: bool = False            # must always be False
    description: str = ""

    def __post_init__(self) -> None:
        if self.available_at not in AVAILABLE_AT_VALUES:
            raise ValueError(
                f"{self.name}: available_at must be one of {AVAILABLE_AT_VALUES}, "
                f"got {self.available_at!r}"
            )
        if self.uses_future_data:
            raise ValueError(
                f"{self.name}: uses_future_data must be False in a causal pipeline"
            )


class BasePatternDetector(ABC):
    """Abstract contract every pattern plugin must implement.

    Spec §2.1.  ``name`` / ``version`` / ``feature_schema`` are class-level;
    ``get_default_config()`` must contain the ``"version"`` key (§6.2) so a
    config never silently changes meaning — semantic changes bump the version.
    """

    #: Canonical pattern name (matches ``pattern_name`` in PatternEvent and
    #: the ``pattern_name`` field in the model registry §5.5).
    name: str
    #: Detector semantic version (bumped on any behavior change).
    version: str
    #: Short unique registration code (2-4 chars) for order comments §9.2.
    short_name: str = ""
    #: Feature schema consumed for scoring (runtime-validated §3.4).
    feature_schema: ClassVar[list[PatternFeature]] = []

    @abstractmethod
    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        """Run detection on an OHLCV frame and return pattern events.

        ``df`` must have columns ``open``, ``high``, ``low``, ``close``
        (and optionally ``volume``) with a DatetimeIndex.  The implementation
        MUST NOT read any bar at or after ``known_at`` of each event for
        scoring features (causal rules §3).
        """

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Return the default detector config — MUST contain key "version".

        Example:
            >>> cfg = detector.get_default_config()
            >>> assert cfg["version"] == detector.version
        """

    # ------------------------------------------------------------------
    # Runtime causal validation (§3.4)
    # ------------------------------------------------------------------
    def validate_causality(self, events: list[PatternEvent]) -> None:
        """Runtime check: known_at >= confirm_time >= detect_time, and every
        scoring feature has ``available_at <= known_at`` / no future data.

        Raises :class:`research.core.causal_checks.CausalityViolation` on the
        first offending event.  Delegates to
        :func:`research.core.causal_checks.validate_causality`; detectors may
        override to add pattern-specific checks, but should still call
        ``super().validate_causality(events)`` first.
        """
        from research.core.causal_checks import validate_causality

        validate_causality(events, feature_schema=self.feature_schema)

    # ------------------------------------------------------------------
    # Plugin registration metadata
    # ------------------------------------------------------------------
    @classmethod
    def registry_key(cls) -> str:
        """Discovery key used by ``pattern_registry`` (spec §2.2)."""
        return f"{cls.name}@{cls.version}"


# ---------------------------------------------------------------------------
# Order-comment short-name registry (§9.2) — extend here as patterns land.
# ---------------------------------------------------------------------------
PATTERN_SHORT_NAMES: dict[str, str] = {
    "liquidity_sweep": "LSW",
    # "double_bottom": "DB",   # reserved by spec §9.2
    # "double_top":    "DT",
    # "rising_wedge":  "RW",
    # "falling_wedge": "FW",
    # "head_shoulders": "HS",
}


def order_comment_for(event: PatternEvent) -> str:
    """§9.2 standardized order comment: ``{short}-v{major}-{event_id_short}``."""
    short = PATTERN_SHORT_NAMES.get(event.pattern_name, "PAT")
    major = event.pattern_version.split(".")[0]
    id_short = event.event_id[-4:] if len(event.event_id) >= 4 else event.event_id
    return f"{short}-v{major}-{id_short}"