"""
causal_checks.py — Runtime causal validator (§3.4)

Closes the gap between static CI analysis (``no_lookahead`` marker) and
actual runtime behavior.  When a detector returns events, the engine (and
the research pipeline) must run :func:`validate_causality`:

  * every scoring feature's ``available_at <= event.known_at``;
  * ``uses_future_data is False`` on every feature;
  * the event timeline is coherent: ``known_at >= confirm_time >= detect_time``.

A violation raises :class:`CausalityViolation`; the engine must drop the
event and surface an alert on the GUI Log tab (§3.4).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    AVAILABLE_AT_ENTRY,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
)


class CausalityViolation(Exception):
    """Raised when an event or its features violate the causal rules.

    Carries the offending ``event_id`` (when known) so the engine can log
    the exact source.
    """

    def __init__(self, message: str, event_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.event_id = event_id


# ---------------------------------------------------------------------------
# Feature availability stamp resolution
# ---------------------------------------------------------------------------

def _available_at_stamp(
    feature: PatternFeature, event: PatternEvent
) -> pd.Timestamp:
    """Map ``available_at`` onto the event's actual timeline stamp.

    * detect  → event.detect_time
    * confirm → event.confirm_time (must be present)
    * entry   → event.entry_time   (must be present)

    Raises ``CausalityViolation`` when the required stamp is unknown (a
    feature asking for entry-level data on an unconfirmed event).
    """
    if feature.available_at == AVAILABLE_AT_DETECT:
        return event.detect_time
    if feature.available_at == AVAILABLE_AT_CONFIRM:
        if event.confirm_time is None:
            raise CausalityViolation(
                f"feature '{feature.name}' available_at='confirm' but event "
                f"{event.event_id} has no confirm_time",
                event_id=event.event_id,
            )
        return event.confirm_time
    if feature.available_at == AVAILABLE_AT_ENTRY:
        if event.entry_time is None:
            raise CausalityViolation(
                f"feature '{feature.name}' available_at='entry' but event "
                f"{event.event_id} has no entry_time",
                event_id=event.event_id,
            )
        return event.entry_time
    raise CausalityViolation(
        f"feature '{feature.name}' has unknown available_at "
        f"{feature.available_at!r}",
        event_id=event.event_id,
    )


# ---------------------------------------------------------------------------
# Public feature-stamp resolver (§3.4) — single source of truth
# ---------------------------------------------------------------------------

def resolve_feature_stamp(
    feature: PatternFeature, event: PatternEvent
) -> pd.Timestamp:
    """Public wrapper for :func:`_available_at_stamp`.

    Exposed so the inheritable no-lookahead test template
    (``tests/no_lookahead_base.py``) and external tooling can resolve a
    feature's availability stamp against an event without reaching into a
    private helper.  Same semantics and exceptions as the private function.
    """
    return _available_at_stamp(feature, event)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def validate_event_timeline(event: PatternEvent) -> None:
    """Assert ordering: known_at >= confirm_time >= detect_time.

    ``known_at`` is auto-derived (= max over available stamps) when unset,
    which guarantees consistency for a coherent detector output.
    """
    if pd.isna(event.detect_time):
        raise CausalityViolation(
            f"{event.event_id}: detect_time missing", event_id=event.event_id
        )
    if event.confirm_time is not None and event.confirm_time < event.detect_time:
        raise CausalityViolation(
            f"{event.event_id}: confirm_time {event.confirm_time} < "
            f"detect_time {event.detect_time}",
            event_id=event.event_id,
        )
    known = event.known_at_ts
    if event.confirm_time is not None and known < event.confirm_time:
        raise CausalityViolation(
            f"{event.event_id}: known_at {known} < confirm_time "
            f"{event.confirm_time}",
            event_id=event.event_id,
        )


def validate_event_features(
    event: PatternEvent, feature_schema: Sequence[PatternFeature]
) -> None:
    """Assert every feature in *feature_schema* is usable at ``known_at``."""
    known = event.known_at_ts
    for feature in feature_schema:
        if feature.uses_future_data:
            raise CausalityViolation(
                f"{event.event_id}: feature '{feature.name}' declares "
                f"uses_future_data=True (forbidden)",
                event_id=event.event_id,
            )
        try:
            stamp = _available_at_stamp(feature, event)
        except CausalityViolation:  # missing required stamp
            raise
        if stamp > known:
            raise CausalityViolation(
                f"{event.event_id}: feature '{feature.name}' available_at "
                f"'{feature.available_at}' ({stamp}) > known_at ({known}) — "
                f"would read future data",
                event_id=event.event_id,
            )


def validate_causality(
    events: Sequence[PatternEvent],
    feature_schema: Sequence[PatternFeature] | None = None,
) -> None:
    """Full runtime causal validation for a batch of events (§3.4).

    Raises ``CausalityViolation`` on the FIRST offending event — the engine
    catches it per event, drops the event, and alerts the GUI Log tab.
    """
    for event in events:
        validate_event_timeline(event)
        if feature_schema is not None:
            validate_event_features(event, feature_schema)


def validate_detector_events(
    detector: BasePatternDetector, events: list[PatternEvent]
) -> None:
    """Convenience: validate a detector's output against its own schema."""
    validate_causality(events, feature_schema=detector.feature_schema)


# ---------------------------------------------------------------------------
# Adversarial helpers (Agent 7 suite reuses these to build bad events)
# ---------------------------------------------------------------------------

def make_lookahead_feature_event(
    feature: PatternFeature, base_event: PatternEvent
) -> PatternEvent:
    """Return a *copy* of *base_event* whose timeline violates the feature's
    availability (feature stamp > known_at).  Used by adversarial tests to
    prove ``validate_causality`` raises.
    """
    import copy

    bad = copy.deepcopy(base_event)
    stamp = _available_at_stamp(feature, bad)
    # Shift known_at BEFORE the feature stamp → feature reads future data.
    bad.known_at = stamp - pd.Timedelta(hours=1)
    return bad