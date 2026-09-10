"""
no_lookahead_base.py — Inheritable no-lookahead test template (§3.4).

Every pattern plugin's test suite subclasses :class:`NoLookaheadTestBase`:

    class TestLiquiditySweepCausality(NoLookaheadTestBase):
        feature_schema = [PatternFeature(name="sweep_penetration", ...), ...]
        detector_class = LiquiditySweepDetector   # optional

        def build_events(self) -> List[PatternEvent]:
            det = self.detector_class()           # or a convenience constructor
            return det.detect(df, det.get_default_config())

The inherited ``test_*`` methods then run automatically under pytest and
fail loudly if the detector leaks future information:

  * every event's timeline is coherent (known_at >= confirm >= detect);
  * ``validate_causality(events, feature_schema)`` passes for the batch;
  * every declared feature has ``available_at`` resolvable at or before
    the event's ``known_at`` (no look-ahead) and never ``uses_future_data``;
  * when ``detector_class`` is provided, its default config is versioned
    and yields a valid ``config_hash`` (§6.2).

This is the CI guarantee demanded by REVERSAL_PATTERN_ENGINE_SPEC_v1.1
§3.4 / §13: a detector whose events violate causality MUST fail the suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from research.core.causal_checks import (
    CausalityViolation,
    resolve_feature_stamp,
    validate_causality,
    validate_event_timeline,
)
from research.core.config_hash import compute_config_hash
from research.core.contracts import BasePatternDetector, PatternEvent, PatternFeature


class NoLookaheadTestBase:
    """Pytest base class enforcing causal event output on pattern suites.

    Subclasses MUST override :meth:`build_events` and MUST set the
    ``feature_schema`` class attribute (an empty schema only disables the
    feature-availability gate, while the timeline gate always applies).
    ``detector_class`` is optional; when set, an additional
    config-hash / causality-through-the-detector-API test is inherited.
    """

    #: Feature schema the detector consumes for scoring (§3.4).
    feature_schema: ClassVar[Sequence[PatternFeature]] = []

    #: Optional detector class (constructor with no args) to validate the
    #: versioned default config + causality through the plugin API.
    detector_class: ClassVar[type[BasePatternDetector] | None] = None

    def build_events(self) -> list[PatternEvent]:
        """Return the events the detector produced on real data."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_events()"
        )

    # ------------------------------------------------------------------
    # Inherited tests — collected by pytest in every subclass
    # ------------------------------------------------------------------

    def test_events_not_empty(self) -> None:
        events = self.build_events()
        assert events, "build_events() must return at least one event"

    def test_all_events_timeline_coherent(self) -> None:
        """known_at >= confirm_time >= detect_time for every event."""
        for ev in self.build_events():
            validate_event_timeline(ev)

    def test_causality_validation_passes(self) -> None:
        """Runtime validator must accept the full batch (§3.4)."""
        events = self.build_events()
        validate_causality(events, feature_schema=self.feature_schema)

    def test_no_feature_declares_future_data(self) -> None:
        """The schema itself must not smuggle future data."""
        for feat in self.feature_schema:
            assert not feat.uses_future_data, (
                f"feature '{feat.name}' declares uses_future_data=True"
            )

    def test_all_feature_stamps_within_known_at(self) -> None:
        """No feature may read a bar at or after the event's known_at."""
        events = self.build_events()
        for ev in events:
            known = ev.known_at_ts
            for feat in self.feature_schema:
                try:
                    stamp = resolve_feature_stamp(feat, ev)
                except CausalityViolation as exc:  # missing detect/confirm/entry stamp
                    raise AssertionError(
                        f"event {ev.event_id} cannot resolve feature "
                        f"'{feat.name}': {exc}"
                    ) from exc
                assert stamp <= known, (
                    f"event {ev.event_id}: feature '{feat.name}' available_at "
                    f"{stamp} > known_at {known} — would read future data"
                )

    def test_detector_default_config_versioned(self) -> None:
        """Detector default config MUST carry the version + hashable (§6.2)."""
        if self.detector_class is None:
            pytest.skip("detector_class not provided — skipping config gate")
        det = self.detector_class()
        cfg = det.get_default_config()
        assert cfg.get("version") == det.version, (
            f"{type(det).__name__}: get_default_config()['version'] must "
            f"equal detector.version ({det.version})"
        )
        h = compute_config_hash(cfg)
        assert len(h) == 12, "config_hash must be 12 hex chars (§6.2)"

    def test_detector_api_validates_own_events(self) -> None:
        """The detector's own validate_causality path must accept its events."""
        if self.detector_class is None:
            pytest.skip("detector_class not provided — skipping API gate")
        det = self.detector_class()
        det.validate_causality(self.build_events())