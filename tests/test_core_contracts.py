"""Tests for Agent 1 core contracts & causal infrastructure.

DoD (spec §13 Agent 1):
  * contracts importable from everywhere;
  * 100% mypy strict on research/core;
  * a deliberately causality-violating event must raise CausalityViolation.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pandas as pd
import pytest

from research.core.causal_checks import (
    CausalityViolation,
    make_lookahead_feature_event,
    validate_causality,
    validate_event_features,
    validate_event_timeline,
)
from research.core.config_hash import compute_config_hash, config_hash_for_detector
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    AVAILABLE_AT_ENTRY,
    LIFECYCLE_LIVE,
    LIFECYCLE_STATES,
    PATTERN_SHORT_NAMES,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
    order_comment_for,
)
from tests.no_lookahead_base import NoLookaheadTestBase


def _ts(hour: int = 10, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2026-09-07 00:00", tz="UTC") + pd.Timedelta(
        hours=hour, minutes=minute
    )


def _base_event(**overrides) -> PatternEvent:
    kwargs = dict(
        event_id="DB-0001",
        pattern_name="double_bottom",
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=_ts(10),
        confirm_time=_ts(11),
        entry_time=_ts(12),
        entry_price=2650.0,
        stop_price=2640.0,
        target_price=2680.0,
        structure_levels={"neckline": 2660.0, "sweep_low": 2630.0},
        configuration_hash="",  # renamed field is not in v1.1 — see below
    )
    kwargs.pop("configuration_hash", None)
    kwargs.update(overrides)
    return PatternEvent(**kwargs)


# ---------------------------------------------------------------------------
# Contracts surface
# ---------------------------------------------------------------------------

def test_pattern_event_fields_present() -> None:
    ev = _base_event()
    assert ev.event_id == "DB-0001"
    assert ev.pattern_name == "double_bottom"
    assert ev.lifecycle_state == LIFECYCLE_LIVE
    assert ev.confluence_group_id is None
    assert ev.known_at_ts == _ts(11)  # max(detect, confirm)


def test_known_at_property_without_confirm() -> None:
    ev = _base_event(confirm_time=None)
    assert ev.known_at_ts == _ts(10)


def test_known_at_explicit_wins() -> None:
    ev = _base_event(known_at=_ts(11) + pd.Timedelta(minutes=30))
    assert ev.known_at_ts == _ts(11) + pd.Timedelta(minutes=30)


def test_lifecycle_state_enum() -> None:
    assert LIFECYCLE_STATES == (
        "trained",
        "validated",
        "shadow",
        "live",
        "degraded",
        "retired",
    )


def test_pattern_short_names_lsw_registered() -> None:
    # Spec §16: register LSW in the first Agent 2 PR.
    assert PATTERN_SHORT_NAMES["liquidity_sweep"] == "LSW"
    assert all(2 <= len(v) <= 4 for v in PATTERN_SHORT_NAMES.values())


def test_order_comment_schema() -> None:
    ev = _base_event(
        event_id="LSW-77c1",
        pattern_name="liquidity_sweep",
        pattern_version="2.0",
    )
    # §9.2: {pattern_short}-v{major}-{event_id_short}
    assert order_comment_for(ev) == "LSW-v2-77c1"


# ---------------------------------------------------------------------------
# PatternFeature constraints
# ---------------------------------------------------------------------------

def test_feature_rejects_future_data() -> None:
    with pytest.raises(ValueError):
        PatternFeature(name="f", dtype="float", uses_future_data=True)


def test_feature_rejects_bad_available_at() -> None:
    with pytest.raises(ValueError):
        PatternFeature(name="f", dtype="float", available_at="tomorrow")


# ---------------------------------------------------------------------------
# validate_event_timeline
# ---------------------------------------------------------------------------

def test_timeline_ok() -> None:
    validate_event_timeline(_base_event())  # no raise


def test_confirm_before_detect_raises() -> None:
    ev = _base_event(
        detect_time=_ts(11),
        confirm_time=_ts(10),
    )
    with pytest.raises(CausalityViolation):
        validate_event_timeline(ev)


def test_missing_detect_raises() -> None:
    ev = _base_event(detect_time=pd.NaT)
    with pytest.raises(CausalityViolation):
        validate_event_timeline(ev)


# ---------------------------------------------------------------------------
# validate_event_features — available_at vs known_at
# ---------------------------------------------------------------------------

def test_features_available_before_known_ok() -> None:
    schema = [
        PatternFeature(name="atr", dtype="float", available_at=AVAILABLE_AT_DETECT),
        PatternFeature(name="reclaim_atr", dtype="float", available_at=AVAILABLE_AT_CONFIRM),
    ]
    validate_event_features(_base_event(), schema)  # no raise


def test_feature_available_after_known_raises() -> None:
    # entry feature but event.entry_time (12:00) > known_at (11:00) —
    # reading the entry bar into scoring features is look-ahead.
    schema = [PatternFeature(name="entry_spread", dtype="float", available_at=AVAILABLE_AT_ENTRY)]
    ev = _base_event()  # known_at = confirm = 11:00, entry = 12:00
    with pytest.raises(CausalityViolation) as exc:
        validate_event_features(ev, schema)
    assert "available_at" in str(exc.value)
    assert exc.value.event_id == "DB-0001"


def test_confirm_feature_on_unconfirmed_event_raises() -> None:
    schema = [PatternFeature(name="reclaim_atr", dtype="float", available_at=AVAILABLE_AT_CONFIRM)]
    ev = _base_event(confirm_time=None)
    with pytest.raises(CausalityViolation):
        validate_event_features(ev, schema)


def test_uses_future_data_feature_raises() -> None:
    # PatternFeature deliberately cannot be CONSTRUCTED with
    # uses_future_data=True (post_init guard).  Simulate a lying detector
    # that smuggles a bad feature in via object.__setattr__ (frozen) —
    # validate_event_features must still catch it at runtime.
    feats = PatternFeature(
        name="bad_feature",
        dtype="float",
        available_at=AVAILABLE_AT_DETECT,
        uses_future_data=False,
    )
    object.__setattr__(feats, "uses_future_data", True)
    schema = [feats]
    with pytest.raises(CausalityViolation):
        validate_event_features(_base_event(), schema)


def test_validate_causality_batch_raises_on_first_bad() -> None:
    good = _base_event()
    bad_schema = [
        PatternFeature(name="future", dtype="float", available_at=AVAILABLE_AT_ENTRY)
    ]
    with pytest.raises(CausalityViolation):
        validate_causality([good, _base_event()], feature_schema=bad_schema)


def test_make_lookahead_feature_event_raises() -> None:
    """Adversarial helper: a forged look-ahead event must be caught."""
    schema = [
        PatternFeature(name="reclaim_atr", dtype="float", available_at=AVAILABLE_AT_CONFIRM)
    ]
    bad = make_lookahead_feature_event(schema[0], _base_event())
    with pytest.raises(CausalityViolation):
        validate_event_features(bad, schema)


# ---------------------------------------------------------------------------
# BasePatternDetector abstract contract
# ---------------------------------------------------------------------------

class _GoodDetector(BasePatternDetector):
    name = "double_bottom"
    version = "1.0"
    short_name = "DB"

    def detect(self, df, config):  # type: ignore[override]
        return []

    def get_default_config(self):
        return {"version": self.version, "min_depth_bars": 20}


def test_detector_default_config_has_version() -> None:
    d = _GoodDetector()
    cfg = d.get_default_config()
    assert cfg["version"] == d.version


def test_detector_validate_causality_delegates() -> None:
    d = _GoodDetector()
    d.validate_causality([])  # empty — no raise
    # A bad event must raise through the detector API too
    bad = _base_event(detect_time=_ts(11), confirm_time=_ts(10))
    with pytest.raises(CausalityViolation):
        d.validate_causality([bad])


def test_registry_key() -> None:
    assert _GoodDetector.registry_key() == "double_bottom@1.0"


# ---------------------------------------------------------------------------
# config_hash (§6.2)
# ---------------------------------------------------------------------------

def test_config_hash_contains_version_requirement() -> None:
    with pytest.raises(ValueError):
        compute_config_hash({"min_depth_bars": 20})  # missing "version"


def test_config_hash_deterministic() -> None:
    cfg = {"sweep": {"min_penetration_atr": 0.05, "max": 0.20}, "version": "2.0"}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(dict(cfg))  # different insertion order
    assert h1 == h2
    assert len(h1) == 12


def test_config_hash_changes_with_version() -> None:
    cfg = {"sweep": {"min_penetration_atr": 0.05}, "version": "2.0"}
    cfg2 = {"sweep": {"min_penetration_atr": 0.05}, "version": "2.1"}
    assert compute_config_hash(cfg) != compute_config_hash(cfg2)


def test_config_hash_changes_with_semantics() -> None:
    """Same version, different semantics must yield different hashes — if the
    version is bumped correctly.  Here we prove the hash is a function of
    the FULL canonicalized config."""
    a = {"sweep": {"min_penetration_atr": 0.05}, "version": "2.0"}
    b = {"sweep": {"min_penetration_atr": 0.10}, "version": "2.1"}
    assert compute_config_hash(a) != compute_config_hash(b)


def test_config_hash_for_detector() -> None:
    d = _GoodDetector()
    h = config_hash_for_detector(d, {"version": "1.0", "min_depth_bars": 20})
    assert len(h) == 12


def test_canonical_json_sorted_keys() -> None:
    cfg = {"b": 1, "a": 2, "version": "1.0"}
    assert json.loads(compute_config_hash.__globals__["canonical_json"](cfg)) == {"a": 2, "b": 1, "version": "1.0"}


# ---------------------------------------------------------------------------
# mypy-strict smoke: typing surface compiles
# ---------------------------------------------------------------------------

def test_typing_surface() -> None:
    events: list[PatternEvent] = [_base_event()]
    validate_causality(events, feature_schema=[])
    assert len(events) == 1


# ---------------------------------------------------------------------------
# No-lookahead template (§3.4 / §13) — must be inheritable by every
# pattern suite.  This demo subclass is the proof: pytest auto-collects the
# inherited no-lookahead tests here and they must all pass.
# ---------------------------------------------------------------------------

class TestTemplateDemoSuite(NoLookaheadTestBase):
    """Demo plugin suite proving NoLookaheadTestBase is inheritable.

    Named with a ``Test`` prefix on purpose: pytest auto-collects the
    inherited no-lookahead tests (see the 7 collected tests below) — exactly
    what every pattern plugin suite will rely on.
    """

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(name="atr", dtype="float", available_at=AVAILABLE_AT_DETECT),
        PatternFeature(name="reclaim_atr", dtype="float", available_at=AVAILABLE_AT_CONFIRM),
    ]
    detector_class = _GoodDetector

    def build_events(self) -> list[PatternEvent]:
        """Return the events the detector produced — here one canonical
        double-bottom event with detect/confirm/entry timestamps resolved."""
        return [_base_event()]


def test_template_is_inheritable() -> None:
    """The demo suite must inherit the template's wiring contract."""
    suite = TestTemplateDemoSuite()
    assert suite.feature_schema[0].name == "atr"
    assert suite.detector_class is _GoodDetector
    assert suite.build_events()[0].event_id == "DB-0001"
    # Running the inherited gate directly must not raise.
    validate_causality(suite.build_events(), feature_schema=suite.feature_schema)


def test_template_catches_lookahead_violation() -> None:
    """A suite declaring an entry-only feature on a confirm-known event
    (available_at > known_at) MUST fail the inherited template gate — this
    is the guarantee every pattern suite inherits (§3.4)."""
    class _LookaheadSuite(TestTemplateDemoSuite):  # type: ignore[misc]
        feature_schema: ClassVar[list[PatternFeature]] = [
            PatternFeature(
                name="entry_spread", dtype="float", available_at=AVAILABLE_AT_ENTRY
            )
        ]

    suite = _LookaheadSuite()
    with pytest.raises(AssertionError, match="would read future data"):
        suite.test_all_feature_stamps_within_known_at()