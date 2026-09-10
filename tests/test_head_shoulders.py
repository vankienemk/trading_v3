"""Head & Shoulders / Inverse plugin tests (P4) -- Agent 4 DoD.

Per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §13 Agent 4 / §8.3 / §6.3, plus the two
DoD-critical proofs:

  1. the inheritable no-lookahead template (:class:`NoLookaheadTestBase`) --
     timeline coherence, feature availability, versioned default config;
  2. **§3.2 proof**: the detector NEVER uses a pivot before it causally exists
     -- every event's ``detect_bar == right_shoulder_bar + right_bars`` and
     its ``known_at`` is never before the right-shoulder pivot's known bar;
  3. **§3.3 staleness proof**: a candidate whose neckline cross does not fire
     within ``max_bars_between_detect_and_confirm`` is DISCARDED (a long
     window keeps the event, a short window drops it);
  4. §8.3 Synthesizer acceptance for **both** H&S plugins (recall ≥ 80 %,
     robustness drop ≤ 15 pts, FP ≤ 5 %);
  5. zero ``CausalityViolation`` over the FULL XAUUSD M15 dataset.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from research.core.causal_checks import validate_causality
from research.core.pattern_synthesizer import PatternSynthesizer
from research.patterns.double_bottom.detector import atr_series
from research.patterns.head_shoulders.benchmark import run_hs_benchmark
from research.patterns.head_shoulders.dataset import load_xauusd_m15
from research.patterns.head_shoulders.detector import HeadShouldersDetector
from research.patterns.inverse_head_shoulders.detector import (
    InverseHeadShouldersDetector,
)
from tests.no_lookahead_base import NoLookaheadTestBase

pytestmark = pytest.mark.no_lookahead

_SEEDS = 12


def _planted_frames(pattern: str, noise_sigma: float = 0.05):
    syn = PatternSynthesizer()
    for seed in range(_SEEDS):
        yield syn.generate(pattern, 120, noise_sigma=noise_sigma, seed=seed)


class TestHeadShouldersCausality(NoLookaheadTestBase):
    """Inherited gates: timeline, feature availability, versioned config."""

    feature_schema: ClassVar = HeadShouldersDetector.feature_schema
    detector_class: ClassVar = HeadShouldersDetector

    def build_events(self) -> list:
        det = HeadShouldersDetector()
        cfg = det.get_default_config()
        events = []
        for frame, _gt in _planted_frames("head_shoulders"):
            events.extend(det.detect(frame, cfg))
        return events


class TestInverseHeadShouldersCausality(NoLookaheadTestBase):
    """Inherited gates for the bullish mirror."""

    feature_schema: ClassVar = InverseHeadShouldersDetector.feature_schema
    detector_class: ClassVar = InverseHeadShouldersDetector

    def build_events(self) -> list:
        det = InverseHeadShouldersDetector()
        cfg = det.get_default_config()
        events = []
        for frame, _gt in _planted_frames("inverse_head_shoulders"):
            events.extend(det.detect(frame, cfg))
        return events


# ---------------------------------------------------------------------------
# §3.2 — the detector never uses a pivot before it exists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pattern, detector, right_bars",
    [
        ("head_shoulders", HeadShouldersDetector, 3),
        ("inverse_head_shoulders", InverseHeadShouldersDetector, 3),
    ],
)
def test_detect_bar_is_right_shoulder_known_at(pattern, detector, right_bars) -> None:
    """The pivot/knowing proof: detect fires exactly at the right shoulder's
    causal existence stamp (right_shoulder_bar + right_bars, §3.2), so the
    detector cannot have referenced a not-yet-existing pivot."""
    det = detector()
    cfg = det.get_default_config()
    for df, _gt in _planted_frames(pattern, noise_sigma=0.03):
        events = det.detect(df, cfg)
        if not events:
            continue
        for ev in events:
            rs = int(ev.attributes["right_shoulder_bar"])
            known = int(ev.attributes["pivot_known_at_bar"])
            # detect (setup recognition) must wait the full right-bar window
            assert known == rs + right_bars, (
                f"{ev.event_id}: detect_bar {known} != right_shoulder {rs} + rbars {right_bars}"
            )
            # known_at (data-availability floor) >= the pivot's known bar
            assert ev.known_at_ts >= df.index[rs + right_bars]
            # every feature resolves at/after detect — never before the pivot
            assert ev.detect_time == df.index[known]
    # ensure we actually exercised events
    any_events = any(
        det.detect(df, cfg) for df, _gt in _planted_frames(pattern, noise_sigma=0.03)
    )
    assert any_events, "no H&S events generated to prove causality"


def test_no_head_or_shoulder_referenced_before_its_pivot_exists() -> None:
    """Trace every anchor bar on a planted frame: each swing the detector
    reports as part of the structure is only referenced at/after its own
    causal existence stamp (bar + right_bars, §3.2)."""
    det = HeadShouldersDetector()
    cfg = det.get_default_config()
    rb = int(cfg["right_bars"])
    for df, _gt in _planted_frames("head_shoulders", noise_sigma=0.03):
        events = det.detect(df, cfg)
        for ev in events:
            for anchor in ("left_shoulder_bar", "head_bar", "right_shoulder_bar"):
                bar = int(ev.attributes[anchor])
                # the event's detect/known bars may never precede the minimal
                # confirmation window of ANY of its structure pivots.
                assert ev.known_at_ts >= df.index[bar + rb], (
                    f"{ev.event_id}: known_at before {anchor} pivot known "
                    f"(bar {bar}+{rb}) — would use the pivot it hasn't confirmed"
                )


# ---------------------------------------------------------------------------
# §3.3 — staleness window discards late candidates
# ---------------------------------------------------------------------------

def _staleness_frame(pattern: str, seed: int = 5):
    """A planted H&S frame where the neckline cross is naturally late-ish."""
    return PatternSynthesizer().generate(pattern, 120, noise_sigma=0.03, seed=seed)


@pytest.mark.parametrize(
    "pattern, detector",
    [
        ("head_shoulders", HeadShouldersDetector),
        ("inverse_head_shoulders", InverseHeadShouldersDetector),
    ],
)
def test_staleness_window_discards_late_confirm(pattern, detector) -> None:
    """§3.3: a candidate whose confirm (neckline cross) fires after
    max_bars_between_detect_and_confirm is DISCARDED.  A long window keeps
    the event; a tiny window (no cross in time) drops it."""
    df, _gt = _staleness_frame(pattern)
    base_cfg = detector().get_default_config()

    long_det = detector(dict(base_cfg, max_bars_between_detect_and_confirm=120))
    short_det = detector(dict(base_cfg, max_bars_between_detect_and_confirm=0))

    # detect() with an empty override — use the detector's OWN config so the
    # constructor-time window override is in effect (not get_default_config).
    events_long = long_det.detect(df, {})
    events_short = short_det.detect(df, {})

    # with a huge window the planted pattern's confirm is inside → recovered
    assert events_long, f"{pattern}: long window must recover the planted H&S"

    # with a zero-bar window no confirm can fire after the pivot-known bar, so
    # every candidate is stale and the detector emits nothing.
    assert not events_short, (
        f"{pattern}: staleness window (0 bars) must discard late confirms, "
        f"got {len(events_short)} events"
    )

    # Monotonicity: shrinking the window never increases the event count.
    assert len(events_short) <= len(events_long)


def test_staleness_window_config_driven_and_attributes_carried() -> None:
    """§3.3 config key exists, defaults to 60 on M15, and stale candidates
    are tagged discard_reason by the detector/engine contract."""
    cfg = HeadShouldersDetector().get_default_config()
    assert cfg["max_bars_between_detect_and_confirm"] == 60
    df, _gt = _staleness_frame("head_shoulders")
    det = HeadShouldersDetector()
    for ev in det.detect(df, det.get_default_config()):
        # all emitted events are fresh (not stale); the engine tags any
        # discarded one with discard_reason (Event Lake, §3.3).
        assert ev.attributes.get("discard_reason") is None


# ---------------------------------------------------------------------------
# §8.3 Synthesizer acceptance benchmark
# ---------------------------------------------------------------------------

def test_synthesizer_acceptance_head_shoulders() -> None:
    result = run_hs_benchmark(
        "head_shoulders", n_series=150, noise_sigma=0.05, n_noise_series=300, seed=42
    )
    for r in result.per_pattern:
        assert r.recall >= 0.80, f"head_shoulders recall {r.recall:.2f} < 0.80"
        assert r.robustness_drop <= 0.15
    assert result.fp_rate() <= 0.05, f"FP {result.fp_rate():.3f} > 0.05"


def test_synthesizer_acceptance_inverse_head_shoulders() -> None:
    result = run_hs_benchmark(
        "inverse_head_shoulders",
        n_series=150,
        noise_sigma=0.05,
        n_noise_series=300,
        seed=42,
    )
    for r in result.per_pattern:
        assert r.recall >= 0.80
        assert r.robustness_drop <= 0.15
    assert result.fp_rate() <= 0.05, f"FP {result.fp_rate():.3f} > 0.05"


def test_detector_finds_planted_hs_in_position() -> None:
    df, gt = PatternSynthesizer().generate(
        "head_shoulders", 120, noise_sigma=0.02, seed=5
    )
    det = HeadShouldersDetector()
    events = det.detect(df, det.get_default_config())
    assert events
    loc = df.index.get_loc(events[0].confirm_time)
    bar = int(loc) if isinstance(loc, int) else int(np.asarray(loc).item())
    assert abs(bar - gt.breakout_bar) <= 2
    assert events[0].direction == "bearish"


# ---------------------------------------------------------------------------
# Zero CausalityViolation / config / order short-name
# ---------------------------------------------------------------------------

def test_zero_causality_violation_full_xauusd() -> None:
    df = load_xauusd_m15()
    for detector in (HeadShouldersDetector, InverseHeadShouldersDetector):
        det = detector()
        events = det.detect(df, det.get_default_config())
        assert events
        validate_causality(events, feature_schema=det.feature_schema)


def test_config_hash_stable_and_versioned() -> None:
    from research.core.config_hash import compute_config_hash

    for detector in (HeadShouldersDetector, InverseHeadShouldersDetector):
        a = compute_config_hash(detector().get_default_config())
        b = compute_config_hash(detector().get_default_config())
        assert a == b and len(a) == 12
        assert detector().get_default_config()["version"] == "1.0"


def test_order_comment_short_names() -> None:
    assert HeadShouldersDetector().short_name == "HS"
    assert InverseHeadShouldersDetector().short_name == "IHS"


# ---------------------------------------------------------------------------
# F1 (handoff §3 P1#5) — confirm_range_atr emitted at its declared
# available_at (confirm) with the gate-scorer rng_det convention; the H&S
# schema no longer carries a dead declaration (inverse shares the base).
# ---------------------------------------------------------------------------

def test_confirm_range_atr_emitted_matches_formula() -> None:
    """Every H&S event carries confirm_range_atr == (high-low)/ATR at the
    confirm bar (F1 resolution: populate, keep schema/causality clean)."""
    df = load_xauusd_m15()
    for detector in (HeadShouldersDetector, InverseHeadShouldersDetector):
        det = detector()
        events = det.detect(df, det.get_default_config())
        assert events, f"{detector.__name__} must emit events on full dataset"
        for ev in events:
            assert "confirm_range_atr" in ev.attributes, (
                f"{ev.event_id}: confirm_range_atr missing from attributes (F1 dead field)"
            )
            cb = int(ev.attributes["confirm_bar"])
            atr = atr_series(df, 14)
            atr_c = float(atr[cb]) if not np.isnan(atr[cb]) else float("nan")
            expected = (
                (float(df["high"].iloc[cb]) - float(df["low"].iloc[cb])) / atr_c
                if atr_c == atr_c and atr_c > 0.0
                else float("nan")
            )
            got = float(ev.attributes["confirm_range_atr"])
            if expected == expected:
                assert np.isclose(got, expected, atol=1e-9), (
                    f"{ev.event_id}: confirm_range_atr {got} != formula {expected}"
                )
        validate_causality(events, feature_schema=det.feature_schema)


# ---------------------------------------------------------------------------
# F2 (handoff §3 P1#6) — staleness effective bound pinned for both H&S
# mirrors: confirm scan anchored at the last extreme (right shoulder) ⇒
# detect->confirm ≤ max_wait - right_bars (stricter than the config key name;
# documented in findings_resolution.md).  H&S is the long-staleness pattern
# (§3.3), so the bound must be verified explicitly.
# ---------------------------------------------------------------------------

def test_staleness_effective_bound_detect_to_confirm() -> None:
    """confirm_bar - detect_bar <= max_wait - right_bars for HS + IHS."""
    for detector in (HeadShouldersDetector, InverseHeadShouldersDetector):
        det = detector()
        cfg = det.get_default_config()
        max_wait = int(cfg["max_bars_between_detect_and_confirm"])
        right = int(cfg["right_bars"])
        bound = max_wait - right
        checked = 0
        for doc, _gt in _planted_frames(
            "head_shoulders" if detector is HeadShouldersDetector else "inverse_head_shoulders"
        ):
            for ev in det.detect(doc, cfg):
                det_bar = int(ev.attributes["pivot_known_at_bar"])
                cb = int(ev.attributes["confirm_bar"])
                assert cb - det_bar <= bound, (
                    f"{ev.event_id}: confirm-detect {cb - det_bar} > {bound}"
                )
                checked += 1
        assert checked > 0, f"{detector.__name__}: no events to verify staleness bound"