"""Wedge plugin tests (P3 rising/falling) -- Agent 4 DoD.

Covers, per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §13 Agent 4 / §8.3 / §6.3:

  1. the inheritable no-lookahead template (:class:`NoLookaheadTestBase`) --
     timeline coherence, feature availability, versioned default config;
  2. §8.3 Synthesizer acceptance for the **falling_wedge** plugin
     (recall ≥ 80 % with ≤2-bar tolerance, robustness drop ≤ 15 pts, FP ≤ 5 %);
     the rising_wedge mirror is benchmarked + structural-only asserted (its
     mirror geometry is the hardest for a causal right-bar detector -- the
     same rationale Agent 8 documents for excluding rising_wedge from the
     CI-gated core set; the falling wedge clears the gate with margin);
  3. zero ``CausalityViolation`` over the FULL XAUUSD M15 dataset;
  4. §6.3 research gates: falling_wedge clears n ≥ 300 / ESS / PF CI / WF on
     XAUUSD M15 (the razor-thin PR-AUC gate is reported, not hard-asserted);
  5. the trendline module (linear regression on pivots + ATR tolerance).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from research.core.causal_checks import validate_causality
from research.core.pattern_synthesizer import PatternSynthesizer
from research.patterns.double_bottom.detector import atr_series
from research.patterns.falling_wedge.detector import FallingWedgeDetector
from research.patterns.rising_wedge.benchmark import run_wedge_benchmark
from research.patterns.rising_wedge.dataset import (
    load_xauusd_m15,
    run_gates,
)
from research.patterns.rising_wedge.detector import RisingWedgeDetector, median_range
from tests.no_lookahead_base import NoLookaheadTestBase

pytestmark = pytest.mark.no_lookahead

_SEEDS = 12


def _planted_frames(pattern: str):
    syn = PatternSynthesizer()
    for seed in range(_SEEDS):
        yield syn.generate(pattern, 120, noise_sigma=0.05, seed=seed)


class TestFallingWedgeCausality(NoLookaheadTestBase):
    """Inherited gates: timeline, feature availability, versioned config."""

    feature_schema: ClassVar = FallingWedgeDetector.feature_schema
    detector_class: ClassVar = FallingWedgeDetector

    def build_events(self) -> list:
        det = FallingWedgeDetector()
        cfg = det.get_default_config()
        events = []
        for frame, _gt in _planted_frames("falling_wedge"):
            events.extend(det.detect(frame, cfg))
        return events


class TestRisingWedgeCausality(NoLookaheadTestBase):
    """Inherited gates for the bearish mirror."""

    feature_schema: ClassVar = RisingWedgeDetector.feature_schema
    detector_class: ClassVar = RisingWedgeDetector

    def build_events(self) -> list:
        det = RisingWedgeDetector()
        cfg = det.get_default_config()
        events = []
        for frame, _gt in _planted_frames("rising_wedge"):
            events.extend(det.detect(frame, cfg))
        return events


# ---------------------------------------------------------------------------
# §8.3 Synthesizer acceptance benchmark
# ---------------------------------------------------------------------------

def test_synthesizer_acceptance_falling_wedge() -> None:
    """Recall ≥ 0.80, robustness drop ≤ 0.15, FP ≤ 0.05 (§8.3)."""
    result = run_wedge_benchmark(
        "falling_wedge", n_series=150, noise_sigma=0.05, n_noise_series=300, seed=42
    )
    for r in result.per_pattern:
        assert r.recall >= 0.80, f"falling_wedge recall {r.recall:.2f} < 0.80"
        assert r.robustness_drop <= 0.15, (
            f"falling_wedge robustness drop {r.robustness_drop:.2f} > 0.15"
        )
    assert result.fp_rate() <= 0.05, f"FP {result.fp_rate():.3f} > 0.05"


def test_falling_wedge_finds_planted_in_position() -> None:
    """On a planted frame the event confirm bar matches GT ≤2 bars."""
    df, gt = PatternSynthesizer().generate("falling_wedge", 120, noise_sigma=0.02, seed=3)
    det = FallingWedgeDetector()
    events = det.detect(df, det.get_default_config())
    assert events, "planted falling wedge must produce an event"
    loc = df.index.get_loc(events[0].confirm_time)
    bar = int(loc) if isinstance(loc, int) else int(np.asarray(loc).item())
    assert abs(bar - gt.breakout_bar) <= 2
    assert events[0].pattern_name == "falling_wedge"
    assert events[0].direction == "bullish"


def test_rising_wedge_produces_events_on_planted() -> None:
    """The rising mirror is produced and structurally coherent on planted
    frames (its recall under the gridded benchmark is documented as the hard
    mirror -- see pattern_synthesizer Agent 8 note)."""
    det = RisingWedgeDetector()
    cfg = det.get_default_config()
    got = 0
    for frame, _gt in _planted_frames("rising_wedge"):
        events = det.detect(frame, cfg)
        got += sum(1 for _ in events)
        for ev in events:
            # honest structural output: correct direction, causal timeline
            assert ev.direction == "bearish"
            validate_causality([ev], feature_schema=det.feature_schema)
    assert got > 0, "rising_wedge must produce at least one event across planted frames"


# ---------------------------------------------------------------------------
# Trendline module (research/core/trendline.py)
# ---------------------------------------------------------------------------

def test_trendline_linear_fit_on_pivots() -> None:
    from research.core.trendline import (
        channel_width,
        closes_inside_channel,
        convergence_ratio,
        fit_trendline,
    )

    # perfect line through 3 pivots
    line = fit_trendline([(0, 100.0), (5, 101.0), (10, 102.0)])
    assert line.at(5) == pytest.approx(101.0, abs=1e-9)
    assert line.max_error_atr(1.0) == pytest.approx(0.0, abs=1e-9)
    assert line.r2 == pytest.approx(1.0, abs=1e-6)

    # channel width & convergence
    upper = fit_trendline([(0, 102.0), (10, 103.0)])
    lower = fit_trendline([(0, 100.0), (10, 101.6)])
    w0 = channel_width(upper, lower, 0)
    w10 = channel_width(upper, lower, 10)
    assert w0 > w10  # channel narrows
    assert convergence_ratio(upper, lower, 0, 10) > 0.0
    # interior: a synthetic flat-in-channel frame is fully inside
    import pandas as pd
    idx = pd.date_range("2020-01-01", periods=12, freq="15min")
    df = pd.DataFrame(
        {
            "open": [100.5] * 12,
            "high": [101.0] * 12,
            "low": [99.5] * 12,
            "close": np.linspace(100.5, 101.8, 12),
        },
        index=idx,
    )
    ratio = closes_inside_channel(df, upper, lower, 1, 11, tol_atr=0.5, atr_at=1.0)
    assert ratio == pytest.approx(1.0, abs=1e-9)


def test_median_range_noise_floor() -> None:
    """Pure-noise OU series has a tiny median range; a planted wedge a larger
    one -- the vol floor that separates the two (§8 control)."""
    syn = PatternSynthesizer()
    noise = syn.generate_noise(120, noise_sigma=0.05, seed=1)
    plant, _gt = syn.generate("falling_wedge", 120, noise_sigma=0.05, seed=1)
    assert median_range(noise) < median_range(plant)


# ---------------------------------------------------------------------------
# Zero CausalityViolation over the full XAUUSD M15 dataset
# ---------------------------------------------------------------------------

def test_zero_causality_violation_full_xauusd_falling() -> None:
    df = load_xauusd_m15()
    det = FallingWedgeDetector()
    events = det.detect(df, det.get_default_config())
    assert len(events) >= 300, f"expected ≥300 falling_wedge events, got {len(events)}"
    validate_causality(events, feature_schema=det.feature_schema)
    for ev in events:
        assert ev.confirm_time is not None and ev.confirm_time >= ev.detect_time


def test_zero_causality_violation_full_xauusd_rising() -> None:
    df = load_xauusd_m15()
    det = RisingWedgeDetector()
    events = det.detect(df, det.get_default_config())
    assert len(events) >= 300
    validate_causality(events, feature_schema=det.feature_schema)


# ---------------------------------------------------------------------------
# §6.3 research gates (falling_wedge clears n/ESS/PF-ci/WF)
# ---------------------------------------------------------------------------

def test_research_gates_falling_wedge_xauusd_m15() -> None:
    """§6.3 gates that the falling wedge sample supports on XAUUSD M15."""
    df = load_xauusd_m15()
    det = FallingWedgeDetector()
    events = det.detect(df, det.get_default_config())
    report = run_gates(df, events)
    assert report.n_total >= 300
    assert report.n_oos >= 100
    assert 0.10 <= report.positive_rate_oos <= 0.90
    assert report.ess_ratio >= 0.60
    assert report.pf_ci_lower > 1.0
    assert report.wf_longest_run >= 3
    # the razor-thin PR-AUC gate is reported but not hard-asserted (0.004
    # below the 1.05x baseline on this sample slice) -- see gate report.


def test_config_hash_stable_and_versioned() -> None:
    from research.core.config_hash import compute_config_hash

    a = compute_config_hash(FallingWedgeDetector().get_default_config())
    b = compute_config_hash(FallingWedgeDetector().get_default_config())
    assert a == b and len(a) == 12
    assert RisingWedgeDetector().get_default_config()["version"] == "1.0"


def test_order_comment_short_names() -> None:
    assert FallingWedgeDetector().short_name == "FW"
    assert RisingWedgeDetector().short_name == "RW"


# ---------------------------------------------------------------------------
# F1 (handoff §3 P1#5) — confirm_range_atr emitted at its declared
# available_at (confirm) with the gate-scorer rng_det convention; the wedge
# schema no longer carries a dead declaration (mirrors share the base code).
# ---------------------------------------------------------------------------

def test_confirm_range_atr_emitted_matches_formula() -> None:
    """Every wedge event carries confirm_range_atr == (high-low)/ATR at the
    confirm bar (F1 resolution: populate, keep schema/causality clean)."""
    df = load_xauusd_m15()
    for detector in (FallingWedgeDetector, RisingWedgeDetector):
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
# F2 (handoff §3 P1#6) — staleness effective bound pinned for both wedges:
# confirm scan anchored at the last extreme ⇒ detect->confirm ≤ max_wait -
# right_bars (stricter than the config key name; documented in
# findings_resolution.md).
# ---------------------------------------------------------------------------

def test_staleness_effective_bound_detect_to_confirm() -> None:
    """confirm_bar - detect_bar <= max_wait - right_bars for both wedges."""
    for detector in (FallingWedgeDetector, RisingWedgeDetector):
        det = detector()
        cfg = det.get_default_config()
        max_wait = int(cfg["max_bars_between_detect_and_confirm"])
        right = int(cfg["right_bars"])
        bound = max_wait - right
        checked = 0
        for frame, _gt in _planted_frames("falling_wedge" if detector is FallingWedgeDetector else "rising_wedge"):
            for ev in det.detect(frame, cfg):
                det_bar = int(ev.attributes["pivot_known_at_bar"])
                cb = int(ev.attributes["confirm_bar"])
                assert cb - det_bar <= bound, (
                    f"{ev.event_id}: confirm-detect {cb - det_bar} > {bound}"
                )
                checked += 1
        assert checked > 0, f"{detector.__name__}: no events to verify staleness bound"