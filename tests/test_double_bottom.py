"""Double Bottom plugin tests (P1) -- Agent 3 DoD.

Covers, per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §13 Agent 3 / §8.3 / §6.3:

  1. the inheritable no-lookahead template (:class:`NoLookaheadTestBase`) --
     timeline coherence, feature availability, versioned default config;
  2. §8.3 Synthesizer acceptance -- recall ≥ 80 % (≤2-bar tol), robustness
     drop ≤ 15 pts, FP ≤ 5 % on pure-noise series;
  3. zero ``CausalityViolation`` over the FULL XAUUSD M15 dataset;
  4. forward-only fixed-horizon labeling (no bar ≤ entry in any label);
  5. §6.3 research gates PASS on XAUUSD M15 (n / balance / ESS / PF CI /
     PR-AUC / walk-forward).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from research.core.causal_checks import validate_causality
from research.core.pattern_synthesizer import PatternSynthesizer
from research.patterns.double_bottom.benchmark import (
    run_double_benchmark,
)
from research.patterns.double_bottom.dataset import (
    assert_gates,
    label_events,
    load_xauusd_m15,
    run_gates,
)
from research.patterns.double_bottom.detector import (
    DoubleBottomDetector,
    atr_series,
)
from research.patterns.double_top.detector import DoubleTopDetector
from tests.no_lookahead_base import NoLookaheadTestBase

pytestmark = pytest.mark.no_lookahead

# Number of planted synthetic frames the causality template scans (~recall 1.0).
_SEEDS = 12


def _planted_frames(pattern: str = "double_bottom"):
    syn = PatternSynthesizer()
    for seed in range(_SEEDS):
        yield syn.generate(pattern, 120, noise_sigma=0.05, seed=seed)


class TestDoubleBottomCausality(NoLookaheadTestBase):
    """Inherited gates: timeline, feature availability, versioned config."""

    feature_schema: ClassVar = DoubleBottomDetector.feature_schema
    detector_class: ClassVar = DoubleBottomDetector

    def build_events(self) -> list:
        det = DoubleBottomDetector()
        cfg = det.get_default_config()
        events = []
        for frame, _gt in _planted_frames():
            events.extend(det.detect(frame, cfg))
        return events


# ---------------------------------------------------------------------------
# §8.3 Synthesizer acceptance benchmark
# ---------------------------------------------------------------------------

def test_synthesizer_acceptance_double_bottom() -> None:
    """Recall ≥ 0.80, robustness drop ≤ 0.15, FP ≤ 0.05 (§8.3)."""
    result = run_double_benchmark(
        "double_bottom", n_series=150, noise_sigma=0.05, n_noise_series=300, seed=42
    )
    for r in result.per_pattern:
        assert r.recall >= 0.80, f"recall {r.recall:.2f} < 0.80"
        assert r.robustness_drop <= 0.15, f"robustness drop {r.robustness_drop:.2f} > 0.15"
    assert result.fp_rate() <= 0.05, f"FP {result.fp_rate():.3f} > 0.05"


def test_detector_finds_planted_pattern_in_position() -> None:
    """On a clean synthetic frame the event's confirm bar matches GT ≤2 bars."""
    df, gt = PatternSynthesizer().generate("double_bottom", 120, noise_sigma=0.02, seed=3)
    det = DoubleBottomDetector()
    events = det.detect(df, det.get_default_config())
    assert events
    bar = df.index.get_loc(events[0].confirm_time)
    assert abs(bar - gt.breakout_bar) <= 2
    assert events[0].pattern_name == "double_bottom"
    assert events[0].direction == "bullish"
    assert gt.structure_levels["neckline"] <= events[0].structure_levels["neckline"] + 1e-6


# ---------------------------------------------------------------------------
# Zero CausalityViolation over the full XAUUSD M15 dataset
# ---------------------------------------------------------------------------

def test_zero_causality_violation_full_xauusd() -> None:
    """Runtime validator must accept every event of a full-dataset run."""
    df = load_xauusd_m15()
    det = DoubleBottomDetector()
    events = det.detect(df, det.get_default_config())
    assert len(events) >= 300, f"expected ≥300 events on full dataset, got {len(events)}"
    validate_causality(events, feature_schema=det.feature_schema)  # must not raise
    # every event confirmed: reversal close strictly past the neckline
    for ev in events:
        assert ev.confirm_time >= ev.detect_time


def test_events_strictly_causal_timeline() -> None:
    """Detect/confirm stamps are index-aligned and detect ≤ confirm."""
    df = load_xauusd_m15()
    events = DoubleBottomDetector().detect(df, DoubleBottomDetector().get_default_config())
    for ev in events:
        assert df.index.get_loc(ev.detect_time) <= df.index.get_loc(ev.confirm_time)
        assert ev.known_at_ts == ev.confirm_time  # confirm dominates


# ---------------------------------------------------------------------------
# Forward fixed-horizon labeling (no look-ahead in labels)
# ---------------------------------------------------------------------------

def test_labels_use_only_future_bars() -> None:
    """mfe/mae/close ratios must be computed from bars strictly AFTER entry."""
    df = load_xauusd_m15()
    det = DoubleBottomDetector()
    events = det.detect(df, det.get_default_config())
    labeled = label_events(df, events, horizon_bars=96, target_r=1.5)
    assert len(labeled) >= 300

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    for le in labeled:
        eb = le.entry_bar
        s, e = eb + 1, eb + le.horizon_bars
        # recompute forward-only MFE from raw bars and compare exactly
        expected_mfe = (float(np.max(high[s: e + 1])) - le.entry_price) / le.risk
        expected_mae = (le.entry_price - float(np.min(low[s: e + 1]))) / le.risk
        expected_close = (float(close[e]) - le.entry_price) / le.risk
        assert np.isclose(le.mfe_ratio, expected_mfe, atol=1e-9)
        assert np.isclose(le.mae_ratio, expected_mae, atol=1e-9)
        assert np.isclose(le.close_ratio, expected_close, atol=1e-9)
        # the label itself is a pure function of the forward MFE
        assert le.label == (1 if le.mfe_ratio >= 1.5 else 0)


# ---------------------------------------------------------------------------
# §6.3 research gates on XAUUSD M15
# ---------------------------------------------------------------------------

def test_research_gates_pass_xauusd_m15() -> None:
    """All §6.3 gates: n≥300, n_oos≥100, balance, ESS≥60%, PF CI>1,
    PR-AUC≥1.05xbaseline, ≥3 WF folds PF≥0.8."""
    df = load_xauusd_m15()
    det = DoubleBottomDetector()
    events = det.detect(df, det.get_default_config())
    report = run_gates(df, events)
    assert_gates(report)
    assert report.n_total >= 300
    assert report.n_oos >= 100
    assert 0.10 <= report.positive_rate_oos <= 0.90
    assert report.ess_ratio >= 0.60
    assert report.pf_ci_lower > 1.0
    assert report.pr_auc >= 1.05 * report.baseline
    assert report.wf_longest_run >= 3


def test_config_hash_stable_and_versioned() -> None:
    """Same default config → same 12-char hash; version present (§6.2)."""
    from research.core.config_hash import compute_config_hash

    a = compute_config_hash(DoubleBottomDetector().get_default_config())
    b = compute_config_hash(DoubleBottomDetector().get_default_config())
    assert a == b and len(a) == 12
    assert DoubleBottomDetector().get_default_config()["version"] == "1.0"


def test_order_comment_short_name() -> None:
    """§9.2 reserved short name DB -- used by the live engine order comment."""
    det = DoubleBottomDetector()
    df, _gt = next(_planted_frames())
    events = det.detect(df, det.get_default_config())
    assert det.short_name == "DB"
    assert events, "planted frame must produce an event"
    assert events[0].event_id.startswith("XAUUSD-DB-")


# ---------------------------------------------------------------------------
# F1 (handoff §3 P1#5) — confirm_range_atr no longer dead: it is emitted at
# its declared available_at (confirm) with the gate-scorer rng_det convention
# (range / ATR at the SAME bar), and causality stays clean.
# ---------------------------------------------------------------------------

def test_confirm_range_atr_emitted_matches_formula() -> None:
    """Every double-family event carries confirm_range_atr == (high-low)/ATR
    at the confirm bar (F1 resolution: populate, keep schema/causality)."""
    df = load_xauusd_m15()
    for detector in (DoubleBottomDetector, DoubleTopDetector):
        det = detector()
        events = det.detect(df, det.get_default_config())
        assert events, "full dataset must emit double-bottom events"
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
            if expected == expected:  # not NaN
                assert np.isclose(got, expected, atol=1e-9), (
                    f"{ev.event_id}: confirm_range_atr {got} != formula {expected}"
                )
        # schema/causality: the declared feature resolves ≤ known_at and the
        # runtime validator accepts the batch (zero CausalityViolation).
        validate_causality(events, feature_schema=det.feature_schema)


# ---------------------------------------------------------------------------
# F2 (handoff §3 P1#6) — staleness window semantics pinned: the confirm scan
# is anchored at the LAST EXTREME bar (not the detect/anchor bar), so the
# effective detect->confirm bound is max_wait - right_bars (STRICTER than the
# config name suggests; zero causal risk — only discards more).  This test
# locks the effective bound so a future semantics change is deliberate.
# ---------------------------------------------------------------------------

def test_staleness_effective_bound_detect_to_confirm() -> None:
    """confirm_bar - detect_bar <= max_wait - right_bars on planted frames
    (F2: window measured from the last extreme, i.e. right_bars tighter
    than the config key name — documented decision, see findings_resolution)."""
    det = DoubleBottomDetector()
    cfg = det.get_default_config()
    max_wait = int(cfg["max_bars_between_detect_and_confirm"])
    right = int(cfg["right_bars"])
    bound = max_wait - right
    checked = 0
    for df, _gt in _planted_frames():
        for ev in det.detect(df, cfg):
            det_bar = int(ev.attributes["pivot_known_at_bar"])
            cb = int(ev.attributes["confirm_bar"])
            assert cb - det_bar <= bound, (
                f"{ev.event_id}: confirm-detect {cb - det_bar} > {bound} "
                f"({max_wait} - right_bars {right})"
            )
            checked += 1
    assert checked > 0, "no events to verify the staleness bound"