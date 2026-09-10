"""Double Top plugin tests (P2, mirror of P1) -- Agent 3 DoD.

Double Top reuses 100 % of P1 infra (spec §11); the tests verify the mirror:

  1. the inheritable no-lookahead template + versioned config;
  2. §8.3 Synthesizer acceptance for the bearish mirror;
  3. zero ``CausalityViolation`` over the FULL XAUUSD M15 dataset;
  4. mirror semantics: H-L-H swings, bearish direction, neckline = middle
     low, confirmation = close strictly BELOW the neckline;
  5. §6.3 research gates PASS on XAUUSD M15.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from research.core.causal_checks import validate_causality
from research.core.pattern_synthesizer import PatternSynthesizer
from research.patterns.double_bottom.benchmark import (
    PluginDetectorAdapter,
    run_double_benchmark,
)
from research.patterns.double_bottom.dataset import (
    assert_gates,
    label_events,
    load_xauusd_m15,
    run_gates,
)
from research.patterns.double_top.detector import DoubleTopDetector
from tests.no_lookahead_base import NoLookaheadTestBase

pytestmark = pytest.mark.no_lookahead

_SEEDS = 12


class TestDoubleTopCausality(NoLookaheadTestBase):
    """Inherited gates: timeline, feature availability, versioned config."""

    feature_schema: ClassVar = DoubleTopDetector.feature_schema
    detector_class: ClassVar = DoubleTopDetector

    def build_events(self) -> list:
        det = DoubleTopDetector()
        cfg = det.get_default_config()
        events = []
        for seed in range(_SEEDS):
            df, _gt = PatternSynthesizer().generate("double_top", 120, noise_sigma=0.05, seed=seed)
            events.extend(det.detect(df, cfg))
        return events


# ---------------------------------------------------------------------------
# §8.3 Synthesizer acceptance benchmark (mirror)
# ---------------------------------------------------------------------------

def test_synthesizer_acceptance_double_top() -> None:
    result = run_double_benchmark(
        "double_top", n_series=150, noise_sigma=0.05, n_noise_series=300, seed=42
    )
    for r in result.per_pattern:
        assert r.recall >= 0.80, f"recall {r.recall:.2f} < 0.80"
        assert r.robustness_drop <= 0.15, f"robustness drop {r.robustness_drop:.2f} > 0.15"
    assert result.fp_rate() <= 0.05, f"FP {result.fp_rate():.3f} > 0.05"


# ---------------------------------------------------------------------------
# Mirror semantics
# ---------------------------------------------------------------------------

def test_mirror_geometry_bearish() -> None:
    """Double Top = H-L-H: two equal highs, middle low neckline, close below."""
    df, gt = PatternSynthesizer().generate("double_top", 120, noise_sigma=0.02, seed=4)
    det = DoubleTopDetector()
    events = det.detect(df, det.get_default_config())
    assert events
    ev = events[0]
    assert ev.pattern_name == "double_top"
    assert ev.direction == "bearish"
    bar = df.index.get_loc(ev.confirm_time)
    assert abs(bar - gt.breakout_bar) <= 2
    # the confirm close is strictly BELOW the neckline (mirror of bottom)
    assert df["close"].iloc[bar] < ev.structure_levels["neckline"]
    # stop sits ABOVE the higher extreme
    assert ev.stop_price > gt.structure_levels["double_top_level"]
    assert ev.target_price < ev.entry_price


def test_mirror_shared_infra_identical_config() -> None:
    """P2 shares P1's config schema + feature schema verbatim."""
    from research.patterns.double_bottom.detector import (
        FEATURE_SCHEMA_VERSION,
        DoublePatternDetectorBase,
    )

    b = DoublePatternDetectorBase()
    t = DoubleTopDetector()
    assert t.get_default_config()["version"] == b.get_default_config()["version"]
    assert [f.name for f in t.feature_schema] == [f.name for f in b.feature_schema]
    assert t.version == "1.0" and t.short_name == "DT"
    assert FEATURE_SCHEMA_VERSION.startswith("double-v")


# ---------------------------------------------------------------------------
# Zero CausalityViolation over the full XAUUSD M15 dataset
# ---------------------------------------------------------------------------

def test_zero_causality_violation_full_xauusd() -> None:
    df = load_xauusd_m15()
    det = DoubleTopDetector()
    events = det.detect(df, det.get_default_config())
    assert len(events) >= 300, f"expected ≥300 events on full dataset, got {len(events)}"
    validate_causality(events, feature_schema=det.feature_schema)  # must not raise


def test_labels_use_only_future_bars_mirror() -> None:
    """Shorts: mfe = entry - min(low) over (entry, H] -- strictly forward."""
    df = load_xauusd_m15()
    det = DoubleTopDetector()
    events = det.detect(df, det.get_default_config())
    labeled = label_events(df, events, horizon_bars=96, target_r=1.5)
    assert len(labeled) >= 300

    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    for le in labeled:
        eb = le.entry_bar
        s, e = eb + 1, eb + le.horizon_bars
        expected_mfe = (le.entry_price - float(np.min(low[s: e + 1]))) / le.risk
        expected_mae = (float(np.max(high[s: e + 1])) - le.entry_price) / le.risk
        expected_close = (le.entry_price - float(close[e])) / le.risk
        assert np.isclose(le.mfe_ratio, expected_mfe, atol=1e-9)
        assert np.isclose(le.mae_ratio, expected_mae, atol=1e-9)
        assert np.isclose(le.close_ratio, expected_close, atol=1e-9)


# ---------------------------------------------------------------------------
# §6.3 research gates on XAUUSD M15 (mirror)
# ---------------------------------------------------------------------------

def test_research_gates_pass_xauusd_m15() -> None:
    df = load_xauusd_m15()
    det = DoubleTopDetector()
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


def test_benchmark_adapter_maps_confirm_bar() -> None:
    """The plugin→benchmark adapter uses the honest confirm bar, not GT."""
    df, _gt = PatternSynthesizer().generate("double_top", 120, noise_sigma=0.05, seed=7)
    det = DoubleTopDetector()
    dets = PluginDetectorAdapter(det).detect(df, "double_top")
    assert dets
    for d in dets:
        assert d.pattern_name == "double_top"
        assert d.direction == "bearish"
        assert 0 <= d.breakout_bar < len(df)