"""test_pattern_rework.py — tests for the pattern-rework spec proposals.

Covers, against the real XAUUSD M15 history where the property is
data-dependent, and with synthetic frames where it is structural:

  §1.3.1  NMS over the structure window          -> test_nms_*
  §1.3.3  opposite-direction overlap guard       -> test_drop_opposite_*
  §2.3.1  structure-based stop                   -> test_stop_mode_*
  §2.3.2  target cap + min_rr gate               -> test_target_cap_*

The rework spec's §2.3 proposals are implemented as OPT-IN options whose
defaults preserve the measured-better legacy behaviour; these tests pin BOTH
sides so a future default flip is a deliberate, test-visible act.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.core.contracts import (
    DIRECTION_BEARISH,
    DIRECTION_BULLISH,
    PatternEvent,
)
from research.core.dedupe import drop_opposite_overlap, structure_span
from research.patterns.double_bottom.detector import DoubleBottomDetector
from research.patterns.double_top.detector import DoubleTopDetector

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def _event(
    *,
    pattern: str,
    direction: str,
    e1: int,
    e2: int,
    neck: int,
    score: float,
    known: str = "2024-01-01T00:00:00+00:00",
    attributes: dict | None = None,
) -> PatternEvent:
    attrs = {"extreme1_bar": e1, "extreme2_bar": e2, "neckline_bar": neck}
    if attributes:
        attrs.update(attributes)
    return PatternEvent(
        event_id=f"{pattern}-{e1}-{e2}",
        pattern_name=pattern,
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        detect_time=pd.Timestamp("2024-01-01T00:00:00+00:00"),
        confirm_time=pd.Timestamp(known),
        entry_time=pd.Timestamp(known),
        entry_price=2000.0,
        stop_price=1990.0,
        target_price=2015.0,
        structure_levels={"neckline": 2005.0},
        rule_score=score,
        model_prob=None,
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# §1.3.1 — NMS over the structure window
# ---------------------------------------------------------------------------
class TestNMS:
    def test_nms_removes_intersecting_windows_keeps_best(self):
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=120, neck=110, score=0.40)
        b = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=110, e2=130, neck=120, score=0.90)
        kept = DoubleTopDetector._nms_structure_overlap([a, b])
        assert len(kept) == 1
        assert kept[0].rule_score == pytest.approx(0.90)

    def test_nms_keeps_disjoint_windows(self):
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=110, neck=105, score=0.40)
        b = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=200, e2=210, neck=205, score=0.90)
        assert len(DoubleTopDetector._nms_structure_overlap([a, b])) == 2

    def test_nms_touching_windows_conflict(self):
        """A shared bar is a shared structure (b==a+1 encodes it)."""
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=120, neck=110, score=0.40)
        b = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=120, e2=140, neck=130, score=0.90)
        assert len(DoubleTopDetector._nms_structure_overlap([a, b])) == 1

    def test_nms_is_deterministic_on_score_tie(self):
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=120, neck=110, score=0.50)
        b = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=110, e2=130, neck=120, score=0.50)
        first = DoubleTopDetector._nms_structure_overlap([a, b])
        second = DoubleTopDetector._nms_structure_overlap([b, a])
        assert [e.event_id for e in first] == [e.event_id for e in second]

    def test_nms_noop_below_two_events(self):
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=120, neck=110, score=0.5)
        assert DoubleTopDetector._nms_structure_overlap([]) == []
        assert len(DoubleTopDetector._nms_structure_overlap([a])) == 1

    def test_nms_toggle_changes_detector_output(self):
        """On the real series the scan emits a few duplicate structures."""
        df = _load_xauusd()
        on = DoubleTopDetector({"nms_overlap": True}).detect(df, {})
        off = DoubleTopDetector({"nms_overlap": False}).detect(df, {})
        assert len(on) <= len(off)
        # windows must be pairwise disjoint once NMS has run
        spans = sorted(structure_span(e) for e in on)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert start > end


# ---------------------------------------------------------------------------
# §1.3.3 — opposite-direction overlap guard
# ---------------------------------------------------------------------------
class TestOppositeOverlap:
    def test_drops_weaker_contrary_reading(self):
        db = _event(pattern="double_bottom", direction=DIRECTION_BULLISH,
                    e1=100, e2=130, neck=115, score=0.90)
        dt = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                    e1=110, e2=125, neck=120, score=0.40)
        kept = drop_opposite_overlap([db, dt])
        assert [e.pattern_name for e in kept] == ["double_bottom"]

    def test_keeps_contrary_events_on_disjoint_structure(self):
        db = _event(pattern="double_bottom", direction=DIRECTION_BULLISH,
                    e1=100, e2=130, neck=115, score=0.90)
        dt = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                    e1=500, e2=530, neck=515, score=0.40)
        assert len(drop_opposite_overlap([db, dt])) == 2

    def test_keeps_same_direction_events(self):
        a = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=100, e2=130, neck=115, score=0.90)
        b = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                   e1=110, e2=140, neck=125, score=0.40)
        assert len(drop_opposite_overlap([a, b])) == 2

    def test_intersecting_windows_without_shared_neckline_are_kept(self):
        """Overlapping boxes alone are not a conflict -- the pivot sequence
        must also be shared, otherwise two genuinely different structures
        that happen to sit close would wrongly cancel out."""
        db = _event(pattern="double_bottom", direction=DIRECTION_BULLISH,
                    e1=100, e2=130, neck=101, score=0.90)
        dt = _event(pattern="double_top", direction=DIRECTION_BEARISH,
                    e1=105, e2=135, neck=134, score=0.40)
        assert len(drop_opposite_overlap([db, dt])) == 2

    def test_real_case_user_reported_2026_07_06(self):
        """The chart the rework spec §0.1 #2 was built from: a double top and
        a double bottom read off one shared middle swing must not both live."""
        df = _load_xauusd()
        events = DoubleBottomDetector().detect(df, {}) + DoubleTopDetector().detect(df, {})
        window = [
            e for e in events
            if (s := structure_span(e)) is not None and 3700 <= s[0] <= 3740
        ]
        assert len(window) == 2, "expected the DT/DB pair from the report"
        kept = drop_opposite_overlap(window)
        assert len(kept) == 1
        assert kept[0].rule_score == max(e.rule_score for e in window)

    def test_guard_is_off_by_default_on_both_sides(self):
        """§12 parity: the guard must be absent from the live scan AND the
        backtest runner unless explicitly enabled, otherwise the same input
        yields different event sets and the parity test breaks."""
        import inspect

        from live.engine.signal_engine_v2 import MultiPatternEngine
        from research.multi_backtest import runner as mb

        live_sig = inspect.signature(MultiPatternEngine.__init__)
        assert live_sig.parameters["opposite_overlap_guard"].default is False

        bt_sig = inspect.signature(mb.run_symbol_backtest)
        assert bt_sig.parameters["opposite_overlap_guard"].default is False

    def test_live_and_backtest_drop_the_same_events_when_enabled(self):
        """Turning the guard on must remove an identical event set on both
        sides — that is what keeps backtest ≡ live once it is adopted."""

        from live.engine.signal_engine_v2 import MultiPatternEngine
        from research.multi_backtest import runner as mb

        df = _load_xauusd().iloc[:5000]
        assignments = mb.build_assignments(
            "XAUUSD", ["double_bottom", "double_top"], df=df, attach_models=False
        )

        runner_events = mb.detect_all(df, assignments)
        runner_kept = {e.event_id for e in drop_opposite_overlap(runner_events)}

        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=assignments,
            candle_fn=lambda _s: df,
            opposite_overlap_guard=True,
        )
        engine.check_new_bar()
        live_kept = {e.event_id for e in engine.get_last_events()}

        assert live_kept == runner_kept
        assert len(live_kept) < len(runner_events), "guard did not trigger"


# ---------------------------------------------------------------------------
# §2.3.1 — structure-based stop
# ---------------------------------------------------------------------------
class TestStructureStop:
    def test_legacy_stop_is_default(self):
        cfg = DoubleBottomDetector().get_default_config()
        assert cfg["stop_mode"] == "legacy"
        assert cfg["structure_target_atr"] == 0.0
        assert cfg["target_cap_atr"] == 0.0

    def test_neckline_stop_is_tighter_and_capped_by_legacy(self):
        df = _load_xauusd()
        legacy = DoubleBottomDetector({"stop_mode": "legacy"}).detect(df, {})
        neck = DoubleBottomDetector({"stop_mode": "neckline"}).detect(df, {})
        by_id = {e.event_id: e for e in legacy}
        assert neck, "neckline stop produced no events"
        for ev in neck:
            old = by_id.get(ev.event_id)
            if old is None:
                continue
            # bullish -> stop can only move UP (tighter), never down
            assert ev.stop_price >= old.stop_price - 1e-9
            assert ev.stop_price < ev.entry_price

    def test_neckline_stop_keeps_almost_every_candidate(self):
        """A tightening stop must not silently delete the signal set."""
        df = _load_xauusd()
        legacy = DoubleBottomDetector({"stop_mode": "legacy"}).detect(df, {})
        neck = DoubleBottomDetector({"stop_mode": "neckline"}).detect(df, {})
        assert len(neck) >= 0.95 * len(legacy)


# ---------------------------------------------------------------------------
# §2.3.2 — target cap and min_rr gate
# ---------------------------------------------------------------------------
class TestTargetCap:
    def test_cap_bounds_target_distance(self):
        df = _load_xauusd()
        evs = DoubleTopDetector(
            {"target_cap_atr": 4.0, "structure_target_atr": 0.0, "min_rr": 0.0}
        ).detect(df, {})
        assert evs
        for ev in evs:
            atr = ev.attributes.get("atr_value")
            if atr:
                assert abs(ev.target_price - ev.entry_price) <= 4.0 * atr * 1.01

    def test_min_rr_drops_sub_one_reward(self):
        df = _load_xauusd()
        ungated = DoubleTopDetector(
            {"structure_target_atr": 4.0, "target_cap_atr": 6.0, "min_rr": 0.0}
        ).detect(df, {})
        gated = DoubleTopDetector(
            {"structure_target_atr": 4.0, "target_cap_atr": 6.0, "min_rr": 1.0}
        ).detect(df, {})
        assert len(gated) < len(ungated)
        for ev in gated:
            assert ev.attributes["realized_rr"] >= 1.0

    def test_spec_combo_would_destroy_the_signal_set(self):
        """Regression guard for the rework spec as literally written.

        §2.3.2's 4 ATR cap is smaller than the stop width this family
        produces (median risk ~4.5 ATR), so combining it with §2.3.1's
        min_rr>=1.0 gate removes ~96% of candidates.  This test documents
        that arithmetic so nobody re-applies the pair by accident.
        """
        df = _load_xauusd()
        literal = DoubleTopDetector(
            {"target_cap_atr": 4.0, "min_rr": 1.0, "structure_target_atr": 4.0}
        ).detect(df, {})
        legacy = DoubleTopDetector().detect(df, {})
        assert len(literal) < 0.10 * len(legacy)

    def test_realized_rr_recorded(self):
        df = _load_xauusd()
        for ev in DoubleBottomDetector().detect(df, {}):
            assert "realized_rr" in ev.attributes
            assert "target_capped" in ev.attributes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_xauusd() -> pd.DataFrame:
    from research.multi_backtest.runner import load_symbol_frame

    return load_symbol_frame("XAUUSD", start="2018-06-01", end="2026-09-03")
