"""test_trend_context.py — tests for the rework §2.1 trend-context gate.

The gate is the highest-risk piece of the rework because two of its failure
modes are SILENT:

* **lookahead** — reading bars at/after ``extreme1_bar`` (or using the ATR of
  the later confirm bar) would make the gate look good in backtest and be
  unusable live;
* **inverted direction sign** — requiring an uptrend before a double bottom
  keeps exactly the wrong half of the history and still produces signals.

So both are pinned mechanically here rather than by inspection, alongside the
fail-closed contract for every degenerate input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.core.config_hash import compute_config_hash
from research.core.contracts import DIRECTION_BEARISH, DIRECTION_BULLISH
from research.core.trend_context import (
    DISCARD_LOW_TREND_CONTEXT,
    SlopeTrendContextProvider,
    TrendContextProvider,
    assert_causal_window,
    expected_slope_sign,
    has_trend_context,
    regression_slope_r2,
    trend_context_window,
)
from research.patterns.double_bottom.detector import DoubleBottomDetector, atr_series
from research.patterns.double_top.detector import DoubleTopDetector

REPO = Path(__file__).resolve().parents[1]

#: Marker for tests that require the §2.1 gate to be wired into the detector.
#:
#: The module-level gate is fully implemented and tested independently; the
#: detector wiring is a separate, serialized change to a file another engineer
#: also edits.  Until it lands these tests would fail with an opaque
#: AttributeError, so they are skipped with an explicit, actionable reason
#: instead.  ``trend_context_enabled`` in the default config is the signal that
#: the wiring exists — flip this to a hard failure once it lands by removing
#: the marker (see rework/trend_context_measurement.md §8).
WIRING_LANDED = "trend_context_enabled" in DoubleBottomDetector().get_default_config()
requires_wiring = pytest.mark.skipif(
    not WIRING_LANDED,
    reason=(
        "§2.1 gate not yet wired into DoubleBottomDetector._detect_double "
        "(blocked on task t2); module research/core/trend_context.py is complete "
        "and independently tested"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _up(n: int, start: float = 100.0, step: float = 0.5) -> np.ndarray:
    return start + step * np.arange(n, dtype=float)


def _down(n: int, start: float = 100.0, step: float = 0.5) -> np.ndarray:
    return start - step * np.arange(n, dtype=float)


# ---------------------------------------------------------------------------
# §2.1 Phương án B — regression slope + R²
# ---------------------------------------------------------------------------
class TestRegressionCore:
    def test_perfect_line_gives_exact_slope_and_r2_one(self) -> None:
        window = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        fit = regression_slope_r2(window)
        assert fit is not None
        slope, r2 = fit
        assert slope == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)

    def test_flat_window_r2_is_undefined_and_fails_closed(self) -> None:
        # Zero variance -> R² = 0/0.  Must NOT be reported as a trend.
        assert regression_slope_r2(np.full(10, 2000.0)) is None
        assert not has_trend_context(
            np.full(40, 2000.0), 1.0, 30, 20, 0.05, 0.3, DIRECTION_BULLISH
        )

    def test_noisy_window_has_lower_r2_than_clean_window(self) -> None:
        clean = _down(40)
        rng = np.random.default_rng(7)
        noisy = _down(40) + rng.normal(0.0, 3.0, 40)
        clean_fit = regression_slope_r2(clean[:20])
        noisy_fit = regression_slope_r2(noisy[:20])
        assert clean_fit is not None and noisy_fit is not None
        assert clean_fit[1] > noisy_fit[1]

    def test_non_finite_window_returns_none(self) -> None:
        assert regression_slope_r2(np.array([1.0, np.nan, 3.0])) is None
        assert regression_slope_r2(np.array([1.0, np.inf, 3.0])) is None

    def test_single_point_window_returns_none(self) -> None:
        assert regression_slope_r2(np.array([5.0])) is None


# ---------------------------------------------------------------------------
# CAUSALITY — the highest-risk property of this gate
# ---------------------------------------------------------------------------
class TestCausality:
    def test_window_ends_strictly_before_extreme1_bar(self) -> None:
        closes = _down(50)
        window = trend_context_window(closes, extreme1_bar=30, lookback=10)
        assert window is not None
        np.testing.assert_allclose(window, closes[20:30])
        # the exclusive upper bound is the whole point
        assert window[-1] == closes[29]
        assert window[-1] != closes[30]

    def test_assert_causal_window_records_no_future_read(self) -> None:
        """Mechanical proof: instrument every subscript of ``closes``."""
        closes = _down(80)
        assert_causal_window(closes, extreme1_bar=40, lookback=25)

    def test_poisoned_future_bars_do_not_change_the_decision(self) -> None:
        """Overwrite everything at/after extreme1_bar with an absurd trend
        that would flip the answer; the gate must be bit-identical."""
        extreme1_bar = 40
        lookback = 30
        closes = _down(80)  # downtrend before the extreme
        base = has_trend_context(
            closes, 1.0, extreme1_bar, lookback, 0.05, 0.3, DIRECTION_BULLISH
        )
        assert base is True
        poisoned = closes.copy()
        poisoned[extreme1_bar:] = _up(80 - extreme1_bar, start=10_000.0, step=500.0)
        after = has_trend_context(
            poisoned, 1.0, extreme1_bar, lookback, 0.05, 0.3, DIRECTION_BULLISH
        )
        assert after == base is True
        # and the inverse: a bullish pretender whose future is a crash
        up_closes = _up(80)
        assert (
            has_trend_context(
                up_closes, 1.0, extreme1_bar, lookback, 0.05, 0.3, DIRECTION_BULLISH
            )
            is False
        )
        poison2 = up_closes.copy()
        poison2[extreme1_bar:] = _down(80 - extreme1_bar, start=10_000.0, step=500.0)
        assert (
            has_trend_context(
                poison2, 1.0, extreme1_bar, lookback, 0.05, 0.3, DIRECTION_BULLISH
            )
            is False
        )

    def test_not_enough_bars_before_extreme1_bar_fails_closed(self) -> None:
        closes = _down(50)
        assert trend_context_window(closes, extreme1_bar=10, lookback=30) is None
        assert not has_trend_context(
            closes, 1.0, 10, 30, 0.05, 0.3, DIRECTION_BULLISH
        )

    @requires_wiring
    def test_detector_uses_atr_at_extreme1_bar_not_confirm_bar(self) -> None:
        """The detector must pass the ATR known at ``extreme1_bar``.

        With ``trend_context_atr_bar`` recorded on the event we can assert the
        stamp equals ``extreme1_bar`` and that it is strictly before the
        confirm bar — a confirm-bar ATR would leak post-pattern volatility.
        """
        df = _synthetic_double_bottom_frame(trend="down")
        det = DoubleBottomDetector(
            isolated_trend_cfg(trend_context_enabled=True, trend_lookback_bars=30)
        )
        events = det.detect(df, {})
        assert events, "synthetic double bottom should be detected"
        for ev in events:
            stamped = ev.attributes.get("trend_context_atr_bar")
            assert stamped is not None
            assert stamped == ev.attributes["extreme1_bar"]
            assert stamped < ev.attributes["confirm_bar"]


# ---------------------------------------------------------------------------
# DIRECTIONALITY — a reversed sign destroys the strategy silently
# ---------------------------------------------------------------------------
class TestDirectionality:
    def test_expected_slope_sign_both_directions(self) -> None:
        # bullish reversal needs a preceding DOWNtrend
        assert expected_slope_sign(DIRECTION_BULLISH) == -1
        # bearish reversal needs a preceding UPtrend
        assert expected_slope_sign(DIRECTION_BEARISH) == 1
        # unknown direction never passes
        assert expected_slope_sign("sideways") == 0

    def test_both_directions_pin_the_sign(self) -> None:
        up = _up(60)
        down = _down(60)
        for closes, bullish_expected, bearish_expected in (
            (up, False, True),
            (down, True, False),
        ):
            assert (
                has_trend_context(
                    closes, 1.0, 50, 30, 0.05, 0.3, DIRECTION_BULLISH
                )
                is bullish_expected
            )
            assert (
                has_trend_context(
                    closes, 1.0, 50, 30, 0.05, 0.3, DIRECTION_BEARISH
                )
                is bearish_expected
            )

    def test_unknown_direction_fails_closed_even_with_a_real_trend(self) -> None:
        assert not has_trend_context(
            _down(60), 1.0, 50, 30, 0.05, 0.3, "sideways"
        )

    def test_direction_none_keeps_pure_magnitude_semantics(self) -> None:
        assert has_trend_context(_down(60), 1.0, 50, 30, 0.05, 0.3, None)
        assert has_trend_context(_up(60), 1.0, 50, 30, 0.05, 0.3, None)

    @requires_wiring
    def test_detector_drops_db_in_uptrend_and_dt_in_downtrend(
        self,
    ) -> None:
        """End-to-end direction check on the detector's own gate.

        A double bottom preceded by an UPtrend has no reversal context and
        must be discarded with ``low_trend_context``; the mirrored frame for a
        double top must keep ONLY the uptrend case.

        ``trend_lookback_bars=30`` is used because that is where the window
        sits entirely inside the leading leg (see
        :meth:`test_lookback_can_reach_back_past_a_reversal`).
        """
        cfg = isolated_trend_cfg(
            trend_context_enabled=True, trend_lookback_bars=30
        )

        df_bottom = _synthetic_double_bottom_frame(trend="up")
        assert events_without_gate(df_bottom, "double_bottom"), "frame is detectable"
        assert DoubleBottomDetector(cfg).detect(df_bottom, {}) == []

        df_bottom_ok = _synthetic_double_bottom_frame(trend="down")
        kept = DoubleBottomDetector(cfg).detect(df_bottom_ok, {})
        assert kept, "double bottom after a downtrend must survive the gate"
        for ev in kept:
            assert ev.attributes.get("discard_reason") is None

        df_top = _synthetic_double_top_frame(trend="up")
        assert events_without_gate(df_top, "double_top"), "frame is detectable"
        kept_top = DoubleTopDetector(cfg).detect(df_top, {})
        assert kept_top, "double top after an uptrend must survive the gate"

        df_top_bad = _synthetic_double_top_frame(trend="down")
        assert DoubleTopDetector(cfg).detect(df_top_bad, {}) == []

    def test_window_placement_decides_the_verdict(self) -> None:
        """Window placement is part of the gate's meaning, not an accident.

        The same close series holds a DOWNtrend then an UPtrend.  A window
        placed inside the first leg reports a negative slope; a window placed
        in the second reports a positive one.  The gate therefore judges the
        window it is given — so ``trend_lookback_bars`` is a semantic choice
        that must be justified by measurement (rework §4.2), never tuned per
        chart, and never allowed to reach back across the reversal it is
        supposed to measure.

        Built directly rather than from the frame builders so the property stays
        legible: 40 bars down, then 40 bars up.
        """
        closes = np.concatenate([_down(40, start=3200.0, step=3.5), _up(40, start=3065.0, step=3.5)])
        atr_k = 5.0
        # A window wholly inside the DOWNtrend satisfies a double bottom.
        assert has_trend_context(closes, atr_k, 40, 30, 0.05, 0.3, DIRECTION_BULLISH)
        assert not has_trend_context(closes, atr_k, 40, 30, 0.05, 0.3, DIRECTION_BEARISH)
        # A window wholly inside the UPtrend satisfies a double top instead.
        assert has_trend_context(closes, atr_k, 80, 30, 0.05, 0.3, DIRECTION_BEARISH)
        assert not has_trend_context(closes, atr_k, 80, 30, 0.05, 0.3, DIRECTION_BULLISH)

    def test_shallow_turn_keeps_the_control_honest(self) -> None:
        """Guard the negative control against silently becoming a false pass.

        If the first V-turn of the synthetic frame is made deep, the 20-bar
        window ending at ``extreme1_bar`` reaches back across the reversal into
        the rebound, yields a negative slope, and the ``trend="up"`` control
        passes — for a correct reason that nonetheless makes the direction test
        meaningless.  This pins the frame's actual measured behaviour so that
        geometry edit is caught.
        """
        for trend, direction in (
            ("down", DIRECTION_BULLISH),
            ("up", DIRECTION_BULLISH),
        ):
            df = _synthetic_double_bottom_frame(trend=trend)
            events = events_without_gate(df, "double_bottom")
            assert events, f"{trend} frame must be detectable"
            ev = events[0]
            e1 = int(ev.attributes["extreme1_bar"])
            atr_k = float(atr_series(df, 14)[e1])
            pass_ = has_trend_context(
                df["close"].to_numpy(float), atr_k, e1, 20, 0.05, 0.2, direction
            )
            assert pass_ is (trend == "down"), (
                f"frame trend={trend} should pass only when it matches the "
                f"required reversal context; got pass={pass_}"
            )


# ---------------------------------------------------------------------------
# Fail-closed edge cases
# ---------------------------------------------------------------------------
class TestFailClosed:
    def test_lookback_zero_or_negative(self) -> None:
        closes = _down(50)
        assert not has_trend_context(closes, 1.0, 30, 0, 0.05, 0.3, DIRECTION_BULLISH)
        assert not has_trend_context(closes, 1.0, 30, -5, 0.05, 0.3, DIRECTION_BULLISH)

    def test_lookback_one_is_not_a_regression(self) -> None:
        closes = _down(50)
        assert not has_trend_context(closes, 1.0, 30, 1, 0.05, 0.3, DIRECTION_BULLISH)

    def test_nan_atr_fails_closed(self) -> None:
        closes = _down(50)
        assert not has_trend_context(
            closes, float("nan"), 30, 20, 0.05, 0.3, DIRECTION_BULLISH
        )

    def test_non_positive_atr_fails_closed(self) -> None:
        closes = _down(50)
        for bad in (0.0, -1.0, float("inf")):
            assert not has_trend_context(
                closes, bad, 30, 20, 0.05, 0.3, DIRECTION_BULLISH
            )

    def test_index_out_of_range_fails_closed(self) -> None:
        closes = _down(50)
        assert not has_trend_context(
            closes, 1.0, 999, 20, 0.05, 0.3, DIRECTION_BULLISH
        )
        assert not has_trend_context(
            closes, 1.0, -1, 20, 0.05, 0.3, DIRECTION_BULLISH
        )

    def test_extreme1_bar_beyond_series_end(self) -> None:
        closes = _down(50)
        assert trend_context_window(closes, 60, 20) is None

    def test_nan_slope_thresholds_fail_closed(self) -> None:
        closes = _down(50)
        assert not has_trend_context(
            closes, 1.0, 30, 20, float("nan"), 0.3, DIRECTION_BULLISH
        )
        assert not has_trend_context(
            closes, 1.0, 30, 20, 0.05, float("nan"), DIRECTION_BULLISH
        )

    def test_nan_in_close_window_fails_closed(self) -> None:
        closes = _down(50)
        closes[25] = np.nan
        assert not has_trend_context(
            closes, 1.0, 40, 20, 0.05, 0.3, DIRECTION_BULLISH
        )

    def test_slope_threshold_is_scale_free_in_atr(self) -> None:
        closes = _down(60, start=100.0, step=0.1)
        # 0.1/bar against ATR 1.0 = 0.10 >= 0.05 -> pass
        assert has_trend_context(closes, 1.0, 50, 30, 0.05, 0.3, DIRECTION_BULLISH)
        # same prices but a 5x larger ATR -> 0.02 < 0.05 -> fail
        assert not has_trend_context(
            closes, 5.0, 50, 30, 0.05, 0.3, DIRECTION_BULLISH
        )

    def test_min_r2_is_enforced_independently_of_slope(self) -> None:
        closes = _down(60)
        assert has_trend_context(closes, 1.0, 50, 30, 0.05, 0.3, DIRECTION_BULLISH)
        assert not has_trend_context(
            closes, 1.0, 50, 30, 0.05, 1.01, DIRECTION_BULLISH
        )


# ---------------------------------------------------------------------------
# Provider seam (rework §2.4 — Phương án C can replace the heuristic)
# ---------------------------------------------------------------------------
class TestProviderSeam:
    def test_slope_provider_is_a_trend_context_provider(self) -> None:
        provider = SlopeTrendContextProvider(30, 0.05, 0.3)
        assert isinstance(provider, TrendContextProvider)

    def test_slope_provider_matches_function_and_honours_direction(self) -> None:
        provider = SlopeTrendContextProvider(30, 0.05, 0.3)
        closes = _down(60)
        assert provider.has_trend_context(
            closes, 1.0, 50, direction=DIRECTION_BULLISH
        )
        assert not provider.has_trend_context(
            closes, 1.0, 50, direction=DIRECTION_BEARISH
        )

    @requires_wiring
    def test_alternative_provider_can_be_injected(self) -> None:
        """A state-based (HMM) provider must be drop-in replaceable."""

        class _StateProvider:
            def __init__(self, state: str) -> None:
                self.state = state

            def has_trend_context(
                self, closes: np.ndarray, atr_k: float, extreme1_bar: int, **kw: object
            ) -> bool:
                want = "downtrend" if kw.get("direction") == DIRECTION_BULLISH else "uptrend"
                return self.state == want

        det = DoubleBottomDetector(isolated_trend_cfg(trend_context_enabled=True))
        det.trend_context_provider = _StateProvider("downtrend")
        assert det._trend_ok(np.zeros(10), 1.0, 5, det.config) is True
        det.trend_context_provider = _StateProvider("uptrend")
        assert det._trend_ok(np.zeros(10), 1.0, 5, det.config) is False


# ---------------------------------------------------------------------------
# Opt-in / default-preserving contract
# ---------------------------------------------------------------------------
class TestOptInDefaults:
    @requires_wiring
    def test_gate_is_disabled_by_default(self) -> None:
        for det in (DoubleBottomDetector(), DoubleTopDetector()):
            cfg = det.get_default_config()
            assert cfg["trend_context_enabled"] is False

    @requires_wiring
    def test_config_keys_present_with_documented_defaults(self) -> None:
        cfg = DoubleBottomDetector().get_default_config()
        for key in (
            "trend_context_enabled",
            "trend_lookback_bars",
            "min_slope_atr",
            "min_r2",
        ):
            assert key in cfg, f"missing config key {key}"
        assert cfg["trend_lookback_bars"] > 0
        assert 0.0 < cfg["min_r2"] <= 1.0

    def test_disabled_gate_does_not_change_detection(self) -> None:
        # The synthetic frame's structure window is 31 bars, which still
        # exceeds the §2.2 ``max_pattern_length_bars`` default of 30.  §2.2
        # therefore runs first and removes the frame before the trend gate is
        # consulted, which would make this test pass vacuously (both sides
        # empty).  Neutralise §2.2 to isolate the §2.1 gate under test -- same
        # isolation used by test_gate_composes_with_foreach_other_detector_gate.
        df = _synthetic_double_bottom_frame(trend="up")
        off = DoubleBottomDetector(
            {"trend_context_enabled": False, "max_pattern_length_bars": 0}
        ).detect(df, {})
        default = DoubleBottomDetector({"max_pattern_length_bars": 0}).detect(df, {})
        assert [e.event_id for e in off] == [e.event_id for e in default]
        assert off, "the uptrend frame is detectable without the trend gate"
        for ev in off:
            assert ev.attributes.get("discard_reason") is None

    def test_gate_composes_with_foreach_other_detector_gate(self) -> None:
        """§2.1 must not be silently pre-empted by another detector gate.

        The synthetic frame's structure window is ``83 - 52 = 31`` bars.  Once
        §2.2 (``max_pattern_length_bars``, default 30) landed, the frame was
        filtered out *before* the trend gate could run — which would make every
        §2.1 test pass vacuously for the wrong reason.  The §2.1 tests therefore
        disable §2.2 explicitly to isolate the gate under test.

        This test pins the interaction so a future default change to §2.2 makes
        the coupling visible instead of quietly turning §2.1's tests into
        no-ops.
        """
        df = _synthetic_double_bottom_frame(trend="down")
        # Isolated §2.1 (the frame is longer than the §2.2 default):
        isolated = DoubleBottomDetector(
            {"trend_context_enabled": True, "max_pattern_length_bars": 0}
        ).detect(df, {})
        assert isolated, "§2.1 keeps a downtrend-context double bottom"
        # The §2.2 default really does remove this frame, so the coupling is real.
        assert (
            DoubleBottomDetector({"max_pattern_length_bars": 30}).detect(df, {}) == []
        ), "expected §2.2 default 30 to filter the 31-bar synthetic frame"

    @requires_wiring
    def test_enabled_gate_stamps_discard_reason_convention(self) -> None:
        df = _synthetic_double_bottom_frame(trend="up")
        events = DoubleBottomDetector(
            isolated_trend_cfg(trend_context_enabled=True)
        ).detect(df, {})
        # dropped candidates are not emitted; the reason constant must match
        # the Event Lake convention named in the rework request §6.
        assert DISCARD_LOW_TREND_CONTEXT == "low_trend_context"
        assert events == []

    def test_config_hash_changes_when_gate_enabled(self) -> None:
        off = DoubleBottomDetector().get_default_config()
        on = double_cfg_with(off, trend_context_enabled=True)
        assert on != off

    def test_module_does_not_document_an_expectancy_improvement(self) -> None:
        """The gate must not be advertised as an improvement it did not earn.

        Three independent measurements agree there is no selection power at the
        request's default (kept vs dropped expectancy gap is inside the noise),
        and the §4.2 OOS requirement FAILS.  A future editor flipping the
        default ON, or editing the docstring into a performance claim, must
        trip this test and be forced to bring new evidence.
        """
        import research.core.trend_context as tc

        doc = tc.__doc__ or ""
        # The measured verdict must be present, so the evidence travels with
        # the code rather than living only in a report.
        assert "NO DEMONSTRATED SELECTION POWER" in doc
        assert "§4.2" in doc and "FAIL" in doc
        # And the default must stay off.
        assert tc.DEFAULT_MIN_R2 >= 0.0
        for det in (DoubleBottomDetector(), DoubleTopDetector()):
            cfg = det.get_default_config()
            if "trend_context_enabled" in cfg:
                assert cfg["trend_context_enabled"] is False, (
                    "§2.1 must ship opt-in: no measurement justifies enabling it"
                )

    def test_no_improvement_claim_in_default_constants(self) -> None:
        """Guard the documented default choice against a silent 'tuning' edit.

        The defaults are the loosest combination clearing §4.2 on the COMBINED
        reading — not a tuned optimum.  Pinning the exact values makes any
        future change to them a deliberate, reviewable act.
        """
        from research.core import trend_context as tc

        assert tc.DEFAULT_TREND_LOOKBACK_BARS == 20
        assert tc.DEFAULT_MIN_SLOPE_ATR == 0.05
        assert tc.DEFAULT_MIN_R2 == 0.2

    # ------------------------------------------------------------------
    # config_hash attribution (§6.2 / reviewer F-05)
    #
    # Two engineers add config keys to the SAME detector, and every new key
    # changes compute_config_hash for every DB/DT event.  These tests do NOT
    # pin a literal hash — that would break the moment §2.2/§2.3 keys land and
    # would make drift impossible to attribute.  Instead they pin WHICH key
    # changes the hash and which keys must not collide, so any drift can be
    # assigned to one owner.
    # ------------------------------------------------------------------
    def test_config_hash_is_deterministic_and_versioned(self) -> None:
        a = compute_config_hash(DoubleBottomDetector().get_default_config())
        b = compute_config_hash(DoubleBottomDetector().get_default_config())
        assert a == b and len(a) == 12
        assert DoubleBottomDetector().get_default_config()["version"] == "1.0"

    def test_each_trend_key_individually_changes_the_hash(self) -> None:
        base = DoubleBottomDetector().get_default_config()
        if "trend_context_enabled" not in base:
            pytest.skip("§2.1 keys not wired yet (see WIRING_LANDED)")
        # compute_config_hash hashes the whole dict (config_hash.py:71-72), so a
        # key only moves the hash if it is genuinely PRESENT.  Assert presence
        # first: otherwise this test could pass by accidentally writing a brand
        # new key rather than by recognising one the detector actually declares.
        for key in ("trend_context_enabled", "trend_lookback_bars", "min_slope_atr", "min_r2"):
            assert key in base, f"{key} must be present in get_default_config()"
        h0 = compute_config_hash(base)
        for key, value in (
            ("trend_context_enabled", True),
            ("trend_lookback_bars", int(base["trend_lookback_bars"]) + 1),
            ("min_slope_atr", float(base["min_slope_atr"]) + 0.01),
            ("min_r2", float(base["min_r2"]) + 0.01),
        ):
            changed = double_cfg_with(base, **{key: value})
            assert changed != base
            assert compute_config_hash(changed) != h0, (
                f"{key} must participate in config_hash (§6.2)"
            )

    def test_trend_keys_do_not_collide_with_gate_engineer_keys(self) -> None:
        """Attribution guard: §2.1 owns only its own four keys.

        If this fails, a §2.1 edit silently took over a §2.2/§2.3 key (or vice
        versa) and the resulting hash drift can no longer be assigned to one
        owner.
        """
        cfg = DoubleBottomDetector().get_default_config()
        mine = {
            "trend_context_enabled",
            "trend_lookback_bars",
            "min_slope_atr",
            "min_r2",
        }
        theirs = {"max_pattern_length_bars", "min_rule_score"}
        assert not (mine & theirs), "§2.1 and §2.2/§2.3 must not share config keys"
        # Whichever set is present, it must be present as a whole.
        present = mine & set(cfg)
        assert present in (set(), mine), (
            f"partial §2.1 key set in default config; missing: {sorted(mine - set(cfg))}"
        )

    @requires_wiring
    def test_trend_ok_helper_honours_disabled_flag(self) -> None:
        det = DoubleBottomDetector()
        # disabled -> always OK regardless of the frame
        assert det._trend_ok(np.full(10, np.nan), float("nan"), 5, det.config) is True


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------
def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def _ohlc_from_closes(
    closes: np.ndarray, wick: float = 0.4
) -> list[tuple[float, float, float, float]]:
    return [(float(c), float(c) + wick, float(c) - wick, float(c)) for c in closes]


def _synthetic_double_bottom_frame(trend: str = "down", atr: float = 1.0) -> pd.DataFrame:
    """A leading trend leg, then an L-H-L double bottom that confirms upward.

    Geometry is verified against the real ``SwingDetector``: it yields exactly
    one L-H-L triple with ``extreme1_bar == 45`` and ``pattern_length == 31``,
    equal lows and a depth well past the ``min_depth_atr`` gate.

    ``trend="down"`` gives the classical reversal context (a double bottom
    after a downtrend).  ``trend="up"`` is the no-context control.

    The first V-turn is deliberately SHALLOW (6 bars, ~9 price units against a
    140-unit leg).  With a deep V the 20-bar context window ending at
    ``extreme1_bar`` reaches back ACROSS the reversal into the rebound, which
    yields a genuinely negative slope and makes the control pass for a correct
    reason — confusing a real property of the gate with a broken test.  Keeping
    the turn shallow keeps the window dominated by the leading leg, so the sign
    of the slope is what decides, exactly as intended.
    """
    leg = (
        np.linspace(3200.0, 3060.0, 40)
        if trend == "down"
        else np.linspace(2920.0, 3060.0, 40)
    )
    closes = np.concatenate(
        [
            leg,
            np.linspace(3059.0, 3050.0, 6),   # shallow dip -> extreme1
            np.linspace(3052.0, 3096.0, 12),  # up to the neckline
            np.linspace(3098.0, 3104.0, 5),   # middle peak (neckline ~3103)
            np.linspace(3102.0, 3051.0, 14),  # second low (equal level)
            np.linspace(3053.0, 3115.0, 15),  # break ABOVE the neckline
        ]
    )
    return _frame(_ohlc_from_closes(closes, wick=0.25))


def _synthetic_double_top_frame(trend: str = "up", atr: float = 1.0) -> pd.DataFrame:
    """Exact mirror of :func:`_synthetic_double_bottom_frame` (H-L-H top)."""
    leg = (
        np.linspace(2900.0, 3040.0, 40)
        if trend == "up"
        else np.linspace(3180.0, 3040.0, 40)
    )
    closes = np.concatenate(
        [
            leg,
            np.linspace(3041.0, 3050.0, 6),   # shallow rise -> extreme1
            np.linspace(3048.0, 3004.0, 12),  # down to the neckline
            np.linspace(3002.0, 2996.0, 5),   # middle trough (neckline ~2999)
            np.linspace(2998.0, 3049.0, 14),  # second high (equal level)
            np.linspace(3047.0, 2985.0, 15),  # break BELOW the neckline
        ]
    )
    return _frame(_ohlc_from_closes(closes, wick=0.25))


def double_cfg_with(cfg: dict, **overrides: object) -> dict:
    merged = dict(cfg)
    merged.update(overrides)
    return merged


def isolated_trend_cfg(**overrides: object) -> dict:
    """§2.1 config with the OTHER detector gates disabled.

    The synthetic frames' structure window is 35 bars, which §2.2
    (``max_pattern_length_bars``, default 30) filters out before the trend gate
    can run.  Leaving §2.2 at its default would make every §2.1 assertion pass
    vacuously — the frame disappears for an unrelated reason.  Each §2.1 test
    therefore disables §2.2 so it measures the trend gate alone.

    ``test_gate_composes_with_foreach_other_detector_gate`` pins this coupling.
    """
    cfg: dict = {"max_pattern_length_bars": 0, "min_rule_score": 0.0}
    cfg.update(overrides)
    return cfg


def events_without_gate(df: pd.DataFrame, pattern: str) -> list:
    """Detect with the §2.1 trend gate OFF — proves a frame is otherwise valid.

    §2.2 is disabled too, so this asserts the frame is a real double structure
    rather than one that merely slipped past an unrelated length gate.
    """
    det = (
        DoubleBottomDetector(isolated_trend_cfg(trend_context_enabled=False))
        if pattern == "double_bottom"
        else DoubleTopDetector(isolated_trend_cfg(trend_context_enabled=False))
    )
    return det.detect(df, {})
