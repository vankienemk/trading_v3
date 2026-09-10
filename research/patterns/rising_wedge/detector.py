"""
detector.py — Rising / Falling Wedge plugin (P3, Agent 4).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P3 rising_wedge / falling_wedge) and
§13 Agent 4.  Implements the converging-channel reversal on the shared
:class:`research.core.swing_detector.SwingDetector` (§3.2 Right-Bar Rule)
and the shared trendline infra (``research.core.trendline`` — linear
regression on pivots + ATR tolerance).

Geometry (bearish rising wedge; the falling wedge is the mirror — see
``patterns/falling_wedge/PATTERN_SPECS.md``):

* 3 monotonic swing LOWS (ascending for a rising wedge / descending for a
  falling wedge) plus ≥ 2 monotonic swing HIGHS strictly between them, all
  forming a CHANNEL between a lower trendline (fit on the lows) and an upper
  trendline (fit on the highs).  The channel must CONVERGE (upper & lower
  lines get closer — the defining wedge property) and keep most closes
  inside it with an ATR tolerance (anti-noise gate);
* confirmation: after the LAST low's pivot is causally known
  (``low3.bar + right_bars``, §3.2), a close crosses the OPPOSING channel
  line decisively — above the upper line (falling wedge, bullish) or below
  the lower line (rising wedge, bearish);
* entry: open of the next bar after confirmation; stop: beyond the far side
  of the wedge (min low - ATR buffer for bullish, max high + ATR buffer for
  bearish); target: ``target_r`` R-multiples.

Causality (§3): every scoring feature is available at ``detect`` (data up to
the last pivot-known bar) or ``confirm`` (the confirm bar itself) — never
past ``known_at``.  ``detect()`` stamps each event with its ``config_hash``
(§6.2) and ``feature_schema_version``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from research.core.config_hash import compute_config_hash
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    DIRECTION_BULLISH,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
)
from research.core.swing_detector import SwingDetector
from research.core.trendline import (
    Trendline,
    channel_width,
    closes_inside_channel,
    convergence_ratio,
    fit_trendline,
)
from research.patterns.double_bottom.detector import atr_series

#: version of the wedge feature schema (bump on semantic change)
FEATURE_SCHEMA_VERSION = "wedge-v1.0"


def median_range(df: pd.DataFrame) -> float:
    """Robust per-bar range = median(high - low) over the frame.

    Absolute-ish noise-scale floor (mirrors the synthesizer reference
    detector's ``_vol``, spec §8).  A pure-noise OU series has a tiny
    median range, so ``depth ≥ min_vol_span * median_range`` separates a
    genuine multi-bar reversal swing from sub-range noise that a purely
    ATR-relative gate would let through (ATR rescales with the noise and
    cannot alone reject an OU control that is itself a many-ATR zig-zag).
    """
    return float(np.median((df["high"] - df["low"]).to_numpy()))


class WedgePatternDetectorBase(BasePatternDetector):
    """Shared machinery for rising_wedge (bearish) and falling_wedge (bullish).

    Subclasses set ``name`` / ``short_name`` / ``direction`` — everything
    else (swing scan, trendline fit + ATR tolerance, convergence/interior
    gates, confirmation, entry/SL/TP, features, config, causality stamps) is
    shared (spec §11: falling_wedge = mirror of rising_wedge).
    """

    name: str = "wedge_pattern"
    version: str = "1.0"
    short_name: str = "WG"
    #: canonical breakout direction of the wedge (bullish falling / bearish rising)
    direction: str = DIRECTION_BULLISH

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(
            "atr", "float", AVAILABLE_AT_DETECT, False,
            "ATR(period) at the last pivot-known (detect) bar",
        ),
        PatternFeature(
            "depth_atr", "float", AVAILABLE_AT_DETECT, False,
            "total monotonic low swing across the 3 lows, in ATR",
        ),
        PatternFeature(
            "width_atr", "float", AVAILABLE_AT_DETECT, False,
            "channel width at detect (upper-lower), in ATR",
        ),
        PatternFeature(
            "convergence_ratio", "float", AVAILABLE_AT_DETECT, False,
            "how much the channel narrowed from first to last high touch",
        ),
        PatternFeature(
            "interior_ratio", "float", AVAILABLE_AT_DETECT, False,
            "fraction of closes inside the channel (ATR tolerance)",
        ),
        PatternFeature(
            "pattern_length", "int", AVAILABLE_AT_DETECT, False,
            "bars between the first and last low touch",
        ),
        PatternFeature(
            "line_error_atr", "float", AVAILABLE_AT_DETECT, False,
            "max trendline anchor deviation in ATR (fit quality)",
        ),
        PatternFeature(
            "confirm_pierce_atr", "float", AVAILABLE_AT_CONFIRM, False,
            "confirm close past the opposing channel line, in ATR",
        ),
        PatternFeature(
            "confirm_range_atr", "float", AVAILABLE_AT_CONFIRM, False,
            "confirm bar range in ATR",
        ),
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self.get_default_config()
        if config:
            merged = dict(self.config)
            merged.update(config)
            merged.setdefault("version", self.version)
            self.config = merged

    # ------------------------------------------------------------------
    # BasePatternDetector contract
    # ------------------------------------------------------------------
    def get_default_config(self) -> dict[str, Any]:
        """Default wedge config -- MUST carry "version" (§6.2).

        Gates (calibrated on XAUUSD M15 2018-2026 + §8.3 noise control):
        a wedge needs a real multi-ATR low swing (2.0 ATR), a trendline fit
        within 1.2 ATR of its pivot anchors, a channel that narrows by at
        least 10 %, and ≥ 70 % of closes inside the channel (0.8 ATR band).
        """
        return {
            "version": self.version,
            "left_bars": 3,
            "right_bars": 3,
            "atr_period": 14,
            "min_separation_bars": 2,
            "min_depth_atr": 1.2,
            # absolute noise floor: depth must also be ≥ min_vol_span *
            # median(high-low) so a pure-noise OU zig-zag (whose own ATR is
            # tiny) cannot satisfy a wedge (spec §8 control).  2.0 keeps the
            # §8.3 FP ≤ 5 % while clearing the §6.3 n ≥ 300 sample gate on
            # XAUUSD M15 for the falling wedge.
            "min_vol_span": 2.0,
            # minimum horizontal span (bars between first and last low) — OU
            # rarely keeps that many bars monotone.
            "min_pattern_bars": 8,
            "max_line_error_atr": 1.4,
            "min_convergence": 0.05,
            "min_interior": 0.70,
            "interior_tol_atr": 0.8,
            "stop_buffer_atr": 0.5,
            "target_r": 1.5,
            # §3.3 staleness: confirm must fire within this many bars after
            # the setup is complete (last pivot known), else discard.
            "max_bars_between_detect_and_confirm": 60,
            "cooldown_bars": 5,
            "symbol": "XAUUSD",
            "timeframe": "M15",
        }

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        cfg = self.config
        if config:
            cfg = {**self.config, **config, "version": self.version}
        return self._detect_wedge(df, cfg)

    # ------------------------------------------------------------------
    # Detection core (both directions)
    # ------------------------------------------------------------------
    def _detect_wedge(self, df: pd.DataFrame, cfg: dict[str, Any]) -> list[PatternEvent]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("wedge detector requires a DatetimeIndex")
        if len(df) < 12:
            return []

        bull = self.direction == DIRECTION_BULLISH
        left = int(cfg["left_bars"])
        right = int(cfg["right_bars"])
        min_sep = int(cfg["min_separation_bars"])
        min_depth_atr = float(cfg["min_depth_atr"])
        min_vol_span = float(cfg["min_vol_span"])
        min_pattern_bars = int(cfg["min_pattern_bars"])
        max_line_error_atr = float(cfg["max_line_error_atr"])
        min_convergence = float(cfg["min_convergence"])
        min_interior = float(cfg["min_interior"])
        interior_tol_atr = float(cfg["interior_tol_atr"])
        stop_buffer_atr = float(cfg["stop_buffer_atr"])
        target_r = float(cfg["target_r"])
        max_wait = int(cfg["max_bars_between_detect_and_confirm"])
        cooldown = int(cfg["cooldown_bars"])
        symbol = str(cfg.get("symbol", "XAUUSD"))
        timeframe = str(cfg.get("timeframe", "M15"))

        sw = SwingDetector(left, right).find_swings(df)
        if len(sw) < 5:
            return []
        lows = [s for s in sw if s.kind == "L"]
        highs = [s for s in sw if s.kind == "H"]
        if len(lows) < 3:
            return []

        atr = atr_series(df, int(cfg["atr_period"]))
        med_range = median_range(df)
        closes = df["close"].to_numpy(dtype=float)
        bar_high = df["high"].to_numpy(dtype=float)
        bar_low = df["low"].to_numpy(dtype=float)
        n = len(df)
        config_hash = compute_config_hash(cfg)
        candidates: list[PatternEvent] = []

        for k in range(len(lows) - 2):
            s0, s1, s2 = lows[k], lows[k + 1], lows[k + 2]
            if not (s0.bar < s1.bar < s2.bar):
                continue
            if s1.bar - s0.bar < min_sep or s2.bar - s1.bar < min_sep:
                continue
            detect_bar = s2.known_at_bar  # last pivot causally known (§3.2)
            if detect_bar >= n or detect_bar >= len(atr) or np.isnan(atr[detect_bar]):
                continue
            atr_k = float(atr[detect_bar])
            if atr_k <= 0.0:
                continue

            v0, v1, v2 = s0.price, s1.price, s2.price
            if bull:
                if not (v0 > v1 > v2):  # falling wedge: descending lows
                    continue
                depth = v0 - v2
            else:
                if not (v0 < v1 < v2):  # rising wedge: ascending lows
                    continue
                depth = v2 - v0
            # An absolute-ish vertical floor: depth must exceed both a
            # multi-ATR multiple AND a multi median-bar-range multiple.  The
            # latter rejects the OU control (whose ATR rescales with noise).
            if depth < min_depth_atr * atr_k or depth < min_vol_span * med_range:
                continue
            if s2.bar - s0.bar < min_pattern_bars:
                continue

            # ≥2 monotonic swing highs strictly between the first and last low
            hw = [s for s in highs if s0.bar < s.bar < s2.bar]
            if len(hw) < 2:
                continue
            hprices = [s.price for s in hw]
            if bull and not all(
                hprices[i] > hprices[i + 1] for i in range(len(hprices) - 1)
            ):
                continue  # falling wedge: descending highs
            if not bull and not all(
                hprices[i] < hprices[i + 1] for i in range(len(hprices) - 1)
            ):
                continue  # rising wedge: ascending highs

            # fit channel lines on the pivots (linear regression) + ATR tolerance
            lo_line = fit_trendline([(s0.bar, v0), (s1.bar, v1), (s2.bar, v2)])
            hi_line = fit_trendline([(s.bar, s.price) for s in hw])
            if (
                lo_line.max_error_atr(atr_k) > max_line_error_atr
                or hi_line.max_error_atr(atr_k) > max_line_error_atr
            ):
                continue
            # channel must converge (narrow) between the first and last high
            a_bar, b_bar = hw[0].bar, hw[-1].bar
            if convergence_ratio(hi_line, lo_line, a_bar, b_bar) < min_convergence:
                continue
            if channel_width(hi_line, lo_line, s2.bar) <= 0.0:
                continue
            interior = closes_inside_channel(
                df, hi_line, lo_line, s0.bar, s2.bar, interior_tol_atr, atr_k
            )
            if interior < min_interior:
                continue

            # Confirmation: close crosses the opposing channel line after the
            # last low's pivot exists.  Causal stamp = max(cross, detect_bar).
            if bull:
                def level(j: int, _line: Trendline = hi_line) -> float:
                    return _line.at(j)
                above = True
            else:
                def level(j: int, _line: Trendline = lo_line) -> float:
                    return _line.at(j)
                above = False
            cross: int | None = None
            scan_end = min(n, s2.bar + 1 + max_wait)
            for j in range(s2.bar + 1, scan_end):
                lv = level(j)
                if above and closes[j] > lv:
                    cross = j
                    break
                if not above and closes[j] < lv:
                    cross = j
                    break
            if cross is None:
                continue
            confirm_bar = max(cross, detect_bar)
            entry_bar = confirm_bar + 1 if confirm_bar + 1 < n else confirm_bar

            entry_price = (
                float(closes[entry_bar])
                if entry_bar == confirm_bar
                else float(df["open"].iloc[entry_bar])
            )
            wedge_low = min(v0, v1, v2)
            wedge_high = max(s.price for s in hw)
            if bull:
                stop_price = wedge_low - stop_buffer_atr * atr_k
            else:
                stop_price = wedge_high + stop_buffer_atr * atr_k
            risk = abs(entry_price - stop_price)
            if risk <= 0.0:
                continue
            if bull:
                target_price = entry_price + target_r * risk
            else:
                target_price = entry_price - target_r * risk

            pierce = (
                closes[confirm_bar] - level(confirm_bar)
                if bull
                else level(confirm_bar) - closes[confirm_bar]
            ) / atr_k
            # F1 (handoff §3 P1#5): emit the declared confirm_range_atr
            # feature (confirm bar range in ATR, normalized by ATR at the
            # confirm bar — same convention as the gate scorer rng_det).
            atr_confirm = float(atr[confirm_bar]) if not np.isnan(atr[confirm_bar]) else atr_k
            confirm_range_atr = (bar_high[confirm_bar] - bar_low[confirm_bar]) / max(atr_confirm, 1e-12)
            width_atr = channel_width(hi_line, lo_line, detect_bar) / atr_k
            line_error = max(
                lo_line.max_error_atr(atr_k), hi_line.max_error_atr(atr_k)
            )
            conv = max(0.0, convergence_ratio(hi_line, lo_line, a_bar, b_bar))
            length = s2.bar - s0.bar

            rule_score = min(
                1.0,
                0.35 * min(depth / (min_depth_atr * atr_k), 1.0)
                + 0.25 * min(conv / 0.3, 1.0)
                + 0.20 * min(interior, 1.0)
                + 0.20 * min(pierce / 1.0, 1.0),
            )

            structure_levels: dict[str, float] = {
                "upper_trendline_slope": hi_line.slope,
                "upper_trendline_intercept": hi_line.intercept,
                "lower_trendline_slope": lo_line.slope,
                "lower_trendline_intercept": lo_line.intercept,
                "upper_line_at_detect": hi_line.at(detect_bar),
                "lower_line_at_detect": lo_line.at(detect_bar),
                # dataset gate features read a canonical "neckline" level
                "neckline": hi_line.at(detect_bar) if bull else lo_line.at(detect_bar),
                "wedge_high": wedge_high,
                "wedge_low": wedge_low,
            }
            attributes: dict[str, Any] = {
                "low1_bar": s0.bar,
                "low2_bar": s1.bar,
                "low3_bar": s2.bar,
                "high_touch_bars": [s.bar for s in hw],
                "pivot_known_at_bar": detect_bar,
                "close_cross_bar": int(cross),
                "confirm_bar": int(confirm_bar),
                "entry_bar": int(entry_bar),
                "depth_atr": float(depth / atr_k),
                "low_offset_atr": float(line_error),
                "confirm_range_atr": float(confirm_range_atr),
                "width_atr": float(width_atr),
                "convergence_ratio": float(conv),
                "interior_ratio": float(interior),
                "left_len": int(s1.bar - s0.bar),
                "right_len": int(s2.bar - s1.bar),
                "pattern_length": int(length),
                "discard_reason": None,
            }
            candidates.append(
                PatternEvent(
                    event_id=f"{symbol}-{self.short_name}-{len(candidates):06d}",
                    pattern_name=self.name,
                    pattern_version=self.version,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=self.direction,
                    detect_time=df.index[detect_bar],
                    confirm_time=df.index[confirm_bar],
                    entry_time=df.index[entry_bar],
                    entry_price=float(entry_price),
                    stop_price=float(stop_price),
                    target_price=float(target_price),
                    structure_levels=structure_levels,
                    rule_score=float(rule_score),
                    model_prob=None,
                    attributes=attributes,
                    config_hash=config_hash,
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                )
            )

        return self._dedupe(candidates, cooldown)

    @staticmethod
    def _dedupe(events: list[PatternEvent], cooldown: int) -> list[PatternEvent]:
        """Cooldown dedup on the confirm bar (spec §4 -- overlapping wedges
        along the same channel are the same idea)."""
        if not events:
            return []
        bar_of = {ev.event_id: int(ev.attributes["confirm_bar"]) for ev in events}
        ordered = sorted(events, key=lambda ev: (bar_of[ev.event_id], -ev.rule_score))
        kept: list[PatternEvent] = []
        last_confirmed: int | None = None
        for ev in ordered:
            cb = bar_of[ev.event_id]
            if last_confirmed is not None and cb - last_confirmed < cooldown:
                continue
            kept.append(ev)
            last_confirmed = cb
        kept.sort(key=lambda ev: int(ev.attributes["confirm_bar"]))
        return kept


class RisingWedgeDetector(WedgePatternDetectorBase):
    """Rising Wedge (P3) -- bearish reversal off a converging rising channel."""

    name = "rising_wedge"
    version = "1.0"
    short_name = "RW"
    direction = "bearish"


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> RisingWedgeDetector:
    return RisingWedgeDetector(config)


DETECTOR_CLASS = RisingWedgeDetector