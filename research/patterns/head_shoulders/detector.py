"""
detector.py — Head & Shoulders / Inverse plugin (P4, Agent 4).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P4) and §13 Agent 4.  Implements the
classical 5-point reversal on the shared
:class:`research.core.swing_detector.SwingDetector` (§3.2 Right-Bar Rule).

Geometry (bearish head & shoulders; the inverse is the exact mirror — see
``patterns/inverse_head_shoulders/PATTERN_SPECS.md``):

* five swings: left SHOULDER high, neckline trough, HEAD high (strictly
  above both shoulders), neckline trough, right SHOULDER high — the two
  troughs defining the neckline and the two shoulders near the same level;
* confirmation: after the RIGHT shoulder's pivot is causally known
  (``right_shoulder.bar + right_bars``, §3.2), a close crosses the neckline
  (higher trough for a bearish H&S) downward;
* entry: open of the next bar after confirmation; stop: above the head
  (+ ATR buffer); target: ``target_r`` R-multiples.

§3.3 staleness is central here (the spec marks H&S as the pattern where the
staleness window matters most — "staleness dài"): a candidate whose neckline
cross does not fire within ``max_bars_between_detect_and_confirm`` (default
60 bars on M15) after the right-shoulder pivot is known is DISCARDED
(never emitted — the Event Lake logs it with ``discard_reason``, which is
the engine's responsibility per §3.3).

Causality (§3): every scoring feature is available at ``detect`` (data up to
the right-shoulder known bar) or ``confirm`` (the confirm bar itself) —
never past ``known_at``.  ``detect()`` stamps each event with its
``config_hash`` (§6.2) and ``feature_schema_version``.
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
from research.patterns.double_bottom.detector import atr_series
from research.patterns.rising_wedge.detector import median_range

#: version of the H&S feature schema (bump on semantic change)
FEATURE_SCHEMA_VERSION = "hs-v1.0"


class HeadShouldersDetectorBase(BasePatternDetector):
    """Shared machinery for head_shoulders (bearish) and its inverse (bullish).

    Subclasses set ``name`` / ``short_name`` / ``direction`` / ``kind_seq`` —
    everything else (swing scan, gates, confirmation, entry/SL/TP, features,
    config, causality stamps) is shared (spec §11: inverse = mirror with
    "100 % infra").
    """

    name: str = "head_shoulders_pattern"
    version: str = "1.0"
    short_name: str = "HS"
    #: canonical breakout direction (bearish H&S / bullish inverse)
    direction: str = DIRECTION_BULLISH
    #: swing kind sequence: (H,L,H,L,H) H&S, (L,H,L,H,L) inverse
    kind_seq: tuple[str, ...] = ("H", "L", "H", "L", "H")

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(
            "atr", "float", AVAILABLE_AT_DETECT, False,
            "ATR(period) at the right-shoulder known (detect) bar",
        ),
        PatternFeature(
            "head_depth_atr", "float", AVAILABLE_AT_DETECT, False,
            "head vs higher-trough (lower for inverse) depth in ATR",
        ),
        PatternFeature(
            "shoulder_offset_atr", "float", AVAILABLE_AT_DETECT, False,
            "left/right shoulder level offset in ATR (symmetry fidelity)",
        ),
        PatternFeature(
            "neckline_offset_atr", "float", AVAILABLE_AT_DETECT, False,
            "two neckline troughs level offset in ATR (neckline fidelity)",
        ),
        PatternFeature(
            "symmetry_ratio", "float", AVAILABLE_AT_DETECT, False,
            "left/right arm length symmetry in [0,1]",
        ),
        PatternFeature(
            "pattern_length", "int", AVAILABLE_AT_DETECT, False,
            "bars between the two shoulders",
        ),
        PatternFeature(
            "confirm_pierce_atr", "float", AVAILABLE_AT_CONFIRM, False,
            "confirm close past the neckline, in ATR",
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
        """Default H&S config -- MUST carry "version" (§6.2).

        Gates calibrated on XAUUSD M15 2018-2026 + §8.3 noise control:
        head depth ≥ 2.0 ATR above/below the neckline, shoulder and neckline
        offsets ≤ 1.5 ATR.  §3.3 staleness default 60 bars on M15.
        """
        return {
            "version": self.version,
            "left_bars": 3,
            "right_bars": 3,
            "atr_period": 14,
            "min_separation_bars": 2,
            "max_shoulder_atr": 1.5,
            "max_neck_atr": 1.5,
            "min_head_depth_atr": 2.0,
            # absolute noise floor: head depth must also be ≥ min_head_vol_span *
            # median(high-low) so a pure-noise OU control (whose own ATR is
            # tiny) cannot emulate a real neckline breakout (spec §8 control).
            "min_head_vol_span": 4.0,
            # minimum horizontal span between the two shoulders — noise rarely
            # keeps a 5-point alternation monotone across this many bars.
            "min_pattern_bars": 12,
            "stop_buffer_atr": 0.5,
            "target_r": 1.5,
            # §3.3 staleness: neckline cross must fire within this many bars
            # after the setup is complete (right-shoulder pivot known), else
            # the candidate is discarded.  H&S = the long-staleness pattern.
            "max_bars_between_detect_and_confirm": 60,
            "cooldown_bars": 5,
            "symbol": "XAUUSD",
            "timeframe": "M15",
        }

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        cfg = self.config
        if config:
            cfg = {**self.config, **config, "version": self.version}
        return self._detect_hs(df, cfg)

    # ------------------------------------------------------------------
    # Detection core (both directions)
    # ------------------------------------------------------------------
    def _detect_hs(self, df: pd.DataFrame, cfg: dict[str, Any]) -> list[PatternEvent]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("H&S detector requires a DatetimeIndex")
        if len(df) < 12:
            return []

        bull = self.direction == DIRECTION_BULLISH
        left = int(cfg["left_bars"])
        right = int(cfg["right_bars"])
        min_sep = int(cfg["min_separation_bars"])
        max_shoulder_atr = float(cfg["max_shoulder_atr"])
        max_neck_atr = float(cfg["max_neck_atr"])
        min_head_depth_atr = float(cfg["min_head_depth_atr"])
        min_head_vol_span = float(cfg["min_head_vol_span"])
        min_pattern_bars = int(cfg["min_pattern_bars"])
        stop_buffer_atr = float(cfg["stop_buffer_atr"])
        target_r = float(cfg["target_r"])
        max_wait = int(cfg["max_bars_between_detect_and_confirm"])
        cooldown = int(cfg["cooldown_bars"])
        symbol = str(cfg.get("symbol", "XAUUSD"))
        timeframe = str(cfg.get("timeframe", "M15"))

        sw = SwingDetector(left, right).find_swings(df)
        if len(sw) < 5:
            return []
        atr = atr_series(df, int(cfg["atr_period"]))
        med_range = median_range(df)
        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        n = len(df)

        want = self.kind_seq
        config_hash = compute_config_hash(cfg)
        candidates: list[PatternEvent] = []

        for k in range(len(sw) - 4):
            s1, s2, s3, s4, s5 = sw[k], sw[k + 1], sw[k + 2], sw[k + 3], sw[k + 4]
            if (s1.kind, s2.kind, s3.kind, s4.kind, s5.kind) != want:
                continue
            i1, v1 = s1.bar, s1.price  # left shoulder
            i2, v2 = s2.bar, s2.price  # neckline trough 1
            i3, v3 = s3.bar, s3.price  # head
            i4, v4 = s4.bar, s4.price  # neckline trough 2
            i5, v5 = s5.bar, s5.price  # right shoulder
            if not (i1 < i2 < i3 < i4 < i5):
                continue
            if i2 - i1 < min_sep or i3 - i2 < min_sep or i4 - i3 < min_sep or i5 - i4 < min_sep:
                continue
            detect_bar = s5.known_at_bar  # right shoulder causally known (§3.2)
            if detect_bar >= n or detect_bar >= len(atr) or np.isnan(atr[detect_bar]):
                continue
            atr_k = float(atr[detect_bar])
            if atr_k <= 0.0:
                continue

            if bull:
                # inverse H&S (L-H-L-H-L): head trough strictly below the
                # shoulder troughs; shoulders + neckline peaks near-equal.
                if not (v3 < v1 and v3 < v5):
                    continue
                if abs(v1 - v5) > max_shoulder_atr * atr_k:
                    continue
                if abs(v2 - v4) > max_neck_atr * atr_k:
                    continue
                head_depth = min(v2, v4) - v3
                neck = min(v2, v4)
                if head_depth < min_head_depth_atr * atr_k:
                    continue
                if head_depth < min_head_vol_span * med_range:
                    continue
            else:
                # H&S (H-L-H-L-H): head peak strictly above both shoulder
                # peaks; shoulders + neckline troughs near-equal.
                if not (v3 > v1 and v3 > v5):
                    continue
                if abs(v1 - v5) > max_shoulder_atr * atr_k:
                    continue
                if abs(v2 - v4) > max_neck_atr * atr_k:
                    continue
                head_depth = v3 - max(v2, v4)
                neck = max(v2, v4)
                if head_depth < min_head_depth_atr * atr_k:
                    continue
                if head_depth < min_head_vol_span * med_range:
                    continue
            if i5 - i1 < min_pattern_bars:
                continue

            # Confirmation: close crosses the neckline after the right
            # shoulder's pivot exists.  Causal stamp = max(cross, detect_bar).
            cross: int | None = None
            scan_end = min(n, i5 + 1 + max_wait)
            for j in range(i5 + 1, scan_end):
                if bull and closes[j] > neck:
                    cross = j
                    break
                if not bull and closes[j] < neck:
                    cross = j
                    break
            if cross is None:
                continue
            confirm_bar = max(cross, detect_bar)
            entry_bar = confirm_bar + 1 if confirm_bar + 1 < n else confirm_bar

            entry_price = (
                float(opens[entry_bar]) if entry_bar != confirm_bar else float(closes[confirm_bar])
            )
            if bull:
                stop_price = v3 - stop_buffer_atr * atr_k
            else:
                stop_price = v3 + stop_buffer_atr * atr_k
            risk = abs(entry_price - stop_price)
            if risk <= 0.0:
                continue
            if bull:
                target_price = entry_price + target_r * risk
            else:
                target_price = entry_price - target_r * risk

            head_depth_atr = head_depth / atr_k
            shoulder_offset_atr = abs(v1 - v5) / atr_k
            neckline_offset_atr = abs(v2 - v4) / atr_k
            left_len = i3 - i1
            right_len = i5 - i3
            symmetry = 1.0 - min(abs(left_len - right_len) / max(left_len, right_len), 1.0)
            pierce = (
                closes[confirm_bar] - neck if bull else neck - closes[confirm_bar]
            ) / atr_k
            # F1 (handoff §3 P1#5): emit the declared confirm_range_atr
            # feature (confirm bar range in ATR, normalized by ATR at the
            # confirm bar — same convention as the gate scorer rng_det).
            atr_confirm = float(atr[confirm_bar]) if not np.isnan(atr[confirm_bar]) else atr_k
            confirm_range_atr = (highs[confirm_bar] - lows[confirm_bar]) / max(atr_confirm, 1e-12)

            rule_score = min(
                1.0,
                0.40 * min(head_depth_atr / min_head_depth_atr, 1.0)
                + 0.20 * (1.0 - min(shoulder_offset_atr / max_shoulder_atr, 1.0))
                + 0.15 * (1.0 - min(neckline_offset_atr / max_neck_atr, 1.0))
                + 0.15 * symmetry
                + 0.10 * min(pierce / 1.0, 1.0),
            )

            level_key = "inverse_hs_level" if bull else "head_shoulders_level"
            structure_levels: dict[str, float] = {
                "neckline": neck,
                level_key: v3,
                "left_shoulder_level": v1,
                "right_shoulder_level": v5,
                "head_level": v3,
            }
            attributes: dict[str, Any] = {
                "left_shoulder_bar": i1,
                "neckline1_bar": i2,
                "head_bar": i3,
                "neckline2_bar": i4,
                "right_shoulder_bar": i5,
                "pivot_known_at_bar": detect_bar,
                "close_cross_bar": int(cross),
                "confirm_bar": int(confirm_bar),
                "entry_bar": int(entry_bar),
                "head_depth_atr": float(head_depth_atr),
                "shoulder_offset_atr": float(shoulder_offset_atr),
                "neckline_offset_atr": float(neckline_offset_atr),
                "confirm_range_atr": float(confirm_range_atr),
                # compatibility aliases consumed by the shared §6.3 gate scorer
                # (research.patterns.double_bottom.dataset.gate_features):
                # "depth_atr" ↔ head depth, "low_offset_atr" ↔ shoulder offset.
                "depth_atr": float(head_depth_atr),
                "low_offset_atr": float(shoulder_offset_atr),
                "left_len": int(left_len),
                "right_len": int(right_len),
                "pattern_length": int(i5 - i1),
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
        """Cooldown dedup on the confirm bar (spec §4 -- overlapping H&S
        structures along the same neckline are the same idea)."""
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


class HeadShouldersDetector(HeadShouldersDetectorBase):
    """Head & Shoulders (P4) — bearish reversal off a higher-middle peak."""

    name = "head_shoulders"
    version = "1.0"
    short_name = "HS"
    direction = "bearish"
    kind_seq = ("H", "L", "H", "L", "H")


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> HeadShouldersDetector:
    return HeadShouldersDetector(config)


DETECTOR_CLASS = HeadShouldersDetector