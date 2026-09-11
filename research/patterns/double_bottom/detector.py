"""
detector.py -- Double Bottom / Double Top plugin (P1/P2, Agent 3).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P1 double_bottom, P2 double_top) and
§13 Agent 3.  Implements the classical double-reversal structure on the
shared :class:`research.core.swing_detector.SwingDetector` with the §3.2
Right-Bar Rule (a pivot only exists causally at ``bar + right_bars``).

Geometry (bullish double bottom; the bearish double top is the exact
mirror -- see patterns/double_top/PATTERN_SPECS.md):

* two swing LOWS at approximately the same level (``max_equal_atr`` ATRs
  apart), separated by a swing HIGH (the neckline touch, ``min_separation_bars``
  away on each side) and a genuine depth (``min_depth_atr`` ATRs);
* confirmation: after the SECOND low's pivot is causally known
  (``extreme2_bar + right_bars``), a close crosses the neckline (the middle
  swing price); the effective confirm bar is the later of that close-cross
  and the pivot-known bar;
* entry: open of the next bar after confirmation; stop: below the lower
  extreme minus ATR buffer; target: ``target_r`` R-multiples.

Causality (§3): every scoring feature is available at ``detect`` (data up to
the pivot-known bar) or ``confirm`` (the confirm bar itself) -- never past
``known_at``.  ``detect()`` stamps each event with its ``config_hash``
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

#: version of the double-pattern feature schema (bump on semantic change)
FEATURE_SCHEMA_VERSION = "double-v1.0"


def atr_series(df: pd.DataFrame, period: int) -> np.ndarray[Any, Any]:
    """Causal ATR: rolling mean of true range, value at bar ``b`` uses only
    bars ``<= b`` (NaN before ``period`` bars)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    rolling = pd.Series(tr).rolling(int(period), min_periods=int(period)).mean()
    return np.asarray(rolling.to_numpy(), dtype=float)


class DoublePatternDetectorBase(BasePatternDetector):
    """Shared machinery for double_bottom (bullish) and double_top (bearish).

    Subclasses set ``name`` / ``short_name`` / ``direction`` / ``kind_seq`` --
    everything else (swing scan, gates, confirmation, entry/SL/TP, features,
    config, causality stamps) is shared (spec §11: double_top = mirror of
    double_bottom with "100 % infra của P1").
    """

    name: str = "double_pattern"
    version: str = "1.0"
    short_name: str = "DP"
    #: canonical direction of the breakout (bullish bottom / bearish top)
    direction: str = DIRECTION_BULLISH
    #: swing kind sequence: (L,H,L) bottom, (H,L,H) top
    kind_seq: tuple[str, ...] = ("L", "H", "L")

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(
            "atr", "float", AVAILABLE_AT_DETECT, False,
            "ATR(period) at the pivot-known (detect) bar",
        ),
        PatternFeature(
            "depth_atr", "float", AVAILABLE_AT_DETECT, False,
            "neckline depth in ATR -- middle swing vs higher of the two extremes",
        ),
        PatternFeature(
            "low_offset_atr", "float", AVAILABLE_AT_DETECT, False,
            "offset between the two extremes in ATR (equal-level fidelity)",
        ),
        PatternFeature(
            "symmetry_ratio", "float", AVAILABLE_AT_DETECT, False,
            "left/right arm length symmetry in [0,1]",
        ),
        PatternFeature(
            "pattern_length", "int", AVAILABLE_AT_DETECT, False,
            "bars between the two extremes",
        ),
        PatternFeature(
            "confirm_reclaim_atr", "float", AVAILABLE_AT_CONFIRM, False,
            "confirm close past the neckline in ATR",
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
        """Default double-pattern config -- MUST carry "version" (§6.2)."""
        return {
            "version": self.version,
            "left_bars": 3,
            "right_bars": 3,
            "atr_period": 14,
            "min_separation_bars": 4,
            # Equal-level tolerance / depth gate calibrated on the full
            # XAUUSD M15 2018-2026 history: 1.5 ATR low offset + 3.0 ATR
            # minimum neckline depth keep §8.3 noise FP ≤ 5 % while clearing
            # the §6.3 sample-size gate (≥300 total / ≥100 OOS events).
            "max_equal_atr": 1.5,
            "min_depth_atr": 3.0,
            "stop_buffer_atr": 0.5,
            "target_r": 1.5,
            # §2.3.2 rework -- reward sizing.
            #
            # Spec §2.1 complained the TP is drawn 3-4x the pattern depth and
            # "quá xa".  Measured on XAUUSD M15 that complaint is real: the
            # legacy target sits 6.7 ATR from entry.  What the spec's §2.3.2
            # fix (cap at 4 ATR) misses is that the STOP is 4.56 ATR wide, so
            # a 4 ATR cap forces realized R:R < 1.0 for ~62% of events --
            # cap+min_rr=1.0 together drop 96% of candidates (DB 354->15).
            #
            # Options are provided for A/B, with the MEASURED best as default.
            #   "legacy"  -> target = target_r * risk        (best expectancy)
            #   "atr"     -> target = structure_target_atr * ATR, capped
            # See docs/pattern_rework_resolution.md for the sweep numbers.
            "structure_target_atr": 0.0,
            "target_cap_atr": 0.0,
            # §2.3.1 rework: stop anchor.
            #   "legacy"   -> beyond both extremes (full pattern depth)
            #   "neckline" -> a close back through the breakout level
            # MEASURED: "neckline" is 5.2x tighter (0.87 vs 4.56 ATR) and is
            # strictly WORSE -- winrate 0.48 -> 0.23, expectancy -0.01R ->
            # -0.47R.  The wide stop is not the bug; it is what makes the
            # modest post-breakout excursion survivable.  Kept for A/B only.
            "stop_mode": "legacy",
            # §2.3.1 rework: pre-cap floor on R:R.  A candidate whose reward is
            # closer than this multiple of risk is DROPPED rather than emitted
            # with a sub-1R payoff.  0.0 disables the gate.
            "min_rr": 1.0,
            # §1.3.1 rework: non-max suppression over the structure window.
            "nms_overlap": True,
            # §3.3 staleness: confirm must fire within this many bars after
            # the setup is complete (last pivot known), else discard.
            "max_bars_between_detect_and_confirm": 60,
            "cooldown_bars": 3,
            "symbol": "XAUUSD",
            "timeframe": "M15",
        }

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        cfg = self.config
        if config:
            cfg = {**self.config, **config, "version": self.version}
        return self._detect_double(df, cfg)

    # ------------------------------------------------------------------
    # Detection core (both directions)
    # ------------------------------------------------------------------
    def _detect_double(self, df: pd.DataFrame, cfg: dict[str, Any]) -> list[PatternEvent]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("double-pattern detector requires a DatetimeIndex")
        if len(df) < 8:
            return []

        bullish = self.direction == DIRECTION_BULLISH
        left = int(cfg["left_bars"])
        right = int(cfg["right_bars"])
        min_sep = int(cfg["min_separation_bars"])
        max_equal_atr = float(cfg["max_equal_atr"])
        min_depth_atr = float(cfg["min_depth_atr"])
        stop_buffer_atr = float(cfg["stop_buffer_atr"])
        target_r = float(cfg["target_r"])
        structure_target_atr = float(cfg.get("structure_target_atr", 0.0) or 0.0)
        target_cap_atr = float(cfg.get("target_cap_atr", 0.0) or 0.0)
        stop_mode = str(cfg.get("stop_mode", "legacy"))
        min_rr = float(cfg.get("min_rr", 0.0) or 0.0)
        nms_overlap = bool(cfg.get("nms_overlap", True))
        max_wait = int(cfg["max_bars_between_detect_and_confirm"])
        cooldown = int(cfg["cooldown_bars"])
        symbol = str(cfg.get("symbol", "XAUUSD"))
        timeframe = str(cfg.get("timeframe", "M15"))

        sw = SwingDetector(left, right).find_swings(df)
        if len(sw) < 3:
            return []
        atr = atr_series(df, int(cfg["atr_period"]))
        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        n = len(df)

        want = self.kind_seq
        config_hash = compute_config_hash(cfg)
        candidates: list[PatternEvent] = []

        for k in range(len(sw) - 2):
            s1, s2, s3 = sw[k], sw[k + 1], sw[k + 2]
            if (s1.kind, s2.kind, s3.kind) != want:
                continue
            i1, v1 = s1.bar, s1.price
            i2, v2 = s2.bar, s2.price
            i3, v3 = s3.bar, s3.price
            if not (i1 < i2 < i3):
                continue
            if i2 - i1 < min_sep or i3 - i2 < min_sep:
                continue
            detect_bar = s3.known_at_bar
            if detect_bar >= n or detect_bar >= len(atr) or np.isnan(atr[detect_bar]):
                continue
            atr_k = float(atr[detect_bar])
            if atr_k <= 0.0:
                continue

            if bullish:
                # L-H-L: lows below the middle high; equal lows; real depth
                if not (v1 <= v2 and v3 <= v2):
                    continue
                if abs(v1 - v3) > max_equal_atr * atr_k:
                    continue
                depth = v2 - max(v1, v3)
            else:
                # H-L-H: highs above the middle low; equal highs; real depth
                if not (v1 >= v2 and v3 >= v2):
                    continue
                if abs(v1 - v3) > max_equal_atr * atr_k:
                    continue
                depth = min(v1, v3) - v2
            if depth < min_depth_atr * atr_k:
                continue

            # Confirmation: close crosses the neckline (middle swing price)
            # after the second extreme; the pattern is not complete until the
            # second extreme's pivot is causally known (§3.2).
            neck = v2
            cross: int | None = None
            scan_end = min(n, i3 + 1 + max_wait)
            for j in range(i3 + 1, scan_end):
                if bullish and closes[j] > neck:
                    cross = j
                    break
                if not bullish and closes[j] < neck:
                    cross = j
                    break
            if cross is None:
                continue
            confirm_bar = max(cross, detect_bar)
            entry_bar = confirm_bar + 1 if confirm_bar + 1 < n else confirm_bar

            entry_price = (
                float(opens[entry_bar]) if entry_bar != confirm_bar else float(closes[confirm_bar])
            )
            # §2.3.1 rework -- STRUCTURE-based stop.
            #
            # Legacy stop = beyond BOTH extremes, i.e. the full pattern depth
            # (median 4.56 ATR because the depth gate alone is 3 ATR).  But
            # the entry fires on the neckline breakout, only ~0.39 ATR from
            # the neckline: the "other" extreme is already far behind and
            # contributes almost nothing to invalidation while quadrupling the
            # risk.  That oversized R is the real source of the "TP quá xa"
            # symptom -- with risk this wide, ANY target with R:R >= 1 must be
            # 4.5-11 ATR away.
            #
            # So anchor the stop on the invalidation that actually matters: a
            # close back through the neckline (plus buffer).  `stop_mode`
            # selects it; "legacy" keeps the old behaviour for A/B.
            if stop_mode == "neckline":
                legacy_stop = (
                    min(v1, v3) - stop_buffer_atr * atr_k
                    if bullish
                    else max(v1, v3) + stop_buffer_atr * atr_k
                )
                if bullish:
                    stop_price = neck - stop_buffer_atr * atr_k
                    # Tightening only, and never at/above entry.  The entry
                    # fires on a close THROUGH the neckline, so price can
                    # close well past it and leave `neck` above `entry`;
                    # an uncapped neckline stop would then sit above the
                    # entry of a long and invert the trade.
                    stop_price = max(stop_price, legacy_stop)
                    if stop_price >= entry_price:
                        stop_price = legacy_stop
                else:
                    stop_price = neck + stop_buffer_atr * atr_k
                    stop_price = min(stop_price, legacy_stop)
                    if stop_price <= entry_price:
                        stop_price = legacy_stop
            elif bullish:
                stop_price = min(v1, v3) - stop_buffer_atr * atr_k
            else:
                stop_price = max(v1, v3) + stop_buffer_atr * atr_k
            risk = abs(entry_price - stop_price)
            if risk <= 0.0:
                continue
            # §2.3.2 rework -- size the reward by STRUCTURE, not by R.
            #
            # `risk` for this family is the whole pattern depth plus buffer
            # (median 4.45 ATR, since the depth gate alone is 3 ATR).  The
            # legacy `target_r * risk` therefore lands 6-10 ATR from entry,
            # past where M15 price action goes after the breakout.  An ATR cap
            # alone cannot repair that: a cap reachable on M15 (4 ATR) forces
            # realized R:R < 1.0 for ~62% of events -- measured 96% event loss
            # (DB 354->15) when combined with min_rr=1.0.
            #
            # So: place the target a fixed, reachable distance from entry
            # (`structure_target_atr`, mirroring how far the post-breakout leg
            # actually travels), keep the legacy R-multiple only as the
            # fallback when that option is disabled, and apply the ATR cap as
            # a final ceiling in both cases.
            if structure_target_atr > 0.0:
                target_distance = structure_target_atr * atr_k
                capped = False
            else:
                target_distance = target_r * risk
                capped = False
            if target_cap_atr > 0.0:
                cap_distance = target_cap_atr * atr_k
                if target_distance > cap_distance:
                    target_distance = cap_distance
                    capped = True
            realized_rr = target_distance / risk
            # A target below min_rr is a structurally bad trade: the stop sits
            # further than the reward can travel.  Drop it rather than emit a
            # sub-1R candidate (fail-closed, §6.3 spirit).
            if min_rr > 0.0 and realized_rr < min_rr:
                continue
            if bullish:
                target_price = entry_price + target_distance
            else:
                target_price = entry_price - target_distance

            depth_atr = depth / atr_k
            low_offset_atr = abs(v1 - v3) / atr_k
            left_len = i2 - i1
            right_len = i3 - i2
            symmetry = 1.0 - min(abs(left_len - right_len) / max(left_len, right_len), 1.0)
            confirm_reclaim = (
                closes[confirm_bar] - neck if bullish else neck - closes[confirm_bar]
            ) / atr_k
            # F1 (handoff §3 P1#5): confirm_range_atr was computed but never
            # assigned (dead expression); emit it so every feature_schema
            # entry is populated at its declared available_at (confirm).
            # Normalize by ATR at the confirm bar itself (matches the gate
            # scorer's rng_det convention and the trainer's confirm feature).
            atr_confirm = float(atr[confirm_bar]) if not np.isnan(atr[confirm_bar]) else atr_k
            confirm_range_atr = (highs[confirm_bar] - lows[confirm_bar]) / max(atr_confirm, 1e-12)

            rule_score = min(
                1.0,
                0.40 * min(depth_atr / min_depth_atr, 1.0)
                + 0.25 * symmetry
                + 0.20 * (1.0 - min(low_offset_atr / max_equal_atr, 1.0))
                + 0.15 * min(confirm_reclaim / 1.0, 1.0),
            )

            level_key = "double_bottom_level" if bullish else "double_top_level"
            structure_levels: dict[str, float] = {
                "neckline": neck,
                level_key: min(v1, v3) if bullish else max(v1, v3),
                "extreme1_level": v1,
                "extreme2_level": v3,
            }
            attributes: dict[str, Any] = {
                "extreme1_bar": i1,
                "neckline_bar": i2,
                "extreme2_bar": i3,
                "pivot_known_at_bar": detect_bar,
                "close_cross_bar": int(cross),
                "confirm_bar": int(confirm_bar),
                "entry_bar": int(entry_bar),
                "depth_atr": float(depth_atr),
                "low_offset_atr": float(low_offset_atr),
                "confirm_range_atr": float(confirm_range_atr),
                "left_len": left_len,
                "right_len": right_len,
                "target_capped": bool(capped),
                "realized_rr": float(realized_rr),
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

        if nms_overlap:
            candidates = self._nms_structure_overlap(candidates)
        return self._dedupe(candidates, cooldown)

    @staticmethod
    def _nms_structure_overlap(events: list[PatternEvent]) -> list[PatternEvent]:
        """§1.3.1 rework -- non-max suppression over the structure window.

        The scan emits an event for EVERY consecutive triple of swings
        matching ``kind_seq``, so one long structure can be reported many
        times with slightly shifted pivots.  ``_dedupe`` only removes events
        whose CONFIRM bars are within ``cooldown_bars``, which cannot see two
        readings of the same structure that confirm far apart.

        Two candidates are considered the same idea when their
        ``[extreme1_bar, extreme2_bar]`` windows intersect.  The
        higher-``rule_score`` event wins; ties break on the earlier detect
        bar so the outcome stays deterministic.
        """
        if len(events) < 2:
            return list(events)

        def score(ev: PatternEvent) -> tuple[float, int]:
            return (-float(ev.rule_score or 0.0), int(ev.attributes["extreme1_bar"]))

        def confirm(ev: PatternEvent) -> int:
            # Not every caller populates confirm_bar; fall back to the last
            # structure anchor so ordering stays total and never raises.
            bar = ev.attributes.get("confirm_bar")
            if bar is None:
                bar = ev.attributes.get("extreme2_bar", 0)
            return int(bar)

        kept: list[PatternEvent] = []
        occupied: list[tuple[int, int]] = []
        for ev in sorted(events, key=score):
            i1 = int(ev.attributes["extreme1_bar"])
            i3 = int(ev.attributes["extreme2_bar"])
            if any(not (i3 < a1 or i1 > a3) for (a1, a3) in occupied):
                continue
            kept.append(ev)
            occupied.append((i1, i3))
        kept.sort(key=confirm)
        return kept

    @staticmethod
    def _dedupe(events: list[PatternEvent], cooldown: int) -> list[PatternEvent]:
        """Cooldown dedup on the confirm bar: keep the highest-scoring event
        and skip events confirming within ``cooldown`` bars of a kept one
        (spec §4 -- overlapping double structures are the same idea)."""
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


class DoubleBottomDetector(DoublePatternDetectorBase):
    """Double Bottom (P1) -- bullish reversal off two equal swing lows."""

    name = "double_bottom"
    version = "1.0"
    short_name = "DB"
    direction = DIRECTION_BULLISH
    kind_seq = ("L", "H", "L")


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> DoubleBottomDetector:
    return DoubleBottomDetector(config)


DETECTOR_CLASS = DoubleBottomDetector