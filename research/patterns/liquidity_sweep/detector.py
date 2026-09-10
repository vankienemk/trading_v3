"""
detector.py — Liquidity Sweep pattern plugin (§2.2, Agent 2)

Refactors the legacy V2 sweep detector into a ``BasePatternDetector`` plugin
while keeping the detection *logic bit-identical* (Agent 2 DoD — golden
test event-by-event equal to the legacy ``signal_engine_v2._check_new_bar``
path).

Shared infra reused from the legacy research package unchanged:
  * ``src.events.sweep_detector_v2.detect_sweeps_v2`` / ``_candidate_rows_v2``
  * ``src.events.deduplication.select_deduplicated_events``
  * ``src.events.confirmation.attach_confirmations`` (+ run helpers)
  * ``src.liquidity.level_registry.build_liquidity_levels``
  * ``src.features.feature_pipeline.build_event_features``
  * ``src.scoring.rule_score.compute_rule_scores``

The plugin composes those modules in EXACTLY the order of the legacy
``signal_engine_v2._check_new_bar`` (steps 2-7):

    detect → dedup → sort → assign event_id → confirm → levels →
    features → rule scores → entry/SL/TP (next open after confirmation,
    stop at sweep extreme ± buffer·ATR, target at reward_r·risk)

It never re-implements a heuristic, so the golden test can prove
event-by-event equality with the legacy composition on the same frame.

Feature / causality notes (§3):
  * ``reclaim_atr`` / ``level_price`` are available at detect time; the
    confirmation column features carry ``available_at=confirm``.
  * No feature reads any bar at/after ``known_at`` (the legacy chain is
    causal; ``validate_causality`` is invoked per batch and in tests).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from research.core.config_hash import (
    config_hash_for_detector,
    set_event_config_hash,
)
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
)

# ---------------------------------------------------------------------------
# Legacy research package is the shared detector infra for this plugin.
# ---------------------------------------------------------------------------
_LEGACY = Path(__file__).resolve().parent
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from src.events.confirmation import (  # noqa: E402
    attach_confirmations,
    run_anchor_position,
    run_opposite_extreme_at,
)
from src.events.deduplication import select_deduplicated_events  # noqa: E402
from src.events.sweep_detector_v2 import (  # noqa: E402
    V2_DEFAULT_TARGET_R,
    _candidate_rows_v2,
    detect_sweeps_v2,
)
from src.features.feature_pipeline import build_event_features  # noqa: E402
from src.liquidity.level_registry import build_liquidity_levels  # noqa: E402
from src.scoring.rule_score import compute_rule_scores  # noqa: E402

#: Feature-schema version stamp recorded on every event (§5.5 / §6.2).
FEATURE_SCHEMA_VERSION = "lsw-2.0"


def _bar_index(index: pd.Index[Any], ts: Any) -> int:
    """Bar position of ``ts`` in ``index``.

    pandas-stubs does not annotate ``Index.get_indexer``, so this small
    typed wrapper is the single place the no-untyped-call strict-mypy
    exception is needed (the legacy live engine does the same lookup
    inline).
    """
    loc = index.get_indexer([ts])[0]  # type: ignore[no-untyped-call]
    if loc == -1:
        raise ValueError(f"timestamp {ts!r} not present in the candles index")
    return int(loc)


class LiquiditySweepDetector(BasePatternDetector):
    """V2 Liquidity Sweep — now a plugin (spec §1.2/1, §11 P0).

    Default config mirrors the frozen live ``signal_engine_v2`` pipeline
    (``research/configs/symbols/XAUUSD.yaml`` → ``pipepline_overrides`` →
    ``v2_frozen.yaml``) so the plugin is a drop-in regression baseline:
    ``target_r=3.0``, ``buffer_atr=0.10``, cooldown 4, confirmation 3/0.60/0.80.
    """

    name = "liquidity_sweep"
    version = "2.0"  # semantic version of the detector behavior (§6.2)
    short_name = "LSW"  # §9.2 registration code — pattern_short lookup

    # Declared feature schema for runtime causal validation (§3.4).  These are
    # the features the rule-score / ML layer actually consumes.  All are
    # available by confirm time at the latest.
    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature("atr", "float", AVAILABLE_AT_DETECT, False, "ATR(14) at the sweep bar"),
        PatternFeature("penetration_atr", "float", AVAILABLE_AT_DETECT, False, "sweep penetration in ATR"),
        PatternFeature("wick_ratio", "float", AVAILABLE_AT_DETECT, False, "lower/upper wick ratio"),
        PatternFeature("reclaim_atr", "float", AVAILABLE_AT_DETECT, False, "close-back distance in ATR (v2: >= 0.0)"),
        PatternFeature("h1_trend", "int", AVAILABLE_AT_DETECT, False, "causal H1 trend (-1/0/1)"),
        PatternFeature("level_price", "float", AVAILABLE_AT_DETECT, False, "swept liquidity level"),
        PatternFeature("confirmation_delay_bars", "float", AVAILABLE_AT_CONFIRM, False, "bars sweep→confirmation candle"),
        PatternFeature("confirmation_range_atr", "float", AVAILABLE_AT_CONFIRM, False, "confirmation range / ATR"),
        PatternFeature("confirmation_strength", "float", AVAILABLE_AT_CONFIRM, False, "0..1 body+range quality"),
        PatternFeature("rule_score", "float", AVAILABLE_AT_CONFIRM, False, "weighted rule score 0..100"),
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self.get_default_config()
        if config:
            merged = dict(self.config)
            merged.update(config)
            # "version" must never be silently overridden by a partial config
            merged.setdefault("version", self.version)
            self.config = merged

    # ------------------------------------------------------------------
    # BasePatternDetector contract
    # ------------------------------------------------------------------
    def get_default_config(self) -> dict[str, Any]:
        """Default sweep config — MUST carry "version" (§6.2).

        Values equal the frozen live pipeline params of the legacy
        ``signal_engine_v2`` (``_extract_pipeline_params`` defaults +
        ``v2_frozen.yaml``): counter-trend sweeps, shallow penetration
        [0.05, 0.20], cooldown 4 / group first, confirmation
        3 bars / 0.60 body / 0.80 ATR, entry at next open after
        confirmation, stop = sweep extreme ± 0.10·ATR, target = 3.0R.
        """
        return {
            "version": self.version,
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "atr_period": 14,
            "level_lookback": 20,
            "min_penetration_atr": 0.05,
            "max_penetration_atr": 0.20,
            "min_wick_ratio": 0.35,
            "min_reclaim_atr": 0.0,
            "v2_nguoc_trend": True,
            "cooldown_bars": 4,
            "group_rule": "first",
            "confirmation": {
                "max_wait_bars": 3,
                "min_body_ratio": 0.60,
                "min_range_atr": 0.80,
                "require_break_sweep_extreme": True,
            },
            "target_r": 3.0,  # frozen v2_target_r (v2_frozen.yaml)
            "buffer_atr": 0.10,
        }

    # ------------------------------------------------------------------
    # Legacy chain — detect → dedup → event_id → confirm → features → scores
    # ------------------------------------------------------------------
    def _run_legacy_chain(
        self, df: pd.DataFrame, cfg: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
        """Compose the exact legacy ``_check_new_bar`` steps 2-6.

        Returns ``(sweep_out, scored_events, features_by_id)``:
          * ``sweep_out`` — the ``detect_sweeps_v2`` feature frame (kept for
            the entry/SL/TP anchoring step 7);
          * ``scored_events`` — deduped, event_id-stamped, confirmed events
            with the ``rule_score`` column (empty → no actionable events);
          * ``features_by_id`` — one row of the registered feature matrix per
            event (for §7 Event Lake attributes).

        The ordering mirrors the legacy engine line-for-line:
        ``_candidate_rows_v2`` → ``select_deduplicated_events`` →
        stable sort by (event_time, direction) → ``event_id`` assignment →
        ``attach_confirmations`` → filter ``is_confirmed`` →
        ``build_liquidity_levels`` → ``build_event_features`` →
        ``compute_rule_scores``.
        """
        conf = cfg.get("confirmation", {})
        sweep_out = detect_sweeps_v2(
            df,
            atr_period=int(cfg["atr_period"]),
            level_lookback=int(cfg["level_lookback"]),
            min_penetration_atr=float(cfg["min_penetration_atr"]),
            max_penetration_atr=float(cfg["max_penetration_atr"]),
            min_wick_ratio=float(cfg["min_wick_ratio"]),
            min_reclaim_atr=float(cfg["min_reclaim_atr"]),
            v2_nguoc_trend=bool(cfg["v2_nguoc_trend"]),
        )
        candidates = _candidate_rows_v2(sweep_out)
        if candidates.empty:
            return sweep_out, pd.DataFrame(), {}

        deduped = select_deduplicated_events(
            candidates,
            cooldown_bars=int(cfg["cooldown_bars"]),
            group_rule=str(cfg["group_rule"]),
        )
        if deduped.empty:
            return sweep_out, pd.DataFrame(), {}
        deduped = deduped.sort_values(["event_time", "direction"], kind="stable")
        symbol = str(cfg.get("symbol", "XAUUSD"))
        deduped["event_id"] = [f"{symbol}-V2-{i:06d}" for i in range(len(deduped))]
        deduped["v2_target_r"] = float(cfg.get("target_r", V2_DEFAULT_TARGET_R))

        confirmed_df = attach_confirmations(
            df,
            deduped,
            {},
            max_wait_bars=int(conf.get("max_wait_bars", 3)),
            min_body_ratio=float(conf.get("min_body_ratio", 0.60)),
            min_range_atr=float(conf.get("min_range_atr", 0.80)),
            require_break_sweep_extreme=bool(conf.get("require_break_sweep_extreme", True)),
        )
        confirmed = confirmed_df[confirmed_df["is_confirmed"] == True].copy()  # noqa: E712
        if confirmed.empty:
            return sweep_out, pd.DataFrame(), {}

        levels = build_liquidity_levels(df, {})
        features_df = build_event_features(df, levels, confirmed, {})
        if features_df.empty:
            return sweep_out, pd.DataFrame(), {}

        scored = compute_rule_scores(confirmed, {}, levels)
        rule_scores = dict(zip(scored["event_id"], scored["rule_score"]))
        confirmed["rule_score"] = [float(rule_scores.get(eid, 0.0)) for eid in confirmed["event_id"]]

        features_by_id: dict[str, dict[str, Any]] = {}
        for _, feat_row in features_df.iterrows():
            feats = {
                col: (
                    None
                    if isinstance(val, float) and not np.isfinite(val)
                    else val
                )
                for col, val in feat_row.drop(["event_id", "event_time"]).to_dict().items()
            }
            features_by_id[str(feat_row["event_id"])] = feats
        return sweep_out, confirmed, features_by_id

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        """Run the V2 sweep + confirmation flow and return PatternEvents.

        Mirrors the legacy ``signal_engine_v2._check_new_bar`` detection
        section (detect → dedup → event_id → confirm → features → rule
        scores → entry/SL/TP) so outputs are bit-identical.
        """
        cfg = self.config
        if config:
            cfg = {**self.config, **config, "version": self.version}
        sweep_out, confirmed, features_by_id = self._run_legacy_chain(df, cfg)
        if confirmed.empty:
            return []

        symbol = str(cfg.get("symbol", "XAUUSD"))
        timeframe = str(cfg.get("timeframe", "M15"))
        target_r = float(cfg.get("target_r", V2_DEFAULT_TARGET_R))
        buffer_atr = float(cfg.get("buffer_atr", 0.10))
        atr_series = sweep_out["atr"]

        rule_scores = {
            eid: float(rs)
            for eid, rs in zip(confirmed["event_id"], confirmed["rule_score"])
        }

        events: list[PatternEvent] = []
        for _, ev in confirmed.iterrows():
            eid = str(ev["event_id"])
            direction = (
                "bullish"
                if str(ev.get("direction", "bullish")).lower() in ("bullish", "long")
                else "bearish"
            )
            is_long = direction == "bullish"
            bar_pos = _bar_index(df.index, ev["event_time"])
            anchor = run_anchor_position(sweep_out, bar_pos, str(ev["direction"]))
            conf_time = pd.Timestamp(ev["confirmation_time"])
            conf_bar = _bar_index(df.index, conf_time)
            entry_bar = conf_bar + 1
            if entry_bar >= len(df):
                continue  # legacy step 7: no executable entry on this frame
            entry_price = float(df["open"].iloc[entry_bar])
            atr_sweep = float(atr_series.iloc[anchor])
            extreme = run_opposite_extreme_at(df, anchor, anchor, str(ev["direction"]))
            if is_long:
                stop_price = extreme - buffer_atr * atr_sweep
                target_price = entry_price + target_r * (entry_price - stop_price)
            else:
                stop_price = extreme + buffer_atr * atr_sweep
                target_price = entry_price - target_r * (stop_price - entry_price)

            detect_time = pd.Timestamp(ev["event_time"])
            attributes: dict[str, Any] = {
                "position": int(ev.get("position", -1)),
                "level_id": str(ev.get("level_id", "")),
                "level_price": float(ev.get("level_price", 0.0)),
                "event_open": float(ev.get("event_open", float("nan"))),
                "event_high": float(ev.get("event_high", float("nan"))),
                "event_low": float(ev.get("event_low", float("nan"))),
                "event_close": float(ev.get("event_close", float("nan"))),
                "penetration_atr": float(ev.get("penetration_atr", 0.0)),
                "wick_ratio": float(ev.get("wick_ratio", 0.0)),
                "reclaim_atr": float(ev.get("reclaim_atr", 0.0)),
                "h1_trend": int(ev.get("h1_trend", 0)),
                "v2_target_r": float(ev.get("v2_target_r", target_r)),
                "atr_value": round(atr_sweep, 5),
                "confirmation_type": str(ev.get("confirmation_type", "")),
                "confirmation_delay_bars": float(ev.get("confirmation_delay_bars", float("nan"))),
                "confirmation_body_ratio": float(ev.get("confirmation_body_ratio", float("nan"))),
                "confirmation_range_atr": float(ev.get("confirmation_range_atr", float("nan"))),
                "confirmation_strength": float(ev.get("confirmation_strength", float("nan"))),
                "sweep_anchor_bar": int(anchor),
                "features": dict(features_by_id.get(eid, {})),
            }
            events.append(
                PatternEvent(
                    event_id=eid,
                    pattern_name=self.name,
                    pattern_version=self.version,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    detect_time=detect_time,
                    confirm_time=conf_time,
                    entry_time=pd.Timestamp(df.index[entry_bar]),
                    entry_price=round(entry_price, 5),
                    stop_price=round(stop_price, 5),
                    target_price=round(target_price, 5),
                    structure_levels={
                        "level_price": float(ev.get("level_price", float("nan"))),
                        "sweep_extreme": float(extreme),
                    },
                    rule_score=round(float(rule_scores.get(eid, 0.0)), 2),
                    model_prob=None,
                    attributes=attributes,
                )
            )
        # §6.2 lineage auditing: stamp the resolved config hash + schema version.
        set_event_config_hash(
            events,
            config_hash_for_detector(self, cfg),
            FEATURE_SCHEMA_VERSION,
        )
        return events

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (used by live engine + golden test)
    # ------------------------------------------------------------------
    def detect_frame(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
        """Return the confirmed+scored event DataFrame (legacy shape).

        Applies the same legacy chain as :meth:`detect` but stops before the
        entry/SL/TP projection, exposing the exact suspicion table the golden
        test and live consumers compare against (columns: the candidate
        schema + confirmation columns + ``rule_score``).
        """
        cfg = self.config
        if config:
            cfg = {**self.config, **config, "version": self.version}
        _, confirmed, _ = self._run_legacy_chain(df, cfg)
        return confirmed


# ---------------------------------------------------------------------------
# Convenience entry points the pattern registry / live engine import (§2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> LiquiditySweepDetector:
    return LiquiditySweepDetector(config)


DETECTOR_CLASS = LiquiditySweepDetector

#: Registration record for ``live/engine/pattern_registry.py`` (§2.2, §9.2):
#: key = pattern name, value = detector class; ``short_name``/``version`` are
#: consumed from the class attributes.
PATTERN_ENTRY: dict[str, type[LiquiditySweepDetector]] = {
    LiquiditySweepDetector.name: LiquiditySweepDetector
}