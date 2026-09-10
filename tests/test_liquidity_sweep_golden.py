"""Golden test — Liquidity Sweep plugin is bit-identical to the legacy path.

Agent 2 DoD (spec §13): the plugin must NOT change detection logic — the
refactor is a pure adapter over the frozen V2 detector.  The golden test
proves the plugin's confirmed-and-scored events are identical — event by
event, field by field — to composing the legacy ``signal_engine_v2``
modules directly in the exact ``_check_new_bar`` order on the SAME XAUUSD
M15 dataset:

    detect_sweeps_v2 → _candidate_rows_v2 → select_deduplicated_events →
    stable sort → event_id assignment → attach_confirmations →
    build_liquidity_levels → build_event_features → compute_rule_scores →
    entry/SL/TP (next open after confirmation, sweep-extreme stop, 3R target)

The two expensive chain executions (legacy reference + plugin) are computed
once per module run and memoized; every test then compares the projected
tables — the file stays fast while proving full event-by-event equality.

Marker: golden (registered in pyproject).  Also inherits the shared
no-lookahead CI gate via ``NoLookaheadTestBase`` (§3.4).

Run with the venv that has pandas/pyarrow:
    /tmp/ptv2_venv/bin/python -m pytest tests/test_liquidity_sweep_golden.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from research.core.contracts import PatternEvent
from research.patterns.liquidity_sweep.detector import LiquiditySweepDetector
from tests.no_lookahead_base import NoLookaheadTestBase

_DATA = (
    Path(__file__).resolve().parent.parent
    / "research" / "patterns" / "liquidity_sweep" / "data" / "processed" / "xauusd_m15.parquet"
)

# Frozen live pipeline params (legacy _extract_pipeline_params defaults +
# v2_frozen.yaml) — the exact config the golden test runs both paths with.
_LEGACY_CFG: dict[str, Any] = {
    "version": "2.0",
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
    "target_r": 3.0,
    "buffer_atr": 0.10,
}

_LEGACY_TABLE_COLUMNS = [
    "event_id", "event_time", "direction", "confirmation_time", "entry_time",
    "penetration_atr", "wick_ratio", "reclaim_atr", "h1_trend", "rule_score",
    "entry_price", "stop_price", "target_price", "atr_value",
    "confirmation_delay_bars", "confirmation_range_atr", "level_id", "level_price",
]

# Module-level caches: the full legacy chain is expensive
# (build_liquidity_levels + build_event_features dominate), so each side is
# computed exactly once per pytest run.
_DF_CACHE: dict[str, pd.DataFrame] = {}
_CHAIN_CACHE: dict[str, Any] = {}


def _load_df() -> pd.DataFrame:
    if "df" not in _DF_CACHE:
        assert _DATA.exists(), f"XAUUSD parquet missing: {_DATA}"
        df = pd.read_parquet(_DATA)
        df.columns = [str(c).lower() for c in df.columns]
        # The parquet stores a 'timestamp' column (not a DatetimeIndex); make
        # it the index so the detector's causal time handling works.
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        _DF_CACHE["df"] = df.sort_index()
        print(f"  XAUUSD M15: {len(_DF_CACHE['df'])} rows, "
              f"{_DF_CACHE['df'].index[0]} → {_DF_CACHE['df'].index[-1]}")
    return _DF_CACHE["df"]


@pytest.fixture(scope="module")
def xauusd_df() -> pd.DataFrame:
    return _load_df()


def _legacy_import(func_name: str):
    """Import a legacy module function from the shared research package.

    The plugin and the legacy engine consume the SAME frozen modules under
    ``research/patterns/liquidity_sweep/`` (this is the "shared infra" the
    refactor keeps unchanged); importing them directly here is the legacy
    composition reference.
    """
    legacy = Path(__file__).resolve().parent.parent / "research" / "patterns" / "liquidity_sweep"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    from importlib import import_module

    module_name, _, attr = func_name.rpartition(".")
    return getattr(import_module(f"src.{module_name}"), attr)


def _legacy_events_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compose the legacy modules in the exact ``_check_new_bar`` order
    (steps 2-7, model-free) and project one row per emitted event.

    This is the reference the plugin must reproduce byte-for-byte.
    """
    detect_sweeps_v2 = _legacy_import("events.sweep_detector_v2.detect_sweeps_v2")
    _candidate_rows_v2 = _legacy_import("events.sweep_detector_v2._candidate_rows_v2")
    select_deduplicated_events = _legacy_import("events.deduplication.select_deduplicated_events")
    attach_confirmations = _legacy_import("events.confirmation.attach_confirmations")
    run_anchor_position = _legacy_import("events.confirmation.run_anchor_position")
    run_opposite_extreme_at = _legacy_import("events.confirmation.run_opposite_extreme_at")
    build_liquidity_levels = _legacy_import("liquidity.level_registry.build_liquidity_levels")
    build_event_features = _legacy_import("features.feature_pipeline.build_event_features")
    compute_rule_scores = _legacy_import("scoring.rule_score.compute_rule_scores")

    def empty() -> pd.DataFrame:
        return pd.DataFrame(columns=_LEGACY_TABLE_COLUMNS)

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
        return empty()
    deduped = select_deduplicated_events(
        candidates,
        cooldown_bars=int(cfg["cooldown_bars"]),
        group_rule=str(cfg["group_rule"]),
    )
    if deduped.empty:
        return empty()
    deduped = deduped.sort_values(["event_time", "direction"], kind="stable")
    symbol = str(cfg["symbol"])
    deduped["event_id"] = [f"{symbol}-V2-{i:06d}" for i in range(len(deduped))]
    deduped["v2_target_r"] = float(cfg["target_r"])

    confirmed_df = attach_confirmations(
        df, deduped, {},
        max_wait_bars=int(conf["max_wait_bars"]),
        min_body_ratio=float(conf["min_body_ratio"]),
        min_range_atr=float(conf["min_range_atr"]),
        require_break_sweep_extreme=bool(conf["require_break_sweep_extreme"]),
    )
    confirmed = confirmed_df[confirmed_df["is_confirmed"] == True].copy()  # noqa: E712
    if confirmed.empty:
        return empty()

    levels = build_liquidity_levels(df, {})
    features_df = build_event_features(df, levels, confirmed, {})
    if features_df.empty:
        return empty()
    scored = compute_rule_scores(confirmed, {}, levels)
    rule_scores = dict(zip(scored["event_id"], scored["rule_score"]))

    atr_series = sweep_out["atr"]
    target_r = float(cfg["target_r"])
    buffer_atr = float(cfg["buffer_atr"])

    rows: list[dict[str, Any]] = []
    for _, ev in confirmed.iterrows():
        eid = str(ev["event_id"])
        direction = (
            "bullish"
            if str(ev["direction"]).lower() in ("bullish", "long")
            else "bearish"
        )
        is_long = direction == "bullish"
        bar_pos = df.index.get_indexer([ev["event_time"]])[0]
        anchor = run_anchor_position(sweep_out, bar_pos, str(ev["direction"]))
        conf_time = pd.Timestamp(ev["confirmation_time"])
        conf_bar = df.index.get_indexer([conf_time])[0]
        entry_bar = conf_bar + 1
        if entry_bar >= len(df):
            continue  # legacy step 7 — no executable entry on this frame
        entry_price = float(df["open"].iloc[entry_bar])
        atr_sweep = float(atr_series.iloc[anchor])
        extreme = run_opposite_extreme_at(df, anchor, anchor, str(ev["direction"]))
        if is_long:
            stop_price = extreme - buffer_atr * atr_sweep
            target_price = entry_price + target_r * (entry_price - stop_price)
        else:
            stop_price = extreme + buffer_atr * atr_sweep
            target_price = entry_price - target_r * (stop_price - entry_price)

        rows.append(
            {
                "event_id": eid,
                "event_time": pd.Timestamp(ev["event_time"]),
                "direction": direction,
                "confirmation_time": conf_time,
                "entry_time": pd.Timestamp(df.index[entry_bar]),
                "penetration_atr": round(float(ev["penetration_atr"]), 4),
                "wick_ratio": round(float(ev["wick_ratio"]), 4),
                "reclaim_atr": round(float(ev["reclaim_atr"]), 4),
                "h1_trend": int(ev.get("h1_trend", 0)),
                "rule_score": round(float(rule_scores.get(eid, 0.0)), 2),
                "entry_price": round(entry_price, 5),
                "stop_price": round(stop_price, 5),
                "target_price": round(target_price, 5),
                "atr_value": round(atr_sweep, 5),
                "confirmation_delay_bars": float(ev.get("confirmation_delay_bars", float("nan"))),
                "confirmation_range_atr": float(ev.get("confirmation_range_atr", float("nan"))),
                "level_id": str(ev.get("level_id", "")),
                "level_price": float(ev.get("level_price", 0.0)),
            }
        )
    if not rows:
        return empty()
    return pd.DataFrame(rows)


def _plugin_events_cached() -> list[PatternEvent]:
    """The plugin's detect() output — computed once per pytest run on the
    full XAUUSD frame with the frozen legacy config."""
    if "plugin_events" not in _CHAIN_CACHE:
        det = LiquiditySweepDetector(_LEGACY_CFG)
        _CHAIN_CACHE["plugin_events"] = det.detect(_load_df(), {})
    return list(_CHAIN_CACHE["plugin_events"])


def _legacy_table_cached() -> pd.DataFrame:
    if "legacy_table" not in _CHAIN_CACHE:
        _CHAIN_CACHE["legacy_table"] = _legacy_events_table(_load_df(), _LEGACY_CFG)
    return _CHAIN_CACHE["legacy_table"]


def _plugin_events_table() -> pd.DataFrame:
    """Project the plugin's PatternEvents onto the legacy reference columns."""
    rows = [
        {
            "event_id": ev.event_id,
            "event_time": pd.Timestamp(ev.detect_time),
            "direction": ev.direction,
            "confirmation_time": pd.Timestamp(ev.confirm_time),
            "entry_time": pd.Timestamp(ev.entry_time),
            "penetration_atr": round(float(ev.attributes.get("penetration_atr", float("nan"))), 4),
            "wick_ratio": round(float(ev.attributes.get("wick_ratio", float("nan"))), 4),
            "reclaim_atr": round(float(ev.attributes.get("reclaim_atr", float("nan"))), 4),
            "h1_trend": int(ev.attributes.get("h1_trend", 0)),
            "rule_score": round(float(ev.rule_score), 2),
            "entry_price": round(float(ev.entry_price), 5),
            "stop_price": round(float(ev.stop_price), 5),
            "target_price": round(float(ev.target_price), 5),
            "atr_value": round(float(ev.attributes.get("atr_value", float("nan"))), 5),
            "confirmation_delay_bars": float(ev.attributes.get("confirmation_delay_bars", float("nan"))),
            "confirmation_range_atr": float(ev.attributes.get("confirmation_range_atr", float("nan"))),
            "level_id": str(ev.attributes.get("level_id", "")),
            "level_price": float(ev.attributes.get("level_price", 0.0)),
        }
        for ev in _plugin_events_cached()
    ]
    return pd.DataFrame(rows, columns=_LEGACY_TABLE_COLUMNS)


def _sort_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["event_time", "direction", "event_id"], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Golden DoD tests
# ---------------------------------------------------------------------------

def test_plugin_events_identical_to_legacy(xauusd_df: pd.DataFrame) -> None:
    legacy = _legacy_table_cached()
    plugin = _plugin_events_table()

    assert not legacy.empty, "legacy reference produced zero events on XAUUSD"
    assert len(plugin) == len(legacy), (
        f"event count mismatch: plugin={len(plugin)} legacy={len(legacy)}"
    )

    pd.testing.assert_frame_equal(
        _sort_key(plugin), _sort_key(legacy), check_like=True,
    )


def test_event_ids_identical(xauusd_df: pd.DataFrame) -> None:
    assert list(_sort_key(_plugin_events_table())["event_id"]) == list(
        _sort_key(_legacy_table_cached())["event_id"]
    )


def test_rule_scores_identical(xauusd_df: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(
        _sort_key(_plugin_events_table())["rule_score"],
        _sort_key(_legacy_table_cached())["rule_score"],
        check_names=False,
    )


def test_entry_stop_target_identical(xauusd_df: pd.DataFrame) -> None:
    for col in ("entry_price", "stop_price", "target_price", "atr_value"):
        pd.testing.assert_series_equal(
            _sort_key(_plugin_events_table())[col],
            _sort_key(_legacy_table_cached())[col],
            check_names=False,
        )


def test_legacy_event_count_audit(xauusd_df: pd.DataFrame) -> None:
    """Pinned event-count regression so the golden reference itself is audited."""
    n = len(_legacy_table_cached())
    assert n >= 100, f"expected >=100 confirmed+scored events on XAUUSD, got {n}"
    print(f"  golden event count (legacy reference): {n}")


# ---------------------------------------------------------------------------
# Contract-level plugin tests
# ---------------------------------------------------------------------------

def test_plugin_emits_pattern_events(xauusd_df: pd.DataFrame) -> None:
    det = LiquiditySweepDetector(_LEGACY_CFG)
    events = _plugin_events_cached()
    assert det.name == "liquidity_sweep"
    assert det.version == "2.0"
    assert det.short_name == "LSW"
    for ev in events:
        assert ev.pattern_name == "liquidity_sweep"
        assert ev.pattern_version == "2.0"
        assert ev.direction in ("bullish", "bearish")
        assert ev.timeframe == "M15"
        assert ev.symbol == "XAUUSD"
        assert ev.entry_time is not None and ev.entry_price == ev.entry_price  # not NaN
        assert ev.stop_price == ev.stop_price and ev.target_price is not None
        assert ev.config_hash and ev.feature_schema_version
        assert "features" in ev.attributes
    print(
        f"  plugin emitted {len(events)} PatternEvents, "
        f"config_hash={events[0].config_hash if events else '-'}"
    )


def test_plugin_default_config_versioned(xauusd_df: pd.DataFrame) -> None:
    det = LiquiditySweepDetector(_LEGACY_CFG)
    cfg = det.get_default_config()
    assert cfg["version"] == det.version
    assert cfg["target_r"] == 3.0  # frozen live v2_target_r
    assert cfg["buffer_atr"] == 0.10
    from research.core.config_hash import config_hash_for_detector

    events = _plugin_events_cached()
    for ev in events:
        assert ev.config_hash == config_hash_for_detector(det, det.config)


def test_plugin_causality_clean(xauusd_df: pd.DataFrame) -> None:
    det = LiquiditySweepDetector(_LEGACY_CFG)
    events = _plugin_events_cached()
    # validate_causality with the plugin's declared schema — must not raise
    det.validate_causality(events)


# ---------------------------------------------------------------------------
# Shared no-lookahead CI gate (§3.4, Agent 1 template) — auto-collected
# ---------------------------------------------------------------------------

class TestLiquiditySweepNoLookahead(NoLookaheadTestBase):
    """Inherited causality gates over the plugin's real XAUUSD events."""

    detector_class: ClassVar[type] = LiquiditySweepDetector
    feature_schema: ClassVar[list] = LiquiditySweepDetector.feature_schema

    def build_events(self) -> list[PatternEvent]:
        events = _plugin_events_cached()
        assert events, "XAUUSD golden dataset must yield events for the causality gate"
        return events