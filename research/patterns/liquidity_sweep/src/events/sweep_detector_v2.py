"""V2 liquidity-sweep detector: baseline + nguoc_trend + max_pen≤0.20 + configurable R:R.

    Baseline detector (src.events.sweep_detector) detects all sweeps:
      low < liq_low, close > liq_low, penetration in [0.05, 0.50],
      wick >= 0.35, reclaim >= 0.00

    V2 adds two filters on top:
      1. **nguoc_trend**: LONG when H1 trend is up (+1), SHORT when H1 trend is down (-1)
         — the sweep thrust counters the H1 trend; trade follows the H1 direction.
      2. **max_penetration_atr ≤ 0.20**: only keep sweeps with shallow penetration.
         (From t10 analysis: triple intersection PF=3.30 pre-cost, n=38)

    V2 event table adds a ``v2_target_r`` column (configurable R:R, default 2.0)
    that downstream labeling can use instead of the fixed 2R target.

    All imports are from the baseline project (src.*) — the v2 module is a
    *consumer* of the baseline project, not a fork.  The baseline v1.2.0
    pipeline is never touched.

    Causality contract
    ------------------
    ``h1_trend`` is computed via ``build_htf_context`` which uses closed H1
    candles only (via ``ffill + shift(1)``), so it is strictly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.events.deduplication import select_deduplicated_events
from src.events.sweep_detector import (
    _EVENT_COLUMNS,
    BEARISH,
    BULLISH,
    add_sweep_features,
)
from src.features.htf import build_htf_context

# Default v2 parameters (from t10 triple intersection analysis)
V2_MAX_PENETRATION_ATR = 0.20
V2_DEFAULT_TARGET_R = 2.0  # configurable; optimal from t10 = 3.0:1

_V2_EVENT_COLUMNS = [*list(_EVENT_COLUMNS), "h1_trend", "v2_target_r"]


def detect_sweeps_v2(
    df: pd.DataFrame,
    atr_period: int = 14,
    level_lookback: int = 20,
    min_penetration_atr: float = 0.05,
    max_penetration_atr: float = V2_MAX_PENETRATION_ATR,
    min_wick_ratio: float = 0.35,
    min_reclaim_atr: float = 0.0,
    v2_nguoc_trend: bool = True,
) -> pd.DataFrame:
    """Detect v2 sweeps: baseline + nguoc_trend + max_penetration_atr ≤ 0.20.

    Returns the feature frame (all rows) with ``sweep_v2_long``/``sweep_v2_short``
    flags and ``h1_trend`` column.  Non-v2 sweeps are still flagged in
    ``sweep_long``/``sweep_short`` (from baseline) for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV frame with ``open``, ``high``, ``low``, ``close`` columns.
    v2_nguoc_trend : bool
        If True (default), apply the nguoc_trend filter.  Set to False to
        isolate the max_penetration_atr effect.
    """
    if max_penetration_atr < min_penetration_atr:
        raise ValueError(
            f"detect_sweeps_v2: max_penetration_atr ({max_penetration_atr}) must be "
            f">= min_penetration_atr ({min_penetration_atr})"
        )

    # Step 1: baseline sweep features (ATR, levels, wicks, reclaim)
    out = add_sweep_features(df, atr_period=atr_period, level_lookback=level_lookback)

    # Step 2: H1 trend (causal, closed H1 candles only)
    htf = build_htf_context(df)
    out["h1_trend"] = htf["h1_trend"]

    # Step 3: baseline sweep flags (same as original)
    out["sweep_long"] = (
        out["liq_low"].notna()
        & out["atr"].notna()
        & (out["low"] < out["liq_low"])
        & (out["close"] > out["liq_low"])
        & (out["low_pen_atr"] >= min_penetration_atr)
        & (out["low_pen_atr"] <= max_penetration_atr)
        & (out["lower_wick_ratio"] >= min_wick_ratio)
        & (out["low_reclaim_atr"] >= min_reclaim_atr)
    )
    out["sweep_short"] = (
        out["liq_high"].notna()
        & out["atr"].notna()
        & (out["high"] > out["liq_high"])
        & (out["close"] < out["liq_high"])
        & (out["high_pen_atr"] >= min_penetration_atr)
        & (out["high_pen_atr"] <= max_penetration_atr)
        & (out["upper_wick_ratio"] >= min_wick_ratio)
        & (out["high_reclaim_atr"] >= min_reclaim_atr)
    )

    # Step 4: v2 filters
    if v2_nguoc_trend:
        # nguoc_trend: bullish sweep while H1 uptrend (+1)
        #              bearish sweep while H1 downtrend (-1)
        trend_ok = out["h1_trend"].notna()
        long_nguoc = out["sweep_long"] & (out["h1_trend"] == 1.0)
        short_nguoc = out["sweep_short"] & (out["h1_trend"] == -1.0)
        out["sweep_v2_long"] = trend_ok & long_nguoc
        out["sweep_v2_short"] = trend_ok & short_nguoc
    else:
        out["sweep_v2_long"] = out["sweep_long"].copy()
        out["sweep_v2_short"] = out["sweep_short"].copy()

    return out


def _candidate_rows_v2(out: pd.DataFrame) -> pd.DataFrame:
    """Convert per-bar v2 sweep flags into the candidate event frame (pre-dedup).

    Uses ``sweep_v2_long``/``sweep_v2_short`` instead of baseline flags.
    Columns include the schema columns plus ``h1_trend``.
    """
    flags = out["sweep_v2_long"].fillna(False) | out["sweep_v2_short"].fillna(False)
    positions = np.flatnonzero(flags.to_numpy())
    rows = []
    for pos in positions:
        if bool(out["sweep_v2_long"].iloc[pos]):
            rows.append(_candidate_row_v2(out, pos, BULLISH))
        if bool(out["sweep_v2_short"].iloc[pos]):
            rows.append(_candidate_row_v2(out, pos, BEARISH))
    return pd.DataFrame(rows, columns=["position", *_V2_EVENT_COLUMNS[1:]])


def _candidate_row_v2(out: pd.DataFrame, pos: int, direction: str) -> dict:
    """One v2 candidate event row for a single (bar, direction) pair."""
    if direction == BULLISH:
        level_id = "rolling_low"
        level_price = out["liq_low"].iloc[pos]
        penetration = out["low_pen_atr"].iloc[pos]
        wick = out["lower_wick_ratio"].iloc[pos]
        reclaim = out["low_reclaim_atr"].iloc[pos]
    else:
        level_id = "rolling_high"
        level_price = out["liq_high"].iloc[pos]
        penetration = out["high_pen_atr"].iloc[pos]
        wick = out["upper_wick_ratio"].iloc[pos]
        reclaim = out["high_reclaim_atr"].iloc[pos]

    h1_t = out["h1_trend"].iloc[pos]
    h1_t = int(h1_t) if pd.notna(h1_t) else 0

    return {
        "position": int(pos),
        "event_time": out.index[pos],
        "direction": direction,
        "level_id": level_id,
        "level_price": float(level_price),
        "event_open": float(out["open"].iloc[pos]),
        "event_high": float(out["high"].iloc[pos]),
        "event_low": float(out["low"].iloc[pos]),
        "event_close": float(out["close"].iloc[pos]),
        "penetration_atr": float(penetration),
        "wick_ratio": float(wick),
        "reclaim_atr": float(reclaim),
        "h1_trend": h1_t,
        "v2_target_r": V2_DEFAULT_TARGET_R,
    }


def build_sweep_events_v2(
    df: pd.DataFrame,
    config: dict | None = None,
    *,
    atr_period: int | None = None,
    level_lookback: int | None = None,
    min_penetration_atr: float | None = None,
    max_penetration_atr: float | None = None,
    min_wick_ratio: float | None = None,
    min_reclaim_atr: float | None = None,
    cooldown_bars: int | None = None,
    group_rule: str | None = None,
    v2_nguoc_trend: bool = True,
    v2_target_r: float = V2_DEFAULT_TARGET_R,
) -> pd.DataFrame:
    """Detect v2 sweeps, deduplicate, and return the v2 event table.

    Parameters resolve in strict priority order (explicit keyword argument →
    ``config`` value → v2 default), same as the baseline function.

    Output columns::
        event_id event_time direction level_id level_price
        event_open event_high event_low event_close
        penetration_atr wick_ratio reclaim_atr
        h1_trend v2_target_r

    ``v2_target_r`` is the configurable R:R ratio (default 2.0) that the
    downstream labeling step should use as the target instead of the fixed 2R.
    """
    cfg = config or {}
    ind = cfg.get("indicators", {})
    liq = cfg.get("liquidity", {})
    swp = cfg.get("sweep", {})
    atr_period = ind.get("atr_period", 14) if atr_period is None else atr_period
    level_lookback = (
        liq.get("rolling_lookback", 20) if level_lookback is None else level_lookback
    )
    min_penetration_atr = (
        swp.get("min_penetration_atr", 0.05)
        if min_penetration_atr is None
        else min_penetration_atr
    )
    max_penetration_atr = (
        V2_MAX_PENETRATION_ATR
        if max_penetration_atr is None
        else max_penetration_atr
    )
    min_wick_ratio = (
        swp.get("min_wick_ratio", 0.35) if min_wick_ratio is None else min_wick_ratio
    )
    min_reclaim_atr = (
        swp.get("min_reclaim_atr", 0.0) if min_reclaim_atr is None else min_reclaim_atr
    )
    cooldown_bars = (
        swp.get("cooldown_bars", 4) if cooldown_bars is None else cooldown_bars
    )
    group_rule = swp.get("group_rule", "first") if group_rule is None else group_rule
    if group_rule not in ("first", "deepest_penetration", "strongest_reclaim"):
        raise ValueError(
            f"build_sweep_events_v2: group_rule must be one of "
            f"{('first', 'deepest_penetration', 'strongest_reclaim')}, "
            f"got {group_rule!r}"
        )

    out = detect_sweeps_v2(
        df,
        atr_period=atr_period,
        level_lookback=level_lookback,
        min_penetration_atr=min_penetration_atr,
        max_penetration_atr=max_penetration_atr,
        min_wick_ratio=min_wick_ratio,
        min_reclaim_atr=min_reclaim_atr,
        v2_nguoc_trend=v2_nguoc_trend,
    )
    candidates = _candidate_rows_v2(out)
    if candidates.empty:
        return pd.DataFrame(columns=_V2_EVENT_COLUMNS)

    selected = select_deduplicated_events(
        candidates,
        cooldown_bars=cooldown_bars,
        group_rule=group_rule,
    )
    selected = selected.sort_values(["event_time", "direction"], kind="stable")
    selected["event_id"] = [
        f"V2-{i:06d}" for i in range(len(selected))
    ]
    # Override target_r from parameter
    selected["v2_target_r"] = v2_target_r
    return selected[_V2_EVENT_COLUMNS].reset_index(drop=True)


def sweep_event_v2_schema_columns() -> list[str]:
    """The exact public column set of :func:`build_sweep_events_v2`."""
    return list(_V2_EVENT_COLUMNS)