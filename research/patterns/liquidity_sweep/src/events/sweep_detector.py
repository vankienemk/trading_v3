"""Baseline liquidity-sweep detector (guide section 10).

A *bullish sweep* candle satisfies every condition of section 10.1::

    low[t] < liq_low                             level pierced on the low
    close[t] > liq_low                           level reclaimed by the close
    min_penetration_atr <= penetration_atr <= max_penetration_atr
    lower_wick_ratio >= min_wick_ratio
    reclaim_atr >= min_reclaim_atr

with the guide's definitions (section 10.1)::

    penetration_atr  = (liq_low - low[t]) / ATR[t]
    lower_wick       = min(open[t], close[t]) - low[t]
    lower_wick_ratio = lower_wick / (high[t] - low[t])
    reclaim_atr      = (close[t] - liq_low) / ATR[t]

A *bearish sweep* (section 10.2) is the mirror image against ``liq_high`` and
uses the upper wick::

    high[t] > liq_high
    close[t] < liq_high
    penetration_atr = (high[t] - liq_high) / ATR[t]
    upper_wick      = high[t] - max(open[t], close[t])
    reclaim_atr     = (liq_high - close[t]) / ATR[t]

Causality contract (guide sections 3.1 / 8 / 9.1)
------------------------------------------------
``liq_low`` / ``liq_high`` come from :mod:`src.liquidity.rolling_levels`
(``shift(1).rolling(lookback)`` — the current candle never forms the level it
is measured against) and ATR from :mod:`src.indicators.atr` (rolling mean of
True Range).  Both are strictly causal, so a sweep flag at candle ``t`` is a
function of candles ``0..t`` only.  Input frames must be chronologically
sorted; a non-monotonic index raises so a shuffled frame cannot silently leak
future data.

Duplicate removal and the cooldown live in
:mod:`src.events.deduplication` (guide section 12) and are applied by
:func:`build_sweep_events`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.atr import add_atr
from src.liquidity.rolling_levels import rolling_level_at

BULLISH = "bullish"
BEARISH = "bearish"

_EVENT_COLUMNS = [
    "event_id",
    "event_time",
    "direction",
    "level_id",
    "level_price",
    "event_open",
    "event_high",
    "event_low",
    "event_close",
    "penetration_atr",
    "wick_ratio",
    "reclaim_atr",
]


def add_sweep_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    level_lookback: int = 20,
) -> pd.DataFrame:
    """Append sweep-metric columns to a copy of ``df``.

    Columns added: ``true_range``/``atr`` (via ``add_atr``), ``liq_low`` /
    ``liq_high`` (via ``rolling_level_at``), ``lower_wick``, ``upper_wick``,
    ``lower_wick_ratio``, ``upper_wick_ratio``, ``low_pen_atr``,
    ``high_pen_atr``, ``low_reclaim_atr``, ``high_reclaim_atr``.

    The input frame is never mutated.  Validation (columns, numeric, non-empty,
    monotonic index) is delegated to ``add_atr`` / ``rolling_level_at``.
    """
    if atr_period < 1:
        raise ValueError(f"add_sweep_features: atr_period must be >= 1, got {atr_period}")
    if level_lookback < 1:
        raise ValueError(
            f"add_sweep_features: level_lookback must be >= 1, got {level_lookback}"
        )

    out = add_atr(df, period=atr_period)
    liq_low, liq_high = rolling_level_at(out, lookback=level_lookback)
    out["liq_low"] = liq_low
    out["liq_high"] = liq_high

    # Zero-range candles (doji / flat ticks) cannot support a wick ratio; the
    # guide replaces range 0 with NaN so those candles can never be sweeps.
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)

    lower_body = out[["open", "close"]].min(axis=1)
    upper_body = out[["open", "close"]].max(axis=1)

    out["lower_wick"] = lower_body - out["low"]
    out["upper_wick"] = out["high"] - upper_body
    out["lower_wick_ratio"] = out["lower_wick"] / candle_range
    out["upper_wick_ratio"] = out["upper_wick"] / candle_range

    out["low_pen_atr"] = (out["liq_low"] - out["low"]) / out["atr"]
    out["high_pen_atr"] = (out["high"] - out["liq_high"]) / out["atr"]
    out["low_reclaim_atr"] = (out["close"] - out["liq_low"]) / out["atr"]
    out["high_reclaim_atr"] = (out["liq_high"] - out["close"]) / out["atr"]
    return out


def detect_sweeps(
    df: pd.DataFrame,
    atr_period: int = 14,
    level_lookback: int = 20,
    min_penetration_atr: float = 0.05,
    max_penetration_atr: float = 0.50,
    min_wick_ratio: float = 0.35,
    min_reclaim_atr: float = 0.0,
) -> pd.DataFrame:
    """Detect baseline sweeps; return the feature frame plus ``sweep_long``/``sweep_short``.

    Mirrors the guide's section 10.3 baseline detector (rolling levels only).
    The strict inequalities ``low < liq_low`` / ``close > liq_low`` (and the
    bearish mirrors) are applied explicitly so the detector matches the guide's
    section 10.1/10.2 wording even when ``min_penetration_atr`` is configured
    to 0.
    """
    if max_penetration_atr < min_penetration_atr:
        raise ValueError(
            f"detect_sweeps: max_penetration_atr ({max_penetration_atr}) must be "
            f">= min_penetration_atr ({min_penetration_atr})"
        )
    if min_penetration_atr < 0.0:
        raise ValueError(
            f"detect_sweeps: min_penetration_atr must be >= 0, got {min_penetration_atr}"
        )
    if not 0.0 <= min_wick_ratio <= 1.0:
        raise ValueError(
            f"detect_sweeps: min_wick_ratio must be in [0, 1], got {min_wick_ratio}"
        )

    out = add_sweep_features(df, atr_period=atr_period, level_lookback=level_lookback)

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
    return out


def _candidate_rows(out: pd.DataFrame) -> pd.DataFrame:
    """Convert per-bar sweep flags into the candidate event frame (pre-dedup).

    A bar can legitimately pierce both levels (bullish AND bearish sweep); each
    direction then contributes its own candidate row.
    """
    flags = out["sweep_long"].fillna(False) | out["sweep_short"].fillna(False)
    positions = np.flatnonzero(flags.to_numpy())
    rows = []
    for pos in positions:
        if bool(out["sweep_long"].iloc[pos]):
            rows.append(_candidate_row(out, pos, BULLISH))
        if bool(out["sweep_short"].iloc[pos]):
            rows.append(_candidate_row(out, pos, BEARISH))
    return pd.DataFrame(rows, columns=["position", *_EVENT_COLUMNS[1:]])


def _candidate_row(out: pd.DataFrame, pos: int, direction: str) -> dict:
    """One candidate event row for a single (bar, direction) pair."""
    if direction == BULLISH:
        level_id, level_price = "rolling_low", out["liq_low"].iloc[pos]
        penetration = out["low_pen_atr"].iloc[pos]
        wick = out["lower_wick_ratio"].iloc[pos]
        reclaim = out["low_reclaim_atr"].iloc[pos]
    else:
        level_id, level_price = "rolling_high", out["liq_high"].iloc[pos]
        penetration = out["high_pen_atr"].iloc[pos]
        wick = out["upper_wick_ratio"].iloc[pos]
        reclaim = out["high_reclaim_atr"].iloc[pos]
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
    }


def build_sweep_events(
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
) -> pd.DataFrame:
    """Detect sweeps, deduplicate, and return the final event table.

    Output columns (guide section 5.4)::

        event_id event_time direction level_id level_price
        event_open event_high event_low event_close
        penetration_atr wick_ratio reclaim_atr

    Parameters resolve in strict priority order (explicit keyword argument →
    ``config`` value → baseline default), so a pipeline that passes its loaded
    ``cfg`` (e.g. ``configs/baseline.yaml``) runs the whole event flow from the
    configuration: ``indicators.atr_period``,
    ``liquidity.rolling_lookback``, ``sweep.min_penetration_atr`` /
    ``max_penetration_atr`` / ``min_wick_ratio`` / ``min_reclaim_atr`` /
    ``cooldown_bars`` and ``sweep.group_rule``.

    ``group_rule`` defaults to ``"first"`` (see ``configs/baseline.yaml``,
    QA finding F1) so the default event feed is strictly causal
    (truncation-invariant); ``deepest_penetration`` / ``strongest_reclaim``
    remain available as explicit opt-ins for retrospective analyses.
    ``event_id`` is an ascending ordinal assigned in chronological order
    (same-bar events of opposite direction are ordered by direction), so ids
    are deterministic for a given input frame.
    """
    from src.events.deduplication import select_deduplicated_events

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
        swp.get("max_penetration_atr", 0.50)
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
            f"build_sweep_events: group_rule must be one of "
            f"{('first', 'deepest_penetration', 'strongest_reclaim')}, "
            f"got {group_rule!r}"
        )

    out = detect_sweeps(
        df,
        atr_period=atr_period,
        level_lookback=level_lookback,
        min_penetration_atr=min_penetration_atr,
        max_penetration_atr=max_penetration_atr,
        min_wick_ratio=min_wick_ratio,
        min_reclaim_atr=min_reclaim_atr,
    )
    candidates = _candidate_rows(out)
    if candidates.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    selected = select_deduplicated_events(
        candidates,
        cooldown_bars=cooldown_bars,
        group_rule=group_rule,
    )
    selected = selected.sort_values(["event_time", "direction"], kind="stable")
    selected["event_id"] = [
        f"SWP-{i:06d}" for i in range(len(selected))
    ]
    return selected[_EVENT_COLUMNS].reset_index(drop=True)


def sweep_event_schema_columns() -> list[str]:
    """The exact public column set of :func:`build_sweep_events` (schema contract)."""
    return list(_EVENT_COLUMNS)