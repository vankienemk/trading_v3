"""Confirmed swing high/low levels (guide §9.2) and H1 swing levels (Phase 4).

Causality contract
------------------
A swing at candle ``t`` with ``left_bars``/``right_bars`` is confirmed only once
the ``right_bars`` candles to its right have closed::

    origin_pos = t            # the pivot candle
    known_pos  = t + right_bars
    known_at   = timestamp[known_pos]

The detector emits the level row, but downstream (sweep detector, registry)
may only *use* the level for bars with ``timestamp >= known_at``.  The pivot
must be strictly lower/higher than *all* left and right neighbours (guide
§9.2: "lower than the low of the three preceding candles" — an equal neighbour
does not confirm a swing).

H1 swing levels (Phase 4) reuse the same fractal rule on *closed* H1 candles.
An H1 candle only closes at its bin end, so the confirming H1 candle
``j + right_bars`` makes the level known at its close time; ``known_at`` maps
to the first M15 bar at/after that instant.  A still-forming (partial) H1
candle never contributes a level.

Schema: rows follow ``src.schema.LEVEL_COLUMNS`` with ``level_type="swing"``,
``direction ∈ {low, high}`` and the locked level-id scheme
``{type}_{direction}_{origin_pos}`` (H1 rows prefixed ``h1_``).  Three
documented extension columns are added: ``known_pos`` (bar position of
``known_at``), ``is_h1`` (True for H1 swings) and ``max_age_bars`` (expiry
policy consumed by ``level_registry``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.data.resampler import resample_ohlcv
from src.schema import LEVEL_COLUMNS

#: Swing detector output = schema §4 columns + known_pos/is_h1/max_age_bars.
SWING_OUTPUT_COLUMNS: list[str] = [
    *LEVEL_COLUMNS, "known_pos", "is_h1", "max_age_bars",
]


def _validate_frame(df: pd.DataFrame, func: str) -> None:
    """Validate a non-empty, chronologically sorted OHLC frame."""
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"{func}: missing required columns {missing}")
    if len(df) == 0:
        raise ValueError(f"{func}: empty DataFrame")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{func}: index must be chronologically sorted (monotonic "
            "increasing); an unsorted frame breaks causal (no-look-ahead) "
            "semantics"
        )
    non_numeric = [c for c in ("high", "low") if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"{func}: columns {non_numeric} must be numeric")


def _swing_positions(
    low: np.ndarray,
    high: np.ndarray,
    left_bars: int,
    right_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly-confirmed swing-low and swing-high positions.

    The pivot must be strictly lower (higher) than every low (high) in the
    ``left_bars`` preceding and ``right_bars`` following candles.
    """
    n = len(low)
    swing_low = np.zeros(n, dtype=bool)
    swing_high = np.zeros(n, dtype=bool)
    for i in range(left_bars, n - right_bars):
        if (
            low[i] < low[i - left_bars : i].min()
            and low[i] < low[i + 1 : i + 1 + right_bars].min()
        ):
            swing_low[i] = True
        if (
            high[i] > high[i - left_bars : i].max()
            and high[i] > high[i + 1 : i + 1 + right_bars].max()
        ):
            swing_high[i] = True
    return np.flatnonzero(swing_low), np.flatnonzero(swing_high)


def detect_swing_levels(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    max_age_bars: int = 200,
) -> pd.DataFrame:
    """Detect confirmed swing highs and lows on the M15 frame.

    Returns a long-format levels table (schema §4 + ``known_pos``/``is_h1``/
    ``max_age_bars``) sorted by ascending ``known_at``.  A swing at candle
    ``t`` is only known at ``t + right_bars`` — rows never leak future data.
    """
    _validate_frame(df, "detect_swing_levels")
    if left_bars < 1:
        raise ValueError(
            f"detect_swing_levels: left_bars must be >= 1, got {left_bars}"
        )
    if right_bars < 1:
        raise ValueError(
            f"detect_swing_levels: right_bars must be >= 1, got {right_bars}"
        )

    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    low_pos, high_pos = _swing_positions(low, high, left_bars, right_bars)
    rows = _swing_rows(
        df, low, high, low_pos, high_pos,
        known_pos_fn=lambda pos: pos + right_bars,
        level_prefix="swing_", is_h1=False, max_age_bars=max_age_bars,
    )
    out = pd.DataFrame(rows, columns=SWING_OUTPUT_COLUMNS)
    if len(out):
        out = out.sort_values("known_at").reset_index(drop=True)
    return out


def detect_h1_swing_levels(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    max_age_bars: int = 200,
    rule: str = "1h",
) -> pd.DataFrame:
    """Detect swing highs/lows on *closed* H1 candles, mapped onto M15 time.

    The M15 frame is resampled to H1; the final partial H1 candle is dropped
    (it has not closed).  Swing detection runs on the closed H1 candles; a
    pivot at H1 candle ``j`` is known once candle ``j + right_bars`` closes,
    i.e. its bin end.  ``known_at`` is the first M15 bar at/after that close
    time; pivots whose confirming candle has not closed in-sample are dropped.
    """
    _validate_frame(df, "detect_h1_swing_levels")
    if left_bars < 1:
        raise ValueError(
            f"detect_h1_swing_levels: left_bars must be >= 1, got {left_bars}"
        )
    if right_bars < 1:
        raise ValueError(
            f"detect_h1_swing_levels: right_bars must be >= 1, got {right_bars}"
        )

    h1 = resample_ohlcv(df, rule)
    if len(h1) == 0:
        return pd.DataFrame(columns=SWING_OUTPUT_COLUMNS)
    delta = pd.Timedelta(rule)
    # Drop the final H1 candle when its bin end is past the last M15 bar
    # (it is still forming and must never contribute).
    h1 = h1[h1.index + delta <= df.index[-1]]
    if len(h1) == 0:
        return pd.DataFrame(columns=SWING_OUTPUT_COLUMNS)

    m15_index = df.index  # DatetimeIndex — pandas searchsorted accepts Timestamps
    h1_low = h1["low"].to_numpy()
    h1_high = h1["high"].to_numpy()
    low_pos, high_pos = _swing_positions(h1_low, h1_high, left_bars, right_bars)

    def _known_pos_fn(pivot_h1_pos: int) -> int:
        # Confirming candle closes at its bin end; level is known at the first
        # M15 bar at/after that instant.
        close_time = h1.index[pivot_h1_pos + right_bars] + delta
        # pandas searchsorted on the DatetimeIndex keeps the query as a
        # Timestamp — np.searchsorted on a datetime64 ndarray raises a
        # TypeError under numpy>=2 when mixed with Timestamp scalars.
        return int(m15_index.searchsorted(close_time, side="left"))

    def _origin_pos_fn(pivot_h1_pos: int) -> int:
        # First M15 bar at/after the pivot H1 candle's start time.
        return int(m15_index.searchsorted(h1.index[pivot_h1_pos], side="left"))

    ts = df.index.to_numpy()
    n = len(df)
    rows: list[dict] = []
    for pos in low_pos:
        known_pos = _known_pos_fn(int(pos))
        if known_pos >= n:
            continue
        origin_pos = _origin_pos_fn(int(pos))
        price = float(h1_low[pos])
        rows.append(
            {
                "level_id": f"h1_swing_low_{origin_pos}",
                "level_type": "swing",
                "direction": "low",
                "price": price,
                "price_min": price,
                "price_max": price,
                "origin_pos": origin_pos,
                "origin_time": ts[origin_pos],
                "known_at": ts[known_pos],
                "status": "active",
                "touch_count": 1,
                "first_touch_time": ts[origin_pos],
                "last_touch_time": ts[origin_pos],
                "known_pos": known_pos,
                "is_h1": True,
                "max_age_bars": int(max_age_bars),
            }
        )
    for pos in high_pos:
        known_pos = _known_pos_fn(int(pos))
        if known_pos >= n:
            continue
        origin_pos = _origin_pos_fn(int(pos))
        price = float(h1_high[pos])
        rows.append(
            {
                "level_id": f"h1_swing_high_{origin_pos}",
                "level_type": "swing",
                "direction": "high",
                "price": price,
                "price_min": price,
                "price_max": price,
                "origin_pos": origin_pos,
                "origin_time": ts[origin_pos],
                "known_at": ts[known_pos],
                "status": "active",
                "touch_count": 1,
                "first_touch_time": ts[origin_pos],
                "last_touch_time": ts[origin_pos],
                "known_pos": known_pos,
                "is_h1": True,
                "max_age_bars": int(max_age_bars),
            }
        )
    out = pd.DataFrame(rows, columns=SWING_OUTPUT_COLUMNS)
    if len(out):
        out = out.sort_values("known_at").reset_index(drop=True)
    return out


def _swing_rows(
    df: pd.DataFrame,
    low: np.ndarray,
    high: np.ndarray,
    low_pos: np.ndarray,
    high_pos: np.ndarray,
    known_pos_fn: Callable[[int], int],
    level_prefix: str,
    is_h1: bool,
    max_age_bars: int,
) -> list[dict]:
    """Build the level rows for M15 swing pivots (``known_at = t + right_bars``)."""
    ts = df.index.to_numpy()
    n = len(df)
    rows: list[dict] = []

    for pos in low_pos:
        known_pos = int(known_pos_fn(pos))
        if known_pos >= n:
            continue
        price = float(low[pos])
        rows.append(
            {
                "level_id": f"{level_prefix}low_{int(pos)}",
                "level_type": "swing",
                "direction": "low",
                "price": price,
                "price_min": price,
                "price_max": price,
                "origin_pos": int(pos),
                "origin_time": ts[pos],
                "known_at": ts[known_pos],
                "status": "active",
                "touch_count": 1,
                "first_touch_time": ts[pos],
                "last_touch_time": ts[pos],
                "known_pos": known_pos,
                "is_h1": is_h1,
                "max_age_bars": int(max_age_bars),
            }
        )
    for pos in high_pos:
        known_pos = int(known_pos_fn(pos))
        if known_pos >= n:
            continue
        price = float(high[pos])
        rows.append(
            {
                "level_id": f"{level_prefix}high_{int(pos)}",
                "level_type": "swing",
                "direction": "high",
                "price": price,
                "price_min": price,
                "price_max": price,
                "origin_pos": int(pos),
                "origin_time": ts[pos],
                "known_at": ts[known_pos],
                "status": "active",
                "touch_count": 1,
                "first_touch_time": ts[pos],
                "last_touch_time": ts[pos],
                "known_pos": known_pos,
                "is_h1": is_h1,
                "max_age_bars": int(max_age_bars),
            }
        )
    return rows