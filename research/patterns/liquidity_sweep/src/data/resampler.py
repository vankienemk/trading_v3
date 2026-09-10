"""Resampling utilities for higher-timeframe context features.

All resampling is causal: a higher-timeframe candle is only available to the
lower timeframe once it has *closed*.  The helpers here return an index-aligned
frame whose values at time ``t`` reflect the most recently *closed* H1/H4
candle (strictly before ``t``), never a candle still forming.

Implementation note (why the close-time relabel is needed): ``pandas``
resample labels each bin by its *start* time, so the H1 bin ``01:00`` really
contains the M15 candles ``01:00..01:45`` and only closes at ``02:00``.
Forward-filling over bins labeled by start time would therefore expose a
still-forming candle — a look-ahead leak.  We relabel every bin to its close
time before the merge, then ``ffill + shift(1)`` so a candle closing at time
``c`` becomes visible from the M15 bar ``c + 15min`` onward.
"""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M15 OHLCV to a coarser timeframe (e.g. ``"1h"``).

    Returns a frame indexed by bin *start* times (pandas convention), with
    ``open=first``, ``high=max``, ``low=min``, ``close=last``, ``volume=sum``.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg if c in df.columns]
    return df[cols].resample(rule).agg({c: agg[c] for c in cols}).dropna(subset=["close"])


def closed_higher_timeframe_merge(
    df: pd.DataFrame,
    rule: str,
    prefix: str,
    columns: tuple = ("open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Merge *closed* higher-timeframe values onto an M15 frame.

    For each M15 row at time ``t`` we attach the most recent higher-timeframe
    candle whose close time is strictly before ``t`` (i.e. the candle has
    closed).  Returns a copy of ``df`` with ``{prefix}_{col}`` columns added.
    """
    ht = resample_ohlcv(df, rule)
    # Relabel bins from start time to close time: a bin starting at L closes
    # at L + rule.
    ht.index = ht.index + pd.Timedelta(rule)

    out = df.copy()
    for col in columns:
        if col not in ht.columns:
            continue
        series = ht[col]
        # Reindex to M15 timestamps and forward-fill from the *previous* HT
        # close: at time t this yields the HT candle whose close time is <= t.
        mapped = series.reindex(out.index, method="ffill")
        # Shift by one M15 bar so a candle closing at t is only usable from
        # t+1 onward (strictly *closed* before the row that uses it).
        out[f"{prefix}_{col}"] = mapped.shift(1)
    return out


def previous_day_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """Attach previous trading day's high/low to each row, causally.

    Uses the calendar date (UTC) of the index.  A row's ``prev_day_high`` /
    ``prev_day_low`` refer to the *prior* calendar date only; rows on the
    first day carry NaN.  Deterministic reindex-based lookup (no dependence
    on group ordering).
    """
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("previous_day_high_low requires a DatetimeIndex frame")
    day = idx.normalize()

    daily = pd.DataFrame(
        {
            "high": out["high"].groupby(day).max(),
            "low": out["low"].groupby(day).min(),
        }
    )
    prev = daily.shift(1)  # shift by one calendar day; first day -> NaN

    day_values = pd.Series(day, index=out.index)
    out["prev_day_high"] = prev["high"].reindex(day_values.to_numpy()).to_numpy()
    out["prev_day_low"] = prev["low"].reindex(day_values.to_numpy()).to_numpy()
    return out