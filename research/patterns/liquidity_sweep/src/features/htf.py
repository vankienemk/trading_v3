"""Higher-timeframe context features (guide section 13.7).

All values are strictly causal: only *closed* H1/H4 candles are used, and only
candles that closed strictly before the row that consumes them (via
:func:`src.data.resampler.closed_higher_timeframe_merge`, which relabels bins
to close time then ``ffill + shift(1)``).  The previous-day high/low come from
:func:`src.data.resampler.previous_day_high_low`, which only uses the prior
completed calendar day.

This module returns a per-bar context frame indexed like the M15 ``df``:
``h1_return``, ``h1_trend``, ``h4_trend``, ``prev_day_high``, ``prev_day_low``.
"""

from __future__ import annotations

import pandas as pd

from src.data.resampler import (
    closed_higher_timeframe_merge,
    previous_day_high_low,
)
from src.indicators.trend import ema_slope_sign


def build_htf_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-bar frame with closed-H1/H4 and previous-day context.

    Result columns are index-aligned with ``df`` and every value is a function
    of bars strictly before the row's own timestamp (or the prior day's close),
    so the frame is truncation-invariant.
    """
    out = df.copy()

    # Closed H1 / H4 context (never a still-forming candle).
    h1 = closed_higher_timeframe_merge(out, "1h", "h1", columns=("close",))
    h4 = closed_higher_timeframe_merge(out, "4h", "h4", columns=("close",))

    # H1 return over the last closed H1 candle; H1 trend = EMA(50) slope sign.
    h1_close = h1["h1_close"]
    h1_return = h1_close.pct_change()
    h1_trend = ema_slope_sign(h1_close, period=50, lag=1)

    h4_close = h4["h4_close"]
    h4_trend = ema_slope_sign(h4_close, period=50, lag=1)

    # Previous completed calendar day high/low (causal).
    pdh = previous_day_high_low(out)

    return pd.DataFrame(
        {
            "h1_return": h1_return,
            "h1_trend": h1_trend,
            "h4_trend": h4_trend,
            "prev_day_high": pdh["prev_day_high"],
            "prev_day_low": pdh["prev_day_low"],
        },
        index=out.index,
    )


__all__ = ["build_htf_context"]
