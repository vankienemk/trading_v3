"""Trend indicators (all causal)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (causal, left-to-right)."""
    return series.ewm(span=period, adjust=False).mean()


def ema_slope_sign(series: pd.Series, period: int, lag: int = 1) -> pd.Series:
    """Sign of the EMA slope: +1 / 0 / -1 using the last ``lag`` bars only."""
    e = ema(series, period)
    slope = e - e.shift(lag)
    sign = np.sign(slope)
    return sign
