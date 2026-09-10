"""Average True Range (ATR) and causal volatility indicators.

Implements section 8 of ``huong_dan.md``: ATR is the simple rolling mean of the
True Range over ``period`` candles (baseline ``atr_period: 14``).

Causality contract (guide sections 3.1/8)
-----------------------------------------
Every value at candle ``t`` is computed exclusively from candles ``0..t``:

* ``true_range[t]`` uses ``high[t]``, ``low[t]`` and ``close[t-1]`` (the
  previous candle's close, via ``shift(1)``) — never ``close[t+1]``.
* ``atr[t]`` is the mean of ``true_range`` over candles ``[t-period+1, t]``
  (``rolling(period).mean()``, no ``center=True``).  The first ``period-1``
  rows are NaN because a full window is not yet available.

Input frames must be chronologically ordered: ``add_atr`` raises on a
non-monotonic index so a shuffled frame can never silently leak data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = ("open", "high", "low", "close")


def _validate_ohlc(df: pd.DataFrame, func: str) -> None:
    """Validate that ``df`` is a non-empty, sorted, numeric OHLC frame."""
    missing = [c for c in _OHLC if c not in df.columns]
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
    non_numeric = [c for c in _OHLC if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"{func}: columns {non_numeric} must be numeric")


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ``true_range`` and ``atr`` columns (simple rolling ATR).

    Mirrors the guide's section 8 exactly — ATR is the rolling mean of the
    true range; Wilder's smoothing can be added later, and the chosen method
    is documented in ``configs/baseline.yaml`` (``indicators.atr_period``).

    Returns a copy of ``df`` with ``true_range`` and ``atr`` appended; the
    input frame is never mutated.
    """
    out = df.copy()
    _validate_ohlc(out, "add_atr")
    if period < 1:
        raise ValueError(f"add_atr: period must be >= 1, got {period}")

    previous_close = out["close"].shift(1)

    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["true_range"] = true_range
    out["atr"] = true_range.rolling(period).mean()
    return out


def atr_percentile_causal(atr: pd.Series, window: int = 200) -> pd.Series:
    """Causal rolling percentile of ATR (never a full-dataset percentile).

    At each row ``t`` the percentile is computed over the trailing ``window``
    ATR values *including* ``t`` via ``rolling(window).rank(pct=True)`` —
    no future data is used.  Rows before the first full window are NaN.
    """
    if window < 1:
        raise ValueError(
            f"atr_percentile_causal: window must be >= 1, got {window}"
        )
    return atr.rolling(window).rank(pct=True)


def volatility_regime_causal(
    atr_percentile: pd.Series,
    low: float = 0.33,
    high: float = 0.67,
) -> pd.Series:
    """Map a causal ATR percentile into a ``low``/``normal``/``high`` regime.

    NaN percentiles propagate as NaN (no regime known yet).
    """
    out = pd.Series("normal", index=atr_percentile.index, dtype="object")
    out[atr_percentile < low] = "low"
    out[atr_percentile > high] = "high"
    out[atr_percentile.isna()] = np.nan
    return out