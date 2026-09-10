"""Rolling high/low liquidity levels (guide section 9.1).

Baseline liquidity levels for Phase 1, and the input to the sweep detector.

Causality contract (guide sections 3.1 / 9.1)
---------------------------------------------
The level at candle ``t`` is derived exclusively from the ``lookback`` candles
*before* ``t``::

    liq_low[t]  = min(low[t-lookback : t])
    liq_high[t] = max(high[t-lookback : t])

implemented as ``df["low"].shift(1).rolling(lookback).min()`` (and the same
for ``high``).  ``shift(1)`` is mandatory per the guide — it guarantees the
current candle never participates in forming the very level it is measured
against, and no ``center=True`` is ever used.  The first ``lookback`` rows are
NaN: a level requires a full window of prior candles to exist.

Input frames must be chronologically ordered; a non-monotonic index raises so
a shuffled frame cannot silently leak future data into a level.
"""

from __future__ import annotations

import pandas as pd


def _validate_frame(df: pd.DataFrame, func: str, lookback: int) -> None:
    """Validate that ``df`` is a non-empty, sorted frame with high/low."""
    missing = [c for c in ("high", "low") if c not in df.columns]
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
    if lookback < 1:
        raise ValueError(f"{func}: lookback must be >= 1, got {lookback}")


def add_rolling_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Add causal ``liq_low`` / ``liq_high`` columns to a copy of ``df``.

    The input frame is returned unchanged (a copy is made) with the two level
    columns appended.  Values are ``shift(1).rolling(lookback).min()/max()`` —
    strictly causal, never centered.
    """
    out = df.copy()
    _validate_frame(out, "add_rolling_liquidity_levels", lookback)

    out["liq_low"] = out["low"].shift(1).rolling(lookback).min()
    out["liq_high"] = out["high"].shift(1).rolling(lookback).max()
    return out


def rolling_level_at(df: pd.DataFrame, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """Return ``(liq_low, liq_high)`` causal rolling levels as Series.

    Convenience accessor for callers that only need the level series (e.g. the
    sweep detector).  Same causality contract as
    :func:`add_rolling_liquidity_levels`.
    """
    _validate_frame(df, "rolling_level_at", lookback)
    liq_low = df["low"].shift(1).rolling(lookback).min()
    liq_high = df["high"].shift(1).rolling(lookback).max()
    liq_low.name = "liq_low"
    liq_high.name = "liq_high"
    return liq_low, liq_high