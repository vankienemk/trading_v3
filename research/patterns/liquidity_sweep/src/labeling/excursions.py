"""Fixed-horizon price excursions after an event (guide section 15).

For each horizon ``H`` (baseline 4/8/16/32 M15 bars = 1/2/4/8 hours) the module
measures, over the window starting at the entry candle (``entry_pos``) and
covering ``H`` bars (``entry_pos .. entry_pos + H - 1`` inclusive — the same
window convention as the triple-barrier walk in
:mod:`src.labeling.triple_barrier`, guide 16.2)::

    MFE  = max(future_high)  - entry   (long)   | entry - min(future_low)  (short)
    MAE  = entry - min(future_low)      (long)   | max(future_high) - entry (short)
    MFE_R = MFE / risk
    MAE_R = MAE / risk
    close_return_R     = (close[end] - entry) / risk          (signed)
    max_close_return_R = max over window of (close - entry) / risk
    min_close_return_R = min over window of (close - entry) / risk

These are *fixed-window* excursions: they ignore the TP/SL barriers and simply
describe what the price did inside the horizon, so they are comparable across
events and horizons.  The window is truncated when the frame ends; the caller
can detect that via ``n_bars < horizon`` / ``window_truncated``.  No data
beyond the horizon or beyond the end of the frame is ever read.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.labeling.triple_barrier import (
    DIRECTION_LONG,
    VALID_DIRECTIONS,
    validate_label_frame,
)


def measure_excursions_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_pos: int,
    horizon: int,
    direction: str,
    entry_price: float,
    risk: float,
) -> dict[str, Any] | None:
    """Fixed-window excursion metrics from pre-extracted float arrays.

    Batch fast path used by :func:`measure_excursions` and by the labeling
    pipeline.  Returns ``None`` when ``risk`` is non-finite / non-positive
    (the caller drops those events, guide 14.4).  ``entry_pos`` past the end
    of the frame is guarded by the caller (arrays can be shorter than the
    frame used for position lookup only in misconfigured pipelines).
    """
    if entry_pos >= len(highs):
        return None
    if not math.isfinite(float(risk)) or float(risk) <= 0:
        return None

    end_pos = min(entry_pos + horizon - 1, len(highs) - 1)
    w_high = highs[entry_pos : end_pos + 1]
    w_low = lows[entry_pos : end_pos + 1]
    w_close = closes[entry_pos : end_pos + 1]

    entry = float(entry_price)
    if direction == DIRECTION_LONG:
        mfe = float(np.max(w_high)) - entry
        mae = entry - float(np.min(w_low))
        close_deltas = w_close - entry
    else:
        mfe = entry - float(np.min(w_low))
        mae = float(np.max(w_high)) - entry
        close_deltas = entry - w_close

    risk = float(risk)
    close_return = float(close_deltas[-1])
    return {
        "entry_pos": int(entry_pos),
        "end_pos": int(end_pos),
        "n_bars": int(end_pos - entry_pos + 1),
        "window_truncated": bool(end_pos - entry_pos + 1 < horizon),
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_r": float(mfe / risk),
        "mae_r": float(mae / risk),
        "close_return": float(close_return),
        "max_close_return": float(np.max(close_deltas)),
        "min_close_return": float(np.min(close_deltas)),
        "close_return_r": float(close_return / risk),
        "max_close_return_r": float(np.max(close_deltas) / risk),
        "min_close_return_r": float(np.min(close_deltas) / risk),
    }


def measure_excursions(
    df: pd.DataFrame,
    entry_pos: int,
    horizon: int,
    direction: str,
    entry_price: float,
    risk: float,
) -> dict[str, Any] | None:
    """Fixed-window MFE/MAE and close-return metrics for one trade window.

    ``risk`` is the trade's risk (entry±stop distance, price units); pass the
    value from the trade-level computation.  Returns ``None`` when ``risk`` is
    non-finite / non-positive — the caller drops those events (guide 14.4).
    """
    if horizon < 1 or not isinstance(horizon, int):
        raise ValueError(
            f"measure_excursions: horizon must be an int >= 1, got {horizon}"
        )
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"measure_excursions: direction must be long|short, got {direction!r}"
        )
    if not isinstance(entry_pos, int) or not 0 <= entry_pos < len(df):
        raise ValueError(
            f"measure_excursions: entry_pos must be an int in [0, {len(df)}), "
            f"got {entry_pos!r}"
        )

    validate_label_frame(df, func="measure_excursions")
    return measure_excursions_arrays(
        highs=df["high"].to_numpy(dtype=float),
        lows=df["low"].to_numpy(dtype=float),
        closes=df["close"].to_numpy(dtype=float),
        entry_pos=entry_pos,
        horizon=horizon,
        direction=direction,
        entry_price=entry_price,
        risk=risk,
    )


def excursion_column_names(horizon: int) -> list[str]:
    """Wide-dataset column names for one horizon's excursion metrics."""
    return [
        f"mfe_r_h{horizon}",
        f"mae_r_h{horizon}",
        f"close_return_r_h{horizon}",
        f"max_close_return_r_h{horizon}",
        f"min_close_return_r_h{horizon}",
    ]
