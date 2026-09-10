"""Unified causal liquidity-level registry (schema §4) + previous-day levels.

Phase 4 (guide ``huong_dan.md`` Phase 4 / §9.2 / §9.3) upgrades the level
stack from rolling levels to a full registry combining:

- confirmed swing levels (M15), ``known_at = t + right_bars``;
- H1 swing levels (closed H1 candles only), ``known_at`` mapped to the first
  M15 bar at/after the confirming H1 candle's close;
- equal highs/lows clusters, ``known_at`` = the touch that completes the
  cluster;
- previous-day high/low levels, ``known_at`` = the first bar of the next
  trading day.

Causality contract
------------------
Every level carrries ``known_at``/``known_pos``; the sweep detector may only
use a level for bars with ``timestamp >= known_at``.  The final registry table
describes each level's *lifecycle as of the last bar of ``df``* (final
``touch_count``, ``age_bars``, ``sweep_state``).  For causal per-bar decisions
(events, features) use :func:`level_state_at_bar`, which recomputes each
level's state using only bars ``<= bar_pos`` — never future data.

Level lifecycle policy (documented defaults)
--------------------------------------------
- ``touch_count``: cumulative touches since the level's origin (swing/equal)
  or since it became known (previous-day).  Equal levels also seed the exact
  cluster positions recorded by :func:`detect_equal_levels`.
- ``sweep_state``: ``active`` until the price is penetrated beyond the level
  within ``touch_tolerance_atr x ATR`` (``swept``), or until age
  ``> max_age_bars`` without a sweep (``invalidated``).  ``swept`` is sticky.
- ``status`` (locked schema enum): ``active`` | ``expired`` — ``expired``
  whenever ``sweep_state != active``.

Schema: the first columns are exactly ``src.schema.LEVEL_COLUMNS``; the
remaining columns are documented extensions of the level table.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.indicators.atr import add_atr
from src.liquidity.equal_levels import detect_equal_levels
from src.liquidity.swing_levels import detect_h1_swing_levels, detect_swing_levels
from src.schema import (
    LEVEL_COLUMNS,
    LEVEL_EXTENSION_COLUMNS,
    LEVEL_REGISTRY_COLUMNS,
    LEVEL_STATE_COLUMNS,
)

#: Documented registry extension columns (schema lock v1.1, src.schema).
EXTENSION_COLUMNS: list[str] = LEVEL_EXTENSION_COLUMNS
#: Full registry column set = schema §4 + §4.1 extensions.
REGISTRY_COLUMNS: list[str] = LEVEL_REGISTRY_COLUMNS
#: Output columns of :func:`level_state_at_bar` (schema §4.2).
STATE_COLUMNS: list[str] = LEVEL_STATE_COLUMNS


def _validate_frame(df: pd.DataFrame, func: str) -> None:
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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{func}: index must be a DatetimeIndex")
    non_numeric = [c for c in ("high", "low") if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"{func}: columns {non_numeric} must be numeric")


# ---------------------------------------------------------------------------
# Previous-day levels
# ---------------------------------------------------------------------------

def detect_previous_day_levels(
    df: pd.DataFrame,
    max_age_bars: int = 200,
) -> pd.DataFrame:
    """Detect previous-trading-day high/low levels, causally.

    For each trading day D after the first, two levels are created from day
    D-1's high and low::

        origin_pos = last bar of day D-1   (the day's high/low becomes final)
        known_pos  = first bar of day D    (the level is usable from here)

    Days are grouped by calendar date (UTC, matching the pipeline); 'previous
    day' means the previous date present in the data (previous *trading* day —
    e.g. Monday references Friday, consistent with
    ``src.data.resampler.previous_day_high_low``).  The first day has no
    previous day and contributes no levels.

    Returns a levels table with ``level_type="prev_day"``, ids
    ``prev_day_high_{origin_pos}`` / ``prev_day_low_{origin_pos}``
    """
    _validate_frame(df, "detect_previous_day_levels")
    if max_age_bars < 1:
        raise ValueError(
            f"detect_previous_day_levels: max_age_bars must be >= 1, got {max_age_bars}"
        )

    ts = df.index.to_numpy()
    n = len(df)
    day = cast(pd.DatetimeIndex, df.index).normalize()

    daily_high = df["high"].groupby(day).max()
    daily_low = df["low"].groupby(day).min()
    # Normalize every timestamp to pandas Timestamp.  Under numpy>=2 a
    # DatetimeIndex must not be converted through np.asarray(): that yields
    # np.datetime64 keys/scalars which no longer interoperate with the
    # Timestamp dict keys / DatetimeIndex.searchsorted queries below
    # (TypeError in np.searchsorted, KeyError in dict lookups).
    days: pd.DatetimeIndex = daily_high.index
    high_by_day = {pd.Timestamp(k): v for k, v in zip(days, daily_high.to_numpy())}
    low_by_day = {pd.Timestamp(k): v for k, v in zip(days, daily_low.to_numpy())}

    rows: list[dict] = []
    for k in range(1, len(days)):
        day_start = pd.Timestamp(days[k])
        first_pos = int(df.index.searchsorted(day_start, side="left"))
        if first_pos >= n:
            break
        origin_pos = first_pos - 1  # last bar of the previous trading day
        prev_day = pd.Timestamp(days[k - 1])
        prev_high = float(high_by_day[prev_day])
        prev_low = float(low_by_day[prev_day])
        rows.append(
            {
                "level_id": f"prev_day_high_{origin_pos}",
                "level_type": "prev_day",
                "direction": "high",
                "price": prev_high,
                "price_min": prev_high,
                "price_max": prev_high,
                "origin_pos": origin_pos,
                "origin_time": ts[origin_pos],
                "known_at": ts[first_pos],
                "status": "active",
                "touch_count": 0,
                "first_touch_time": pd.NaT,
                "last_touch_time": pd.NaT,
                "known_pos": first_pos,
                "is_h1": False,
                "max_age_bars": int(max_age_bars),
            }
        )
        rows.append(
            {
                "level_id": f"prev_day_low_{origin_pos}",
                "level_type": "prev_day",
                "direction": "low",
                "price": prev_low,
                "price_min": prev_low,
                "price_max": prev_low,
                "origin_pos": origin_pos,
                "origin_time": ts[origin_pos],
                "known_at": ts[first_pos],
                "status": "active",
                "touch_count": 0,
                "first_touch_time": pd.NaT,
                "last_touch_time": pd.NaT,
                "known_pos": first_pos,
                "is_h1": False,
                "max_age_bars": int(max_age_bars),
            }
        )

    out = pd.DataFrame(
        rows, columns=[*LEVEL_COLUMNS, "known_pos", "is_h1", "max_age_bars"]
    )
    if len(out):
        out = out.sort_values("known_at").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Lifecycle computation
# ---------------------------------------------------------------------------

def _lifecycle_state(level: dict, low, high, atr_arr, ts, limit_pos: int) -> dict | None:
    """Compute a level's causal state using only bars ``<= limit_pos``.

    Returns None when the level is not yet known at ``limit_pos`` (its
    ``known_pos > limit_pos``) — such a level does not exist for decision
    making at that bar.
    """
    direction = level["direction"]
    price = float(level["price"])
    tol_atr = float(level.get("touch_tolerance_atr", 0.0))
    known_pos = int(level["known_pos"])
    origin_pos = int(level["origin_pos"])
    max_age = int(level["max_age_bars"])
    level_type = level["level_type"]

    if known_pos > limit_pos:
        return None

    # --- touches ----------------------------------------------------------
    seed = None
    if level_type == "equal":
        raw_seed = level.get("touch_positions")
        if raw_seed:
            seed = np.asarray(raw_seed, dtype=np.int64)
        count_from = known_pos + 1  # exclude cluster members (already seeded)
    elif level_type == "prev_day":
        count_from = known_pos  # touches only after the level is known
    else:
        count_from = origin_pos  # swing: origin touch is the level itself
        seed = level.get("touch_positions")  # normally None

    if count_from <= limit_pos:
        sl = slice(count_from, limit_pos + 1)
        if direction == "low":
            mask = low[sl] <= price + tol_atr * atr_arr[sl]
        else:
            mask = high[sl] >= price - tol_atr * atr_arr[sl]
        recount = np.flatnonzero(mask) + count_from
    else:
        recount = np.zeros(0, dtype=np.int64)

    if seed is not None and len(seed):
        # Seed positions may interleave with the recount range only at the
        # boundary known_pos (count_from = known_pos + 1 avoids doubling).
        touches = np.unique(np.concatenate([seed, recount]))
    else:
        touches = np.unique(recount)

    # --- sweeps (strict penetration, only once the level is usable) -------
    sl = slice(known_pos, limit_pos + 1)
    if direction == "low":
        smask = low[sl] < price - tol_atr * atr_arr[sl]
    else:
        smask = high[sl] > price + tol_atr * atr_arr[sl]
    sweep_pos = np.flatnonzero(smask) + known_pos

    age = int(limit_pos - origin_pos)
    first_sweep = int(sweep_pos[0]) if len(sweep_pos) else None

    if first_sweep is not None:
        sweep_state, status = "swept", "expired"
        invalidated_pos = None
    elif age > max_age:
        sweep_state, status = "invalidated", "expired"
        invalidated_pos = origin_pos + max_age + 1
    else:
        sweep_state, status = "active", "active"
        invalidated_pos = None

    return {
        "touch_count": len(touches),
        "first_touch_pos": touches[0] if len(touches) else None,
        "last_touch_pos": touches[-1] if len(touches) else None,
        "age_bars": age,
        "bars_since_last_touch": (
            float(limit_pos - int(touches[-1])) if len(touches) else np.nan
        ),
        "sweep_state": sweep_state,
        "status": status,
        "first_sweep_pos": first_sweep,
        "first_swept_at": ts[first_sweep] if first_sweep is not None else pd.NaT,
        "invalidated_at": (
            ts[invalidated_pos] if invalidated_pos is not None and invalidated_pos <= limit_pos else pd.NaT
        ),
    }


def _clean_touch_positions(value) -> list | None:
    """Normalize a touch_positions cell (list for equal rows, else None)."""
    return value if isinstance(value, list) else None


def _finalize_registry(
    levels: pd.DataFrame,
    df: pd.DataFrame,
    atr: pd.Series,
    equal_tolerance_atr: float,
) -> pd.DataFrame:
    """Apply the lifecycle view to the concatenated level tables."""
    n = len(df)
    limit = n - 1
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    atr_arr = atr.to_numpy(dtype="float64")
    ts = df.index.to_numpy()

    # Touch tolerance: equal levels use the clustering tolerance; everything
    # else interacts with the exact price (tolerance 0).
    levels = levels.copy()
    levels["touch_tolerance_atr"] = np.where(
        levels["level_type"] == "equal", equal_tolerance_atr, 0.0
    )
    if "touch_positions" not in levels.columns:
        levels["touch_positions"] = pd.Series(
            [None] * len(levels), index=levels.index, dtype=object
        )

    rows: list[dict] = []
    for _, level in levels.iterrows():
        base = {
            "level_id": level["level_id"],
            "level_type": level["level_type"],
            "direction": level["direction"],
            "price": level["price"],
            "price_min": level["price_min"],
            "price_max": level["price_max"],
            "origin_pos": level["origin_pos"],
            "origin_time": level["origin_time"],
            "known_at": level["known_at"],
            "known_pos": int(level["known_pos"]),
            "is_h1": bool(level["is_h1"]),
            "max_age_bars": int(level["max_age_bars"]),
            "touch_tolerance_atr": float(level["touch_tolerance_atr"]),
            "touch_positions": _clean_touch_positions(level["touch_positions"]),
        }
        st = _lifecycle_state(base, low, high, atr_arr, ts, limit)
        if st is None:  # cannot happen: known_pos <= n-1 for emitted rows
            continue
        known_pos = base["known_pos"]
        eq_disp = np.nan
        form_tol = np.nan
        if base["level_type"] == "equal" and np.isfinite(atr_arr[known_pos]):
            eq_disp = (float(level["price_max"]) - float(level["price_min"])) / float(atr_arr[known_pos])
            form_tol = float(base["touch_tolerance_atr"]) * float(atr_arr[known_pos])
        rows.append(
            {
                **base,
                "status": st["status"],
                "touch_count": st["touch_count"],
                "first_touch_time": (
                    ts[st["first_touch_pos"]] if st["first_touch_pos"] is not None else pd.NaT
                ),
                "last_touch_time": (
                    ts[st["last_touch_pos"]] if st["last_touch_pos"] is not None else pd.NaT
                ),
                "age_bars": st["age_bars"],
                "bars_since_last_touch": st["bars_since_last_touch"],
                "sweep_state": st["sweep_state"],
                "first_swept_at": st["first_swept_at"],
                "invalidated_at": st["invalidated_at"],
                "touch_tolerance_atr": float(base["touch_tolerance_atr"]),
                "equal_dispersion_atr": float(eq_disp),
                "formation_atr_tolerance": float(form_tol),
            }
        )

    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    if len(out):
        out = out.sort_values(["known_at", "level_id"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Unified registry
# ---------------------------------------------------------------------------

def build_liquidity_levels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build the unified causal level registry (schema §4 + extensions).

    Consumes ``config`` (baseline.yaml shape):

    - ``indicators.atr_period`` (default 14) for the ATR series;
    - ``liquidity.swing.{enabled,left_bars,right_bars,max_age_bars}``;
    - ``liquidity.h1.{enabled,left_bars,right_bars,max_age_bars}``
      (enabled by default; falls back to swing params);
    - ``liquidity.equal_levels.{enabled,tolerance_atr,min_touches,max_age_bars}``;
    - ``liquidity.prev_day.{enabled,max_age_bars}`` (enabled by default).

    Only levels whose ``known_pos < len(df)`` are emitted (a level must be
    known in-sample to be usable).  Returns the schema-compatible registry
    table sorted by ``known_at``; every row carries ``known_at``.
    """
    _validate_frame(df, "build_liquidity_levels")

    indicators = config.get("indicators", {}) if config else {}
    liq = config.get("liquidity", {}) if config else {}

    atr_period = int(indicators.get("atr_period", 14))
    atr = add_atr(df, atr_period)["atr"]

    swing = liq.get("swing", {})
    equal = liq.get("equal_levels", {})
    h1 = liq.get("h1", {})
    prev = liq.get("prev_day", {})

    parts = []
    if swing.get("enabled", True):
        parts.append(
            detect_swing_levels(
                df,
                left_bars=int(swing.get("left_bars", 3)),
                right_bars=int(swing.get("right_bars", 3)),
                max_age_bars=int(swing.get("max_age_bars", 200)),
            )
        )
    if h1.get("enabled", True):
        parts.append(
            detect_h1_swing_levels(
                df,
                left_bars=int(h1.get("left_bars", swing.get("left_bars", 3))),
                right_bars=int(h1.get("right_bars", swing.get("right_bars", 3))),
                max_age_bars=int(h1.get("max_age_bars", 200)),
            )
        )
    if equal.get("enabled", True):
        parts.append(
            detect_equal_levels(
                df,
                tolerance_atr=float(equal.get("tolerance_atr", 0.10)),
                min_touches=int(equal.get("min_touches", 2)),
                max_age_bars=int(equal.get("max_age_bars", 200)),
                atr=atr,
            )
        )
    if prev.get("enabled", True):
        parts.append(
            detect_previous_day_levels(
                df,
                max_age_bars=int(prev.get("max_age_bars", 200)),
            )
        )

    if not parts:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    parts = [p for p in parts if len(p)]  # drop empty sections (e.g. prev-day
    # on a single-day frame) so concat dtype inference stays stable
    if not parts:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="level_id", keep="first")
    equal_tolerance_atr = float(equal.get("tolerance_atr", 0.10))
    return _finalize_registry(combined, df, atr, equal_tolerance_atr)


# ---------------------------------------------------------------------------
# Per-bar causal state
# ---------------------------------------------------------------------------

def level_state_at_bar(
    registry: pd.DataFrame,
    df: pd.DataFrame,
    atr: pd.Series,
    bar_pos: int,
) -> pd.DataFrame:
    """Causal state of every level as of bar ``bar_pos`` (bars ``0..bar_pos``).

    Levels with ``known_pos > bar_pos`` are excluded — they do not exist for
    decision making at that bar.  Returns one row per *known* level with:
    ``level_id, level_type, direction, price, known_at, touch_count,
    last_touch_time, age_bars, bars_since_last_touch, sweep_state, status,
    first_swept_at, invalidated_at``.

    This is the no-look-ahead interface for the sweep detector and features:
    each value uses only bars ``<= bar_pos``.
    """
    if not isinstance(bar_pos, (int, np.integer)) or bar_pos < 0:
        raise ValueError(f"level_state_at_bar: bar_pos must be >= 0, got {bar_pos}")
    if bar_pos >= len(df):
        raise ValueError(
            f"level_state_at_bar: bar_pos {bar_pos} out of range (frame has {len(df)} bars)"
        )
    if len(atr) != len(df):
        raise ValueError(
            f"level_state_at_bar: ATR length {len(atr)} != frame length {len(df)}"
        )

    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    atr_arr = atr.to_numpy(dtype="float64")
    ts = df.index.to_numpy()

    rows: list[dict] = []
    for _, level in registry.iterrows():
        base = {
            "level_id": level["level_id"],
            "level_type": level["level_type"],
            "direction": level["direction"],
            "price": level["price"],
            "origin_pos": level["origin_pos"],
            "origin_time": level["origin_time"],
            "known_at": level["known_at"],
            "known_pos": int(level["known_pos"]),
            "max_age_bars": int(level["max_age_bars"]),
            "touch_tolerance_atr": float(level["touch_tolerance_atr"]),
            "touch_positions": _clean_touch_positions(level["touch_positions"]),
        }
        st = _lifecycle_state(base, low, high, atr_arr, ts, bar_pos)
        if st is None:
            continue
        rows.append(
            {
                "level_id": base["level_id"],
                "level_type": base["level_type"],
                "direction": base["direction"],
                "price": base["price"],
                "known_at": base["known_at"],
                "touch_count": st["touch_count"],
                "last_touch_time": (
                    ts[st["last_touch_pos"]] if st["last_touch_pos"] is not None else pd.NaT
                ),
                "age_bars": st["age_bars"],
                "bars_since_last_touch": st["bars_since_last_touch"],
                "sweep_state": st["sweep_state"],
                "status": st["status"],
                "first_swept_at": st["first_swept_at"],
                "invalidated_at": st["invalidated_at"],
            }
        )

    return pd.DataFrame(rows, columns=STATE_COLUMNS)