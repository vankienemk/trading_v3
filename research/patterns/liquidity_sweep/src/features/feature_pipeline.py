"""Causal event-level feature pipeline (guide section 13, INTERFACES.md §5).

One row per sweep event; every feature is a function of data known at the
event's *decision* bar:

* ``available_at="event_time"`` features use candles up to and including the
  sweep candle (``event_time``) — the sweep detector only fires after the
  candle closes;
* ``available_at="confirmation_time"`` features use candles up to and including
  the confirmation candle (only populated for confirmed events; NaN otherwise).

No-look-ahead guarantees
------------------------
* ATR / ATR-percentile / volatility regime are causal rolling statistics
  (:mod:`src.indicators.atr`).
* Volume z-score excludes the current bar's own volume
  (:mod:`src.indicators.volume`).
* Higher-timeframe context uses only *closed* H1 candles / the prior completed
  day (:mod:`src.features.htf`).
* Liquidity-level state comes from
  :func:`src.liquidity.level_registry.level_state_at_bar` at the event bar
  (recomputed using bars ``<= bar_pos`` only).
* Session flags are pure functions of the event's own timestamp.

The output column set is locked to ``configs/features.yaml`` (rule 30.4 / schema
§7): the table must carry ``event_id`` + ``event_time`` and **exactly** the
registered feature names.  Any divergence raises
:class:`src.features.registry.FeatureRegistryError`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.atr import add_atr, atr_percentile_causal, volatility_regime_causal
from src.indicators.volume import volume_percentile_causal, volume_zscore
from src.liquidity.level_registry import level_state_at_bar

from .htf import build_htf_context
from .registry import FeatureRegistryError, load_features
from .sessions import session_flags

_VALID_LEVEL_TYPES = ("rolling", "swing", "equal", "prev_day")

#: Max distance (in ATR) between the swept level and the nearest structural
#: level for the nearest-level attribution heuristic to apply.
_LEVEL_MATCH_MAX_ATR = 2.0


class FeaturePipelineError(ValueError):
    """Raised on invalid inputs to the feature pipeline."""


def _validate_candles(df: pd.DataFrame) -> None:
    missing = [
        c for c in ("open", "high", "low", "close", "volume") if c not in df.columns
    ]
    if missing:
        raise FeaturePipelineError(f"candles missing required columns {missing}")
    if len(df) == 0:
        raise FeaturePipelineError("candles is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise FeaturePipelineError("candles index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise FeaturePipelineError(
            "candles index must be chronologically sorted (monotonic increasing); "
            "an unsorted frame breaks causal (no-look-ahead) semantics"
        )
    non_numeric = [
        c
        for c in ("open", "high", "low", "close")
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise TypeError(f"candles columns {non_numeric} must be numeric")


def _validate_events(events: pd.DataFrame) -> None:
    missing = [
        c for c in ("event_id", "event_time", "direction") if c not in events.columns
    ]
    if missing:
        raise FeaturePipelineError(f"events table missing required columns {missing}")
    if len(events) == 0:
        return


def _per_bar_context(
    candles: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.Series]:
    """Causal per-bar statistics needed to describe the event bar.

    Returns a dict of index-aligned Series (ATR, ATR-percentile, volatility
    regime, volume z-score / percentile, H1 trend, previous-day high/low).
    """
    ind = config.get("indicators", {}) if config else {}
    atr_period = int(ind.get("atr_period", 14))
    vol_win = int(ind.get("volume_zscore_window", 50))

    atr_frame = add_atr(candles, period=atr_period)
    atr = atr_frame["atr"]
    atr_pct = atr_percentile_causal(atr, window=200)
    vol_regime = volatility_regime_causal(atr_pct)
    vol_z = volume_zscore(candles["volume"], window=vol_win)
    vol_pct = volume_percentile_causal(candles["volume"], window=200)

    htf = build_htf_context(candles)
    return {
        "atr": atr,
        "atr_percentile": atr_pct,
        "volatility_regime": vol_regime,
        "volume_zscore": vol_z,
        "volume_percentile": vol_pct,
        "h1_trend": htf["h1_trend"],
        "prev_day_high": htf["prev_day_high"],
        "prev_day_low": htf["prev_day_low"],
    }


def _time_features(
    ts: pd.Timestamp, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hour / day-of-week / session features from an event timestamp (UTC)."""
    hour_utc = int(ts.hour)
    d = int(ts.dayofweek)
    day_of_week = d if d < 5 else d - 5  # 0=Mon..4=Fri, 5=Sat, 6=Sun -> weekend
    return {
        "hour_utc": hour_utc,
        "day_of_week": day_of_week,
        **session_flags_from_hour(hour_utc, config),
    }


_SESSION_FLAG_NAMES = ("session_asia", "session_new_york", "session_london")


def session_flags_from_hour(
    hour_utc: int, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return only the registered ``session_*`` boolean flags for an hour."""
    all_flags = session_flags(hour_utc, config)
    return {k: all_flags[k] for k in _SESSION_FLAG_NAMES if k in all_flags}


def _rolling_level_default() -> dict[str, Any]:
    """Level feature values for a baseline rolling event (no registry match)."""
    return {
        "level_type": "rolling",
        "level_age_bars": np.nan,
        "level_touch_count": np.nan,
        "bars_since_last_touch": np.nan,
        "equal_level_dispersion_atr": np.nan,
        "level_is_previous_day_high_low": False,
        "level_is_h1_swing": False,
        "level_already_partially_swept": False,
    }


def _level_features(
    levels: pd.DataFrame,
    df: pd.DataFrame,
    atr: pd.Series,
    bar_pos: int,
    level_id: str,
    level_price: float,
    direction: str,
) -> dict[str, Any]:
    """Causal liquidity-level features resolved at the event bar.

    Side = ``low`` for a bullish sweep (a support level was swept), ``high`` for
    a bearish sweep.  Prefers an exact ``level_id`` match in the registry; else
    picks the known level of the same side whose price is nearest the swept
    level; fails back to the rolling default when no structural level applies.

    The per-bar causal state (age/touch/sweep) is computed via
    ``level_state_at_bar`` on a *single*-level subset of the registry, so cost
    is O(1) per event instead of O(levels).  Input ``levels`` is never mutated.
    """
    side = "low" if direction == "bullish" else "high"
    atr_value = float(atr.iloc[bar_pos]) if np.isfinite(atr.iloc[bar_pos]) else np.nan

    if len(levels) == 0:
        return _rolling_level_default()

    # --- 1. Resolve the candidate level (vectorised, causal). ------------
    known = levels[(levels["known_pos"] <= bar_pos)]
    if known.empty:
        return _rolling_level_default()

    exact = known[known["level_id"] == level_id]
    if not exact.empty:
        cand = exact
    else:
        same_side = known[known["direction"] == side]
        if same_side.empty:
            return _rolling_level_default()
        diffs = (same_side["price"].astype(float) - float(level_price)).abs()
        nearest_idx = diffs.idxmin()
        # Guard: only associate the event with a structural level we actually
        # probed; skip the nearest-level heuristic for implausibly far matches.
        if (
            np.isfinite(atr_value)
            and atr_value > 0
            and diffs.min() > _LEVEL_MATCH_MAX_ATR * atr_value
        ):
            return _rolling_level_default()
        cand = same_side.loc[[nearest_idx]]

    # --- 2. Static attributes from the registry row. ----------------------
    lrow = cand.iloc[0]
    is_h1 = bool(lrow.get("is_h1", False))
    disp = lrow.get("equal_dispersion_atr")
    dispersion = float(disp) if disp is not None and np.isfinite(float(disp)) else np.nan

    # --- 3. Causal per-bar state on the single candidate level. -----------
    row = None
    level_state = level_state_at_bar(cand, df, atr, bar_pos)
    if level_state.empty:
        # The candidate is not yet usable at this bar; default to rolling.
        return _rolling_level_default()
    row = level_state.iloc[0]

    pre_swept = False
    if bar_pos > 0:
        pre_state = level_state_at_bar(cand, df, atr, bar_pos - 1)
        if not pre_state.empty:
            pre_swept = pre_state.iloc[0]["sweep_state"] != "active"

    return {
        "level_type": str(row["level_type"]),
        "level_age_bars": float(row["age_bars"]),
        "level_touch_count": float(row["touch_count"]),
        "bars_since_last_touch": float(row["bars_since_last_touch"]),
        "equal_level_dispersion_atr": dispersion,
        "level_is_previous_day_high_low": bool(row["level_type"] == "prev_day"),
        "level_is_h1_swing": is_h1,
        "level_already_partially_swept": pre_swept,
    }


def _safe_div(num: float, den: float) -> float:
    den = float(den)
    if not np.isfinite(den) or den == 0.0:
        return np.nan
    return float(num) / den


def _event_feature_row(
    event: pd.Series,
    df: pd.DataFrame,
    ctx: dict[str, pd.Series],
    levels: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    pos = df.index.get_indexer([event["event_time"]])
    bar_pos = int(pos[0]) if pos[0] >= 0 else None
    if bar_pos is None:
        raise FeaturePipelineError(
            f"event_time {event['event_time']} not present in candles index"
        )

    direction = str(event["direction"])
    # Trade-consistent side / whether the level is a support (low) or resistance (high).
    atr = (
        float(ctx["atr"].iloc[bar_pos])
        if np.isfinite(ctx["atr"].iloc[bar_pos])
        else np.nan
    )

    e_open, e_high, e_low, e_close = (
        float(event["event_open"]),
        float(event["event_high"]),
        float(event["event_low"]),
        float(event["event_close"]),
    )
    rng = e_high - e_low
    body = abs(e_close - e_open)

    row: dict[str, Any] = {}

    # --- Candle features ---------------------------------------------------
    row["range_atr"] = _safe_div(rng, atr)
    row["body_ratio"] = _safe_div(body, rng)
    lower_wick = min(e_open, e_close) - e_low
    upper_wick = e_high - max(e_open, e_close)
    row["lower_wick_ratio"] = _safe_div(lower_wick, rng)
    row["upper_wick_ratio"] = _safe_div(upper_wick, rng)
    row["close_location"] = _safe_div(e_close - e_low, rng)

    # --- Sweep features ----------------------------------------------------
    row["penetration_atr"] = float(event["penetration_atr"])
    row["reclaim_atr"] = float(event["reclaim_atr"])
    row["wick_ratio"] = float(event["wick_ratio"])
    row["sweep_volume_zscore"] = (
        float(ctx["volume_zscore"].iloc[bar_pos])
        if np.isfinite(ctx["volume_zscore"].iloc[bar_pos])
        else np.nan
    )

    # --- Liquidity level features -----------------------------------------
    lvl = _level_features(
        levels,
        df,
        ctx["atr"],
        bar_pos,
        str(event["level_id"]),
        float(event["level_price"]),
        direction,
    )
    row.update(lvl)

    # --- Volatility features ----------------------------------------------
    row["atr_percentile"] = (
        float(ctx["atr_percentile"].iloc[bar_pos])
        if np.isfinite(ctx["atr_percentile"].iloc[bar_pos])
        else np.nan
    )
    regime = ctx["volatility_regime"].iloc[bar_pos]
    row["volatility_regime"] = str(regime) if pd.notna(regime) else np.nan

    # --- Volume features ---------------------------------------------------
    row["volume_zscore"] = (
        float(ctx["volume_zscore"].iloc[bar_pos])
        if np.isfinite(ctx["volume_zscore"].iloc[bar_pos])
        else np.nan
    )

    # --- Time / session features ------------------------------------------
    row.update(_time_features(event["event_time"], config))

    # --- Higher-timeframe context -----------------------------------------
    h1_trend = ctx["h1_trend"].iloc[bar_pos]
    row["h1_trend"] = float(h1_trend) if np.isfinite(h1_trend) else np.nan
    pdh = ctx["prev_day_high"].iloc[bar_pos]
    pdl = ctx["prev_day_low"].iloc[bar_pos]
    row["distance_to_previous_day_high_atr"] = (
        _safe_div(e_close - float(pdh), atr) if np.isfinite(pdh) else np.nan
    )
    row["distance_to_previous_day_low_atr"] = (
        _safe_div(e_close - float(pdl), atr) if np.isfinite(pdl) else np.nan
    )

    # --- Confirmation features (only when confirmed) -----------------------
    confirmed = bool(event.get("is_confirmed", False))
    if confirmed and pd.notna(event.get("confirmation_time", pd.NaT)):
        row["confirmation_delay_bars"] = (
            float(event["confirmation_delay_bars"])
            if pd.notna(event.get("confirmation_delay_bars"))
            else np.nan
        )
        row["confirmation_range_atr"] = (
            float(event["confirmation_range_atr"])
            if pd.notna(event.get("confirmation_range_atr"))
            else np.nan
        )
        row["confirmation_body_ratio"] = (
            float(event["confirmation_body_ratio"])
            if pd.notna(event.get("confirmation_body_ratio"))
            else np.nan
        )
        conf_ts = event["confirmation_time"]
        # Defensive tz handling: the confirmation column may be tz-naive when
        # the confirmation stage produced all-NaT placeholders (t9 finding N3).
        if (
            hasattr(conf_ts, "tzinfo")
            and conf_ts.tzinfo is None
            and isinstance(df.index, pd.DatetimeIndex)
        ):
            conf_ts = conf_ts.tz_localize(df.index.tz)
        conf_bar = df.index.get_indexer([conf_ts])
        cpos = int(conf_bar[0]) if conf_bar[0] >= 0 else None
        if cpos is not None:
            cvol = float(ctx["volume_zscore"].iloc[cpos])
            row["confirmation_volume_zscore"] = cvol if np.isfinite(cvol) else np.nan
        else:
            row["confirmation_volume_zscore"] = np.nan
    else:
        row["confirmation_delay_bars"] = np.nan
        row["confirmation_range_atr"] = np.nan
        row["confirmation_body_ratio"] = np.nan
        row["confirmation_volume_zscore"] = np.nan

    return row


def build_event_features(
    candles: pd.DataFrame,
    levels: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the causal event-level feature matrix (INTERFACES.md §5).

    Parameters
    ----------
    candles:
        Sorted OHLCV frame (DatetimeIndex, UTC) the events were detected on.
    levels:
        Liquidity-level registry table (schema §4 + extensions); may be empty.
    events:
        Event table from ``build_sweep_events`` (with the confirmation columns
        from ``attach_confirmations`` when confirmation is used).
    config:
        Merged config (``configs/baseline.yaml`` shape); optional.

    Returns
    -------
    ``DataFrame`` with one row per event, keyed by ``event_id`` + ``event_time``
    and carrying exactly the registered feature columns.  Confirmation features
    are NaN for unconfirmed events (``available_at`` semantics).
    """
    cfg = dict(config or {})
    _validate_candles(candles)
    _validate_events(events)
    registry = load_features()
    registered = {f["name"]: f["available_at"] for f in registry}

    if events.empty:
        out = pd.DataFrame(
            {
                "event_id": pd.Series(dtype="object"),
                "event_time": pd.Series(dtype="object"),
            }
        )
        for name in registered:
            out[name] = pd.Series(dtype="float64")
        return out

    ctx = _per_bar_context(candles, cfg)

    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        feat = _event_feature_row(event, candles, ctx, levels, cfg)
        feat["event_id"] = event["event_id"]
        feat["event_time"] = event["event_time"]
        rows.append(feat)

    out = pd.DataFrame(rows).set_index("event_id", drop=False)
    out = out.reset_index(drop=True)

    # Column order: event_key columns first, then registered features in order.
    cols = ["event_id", "event_time"] + [f["name"] for f in registry]
    out = out[cols]

    # Registry conformance: no unregistered column may leak into the matrix.
    actual = set(out.columns)
    expected = set(["event_id", "event_time", *registered])
    extra = actual - expected
    if extra:
        raise FeatureRegistryError(
            f"feature pipeline produced unregistered columns {sorted(extra)}"
        )
    missing = expected - actual
    if missing:
        raise FeatureRegistryError(
            f"feature pipeline is missing registered columns {sorted(missing)}"
        )

    # Enforce category column dtypes as plain str (schema §7).
    for f in registry:
        if f["dtype"] == "category":
            out[f["name"]] = out[f["name"]].astype("object")

    return out


__all__ = ["FeaturePipelineError", "build_event_features"]
