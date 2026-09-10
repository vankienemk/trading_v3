"""Rule-based scoring for liquidity sweep events (guide section 19).

Scores each event 0-100 with a breakdown into six components:

| Component                | Max points | Config key                 |
|--------------------------|-----------:|----------------------------|
| Liquidity level quality  |         20 | ``scoring.weights.level``  |
| Sweep quality            |         25 | ``scoring.weights.sweep``  |
| Reclaim                  |         15 | ``scoring.weights.reclaim``|
| Confirmation             |         20 | ``scoring.weights.confirmation`` |
| Higher-timeframe context |         10 | ``scoring.weights.context``|
| Volume & volatility      |         10 | ``scoring.weights.volume`` |

Weights are loaded from config; **never hard-coded** in source.

Output columns (per guide §19.3):
    score_level, score_sweep, score_reclaim, score_confirmation,
    score_context, score_volume, rule_score

Bucket evaluation (§19.4):
    Buckets 0-39 / 40-49 / 50-59 / 60-69 / 70-79 / 80-100 -> report
    event_count, win_rate, loss_rate, ambiguous_rate, avg_mfe_r,
    avg_mae_r, avg_net_result_r, profit_factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ScoringError(ValueError):
    """Raised when scoring inputs are invalid."""


# ---------------------------------------------------------------------------
# Weight defaults — overridden by config (NEVER used directly in production)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "level": 20,
    "sweep": 25,
    "reclaim": 15,
    "confirmation": 20,
    "context": 10,
    "volume": 10,
}

SCORE_COMPONENTS = list(_DEFAULT_WEIGHTS.keys())
TOTAL_MAX = sum(_DEFAULT_WEIGHTS.values())  # 100

# Bucket boundaries per guide §19.4
SCORE_BUCKETS = [
    (0, 39, "0-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 69, "60-69"),
    (70, 79, "70-79"),
    (80, 100, "80-100"),
]


def _get_weights(config: dict[str, Any]) -> dict[str, int]:
    """Load scoring weights from config, falling back to defaults.

    Config path: ``config["scoring"]["weights"]`` with keys matching
    SCORE_COMPONENTS. Missing keys use defaults; extra keys are ignored.
    """
    weights = dict(_DEFAULT_WEIGHTS)  # start with defaults
    scoring_cfg = config.get("scoring", {})
    cfg_weights = scoring_cfg.get("weights", {})
    for key in SCORE_COMPONENTS:
        if key in cfg_weights:
            val = cfg_weights[key]
            if isinstance(val, (int, float)) and val >= 0:
                weights[key] = int(val)
    return weights


def _validate_inputs(df: pd.DataFrame) -> None:
    """Check that the event/feature table has required columns."""
    required = ["event_id", "direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScoringError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise ScoringError("Input dataframe is empty")


def _score_level(row: pd.Series, levels_df: pd.DataFrame | None = None) -> float:
    """Score liquidity level quality (0-max_level_points).

    Criteria (guide §19.2):
    - Clear equal highs/lows: +8
    - 3 or more touches: +4
    - H1 or previous-day level: +5
    - Level never swept before: +3

    Returns partial score (will be scaled by weight later).
    """
    score = 0.0
    max_points = 20  # reference max for this component

    # Check level_type for equal levels
    level_type = row.get("level_type", "")
    if level_type == "equal":
        score += 8

    # Check touch count (need registry data)
    touch_count = row.get("touch_count", 0)
    if pd.notna(touch_count) and touch_count >= 3:
        score += 4

    # Check if H1 or previous-day level
    is_h1 = row.get("is_h1", False)
    is_prev_day = row.get("level_is_previous_day_high_low", False)
    if is_h1 or is_prev_day:
        score += 5

    # Check if level was already swept (age/freshness heuristic)
    # If bars_since_last_touch is NaN, level was never touched before
    bars_since = row.get("bars_since_last_touch", np.nan)
    if pd.isna(bars_since):
        score += 3

    return min(score, max_points)


def _score_sweep(row: pd.Series) -> float:
    """Score sweep quality (0-max_sweep_points).

    Criteria (guide §19.2):
    - Penetration within optimal zone: +8
    - Large wick ratio: +7
    - Large range relative to ATR: +5
    - Close location favors direction: +5
    """
    score = 0.0
    max_points = 25

    # Penetration in optimal zone (0.05-0.50 ATR per guide §10)
    pen_atr = row.get("penetration_atr", 0)
    if pd.notna(pen_atr):
        if 0.10 <= pen_atr <= 0.35:  # sweet spot
            score += 8
        elif 0.05 <= pen_atr < 0.10 or 0.35 < pen_atr <= 0.50:
            score += 5  # acceptable but not optimal

    # Wick ratio (guide §10.3: min 0.35)
    wick_ratio = row.get("wick_ratio", 0)
    if pd.notna(wick_ratio):
        if wick_ratio >= 0.70:
            score += 7
        elif wick_ratio >= 0.50:
            score += 5
        elif wick_ratio >= 0.35:
            score += 3

    # Range relative to ATR (use range_atr if available, else estimate)
    range_atr = row.get("range_atr", np.nan)
    if pd.isna(range_atr):
        # Estimate from OHLC if available
        if all(c in row.index for c in ["event_open", "event_high", "event_low"]):
            atr_val = row.get("atr", 1.0)
            if pd.notna(atr_val) and atr_val > 0:
                candle_range = abs(row["event_high"] - row["event_low"])
                range_atr = candle_range / atr_val

    if pd.notna(range_atr):
        if range_atr >= 2.0:
            score += 5
        elif range_atr >= 1.5:
            score += 3
        elif range_atr >= 1.0:
            score += 2

    # Close location favors direction
    direction = str(row.get("direction", "")).lower()
    if all(c in row.index for c in ["event_open", "event_close", "event_high", "event_low"]):
        close_p = row["event_close"]
        high_p = row["event_high"]
        low_p = row["event_low"]
        candle_range = high_p - low_p
        if candle_range > 0:
            if direction == "long":  # bullish sweep -> want close near high
                close_position = (close_p - low_p) / candle_range
                if close_position >= 0.7:
                    score += 5
                elif close_position >= 0.5:
                    score += 3
            elif direction == "short":  # bearish sweep -> want close near low
                close_position = (high_p - close_p) / candle_range
                if close_position >= 0.7:
                    score += 5
                elif close_position >= 0.5:
                    score += 3

    return min(score, max_points)


def _score_reclaim(row: pd.Series) -> float:
    """Score reclaim quality (0-max_reclaim_points).

    Criteria (guide §19.2):
    - Close reclaims the level: +5
    - Reclaim >= 0.10 ATR: +5
    - Reclaim >= 0.25 ATR: +5
    """
    score = 0.0
    max_points = 15

    reclaim_atr = row.get("reclaim_atr", 0)
    if pd.notna(reclaim_atr):
        # Base: any positive reclaim
        if reclaim_atr > 0:
            score += 5

        # Tiered reclaim strength
        if reclaim_atr >= 0.25:
            score += 10  # both tiers
        elif reclaim_atr >= 0.10:
            score += 5

    return min(score, max_points)


def _score_confirmation(row: pd.Series) -> float:
    """Score confirmation quality (0-max_confirmation_points).

    Criteria (guide §19.2):
    - Confirmation within 1-3 candles: +5
    - Displacement present: +5
    - Structure break present: +5
    - High confirmation volume: +5
    """
    score = 0.0
    max_points = 20

    # Confirmation delay (1-3 bars is ideal per guide §11)
    delay = row.get("confirmation_delay_bars", np.nan)
    is_confirmed = row.get("is_confirmed", False)

    if is_confirmed and pd.notna(delay):
        if 1 <= delay <= 3:
            score += 5
        elif delay == 0:
            score += 3  # immediate but possibly noisy

    # Confirmation type: structure_break is strongest
    conf_type = row.get("confirmation_type", "")
    if conf_type == "structure_break":
        score += 5
    elif conf_type == "displacement":
        score += 5
    elif conf_type == "close_break":
        score += 3

    # Confirmation strength (if available from t9)
    conf_strength = row.get("confirmation_strength", np.nan)
    if pd.notna(conf_strength) and conf_strength >= 0.7:
        score += 5
    elif pd.notna(conf_strength) and conf_strength >= 0.4:
        score += 3

    # Confirmation volume (relative z-score or raw)
    conf_vol = row.get("confirmation_volume", np.nan)
    if pd.notna(conf_vol):
        vol_z = row.get("volume_zscore", np.nan)
        if pd.notna(vol_z) and vol_z > 1.0:
            score += 5
        elif pd.notna(vol_z) and vol_z > 0.5:
            score += 3
        elif conf_vol > 0:
            score += 2  # some volume at least

    return min(score, max_points)


def _score_context(row: pd.Series) -> float:
    """Score higher-timeframe context (0-max_context_points).

    Criteria (inferred from guide §13.5 HTF features):
    - H1 trend alignment: +5
    - Distance to previous day high/low: +3
    - Session timing (London/NY overlap): +2
    """
    score = 0.0
    max_points = 10

    # H1 trend alignment
    h1_trend = row.get("h1_trend", "")
    direction = str(row.get("direction", "")).lower()
    if h1_trend == "up" and direction == "long":
        score += 5
    elif h1_trend == "down" and direction == "short":
        score += 5
    elif h1_trend in ("up", "down"):
        score += 2  # has trend but misaligned

    # Distance to PDH/PDL (if available)
    dist_pdh = row.get("distance_to_previous_day_high_atr", np.nan)
    dist_pdl = row.get("distance_to_previous_day_low_atr", np.nan)
    if pd.notna(dist_pdh) and pd.notna(dist_pdl):
        # Being near a key daily level adds context
        if min(abs(dist_pdh), abs(dist_pdl)) < 0.5:
            score += 3
        elif min(abs(dist_pdh), abs(dist_pdl)) < 1.0:
            score += 2

    # Session timing
    session_london = row.get("session_london", False)
    session_ny = row.get("session_new_york", False)
    if session_london and session_ny:  # overlap period
        score += 2
    elif session_london or session_ny:
        score += 1

    return min(score, max_points)


def _score_volume(row: pd.Series) -> float:
    """Score volume and volatility characteristics (0-max_volume_points).

    Criteria:
    - Volume z-score: +5
    - Volume percentile: +3
    - Volatility regime: +2
    """
    score = 0.0
    max_points = 10

    # Volume z-score
    vol_z = row.get("volume_zscore", np.nan)
    if pd.notna(vol_z):
        if vol_z > 2.0:
            score += 5
        elif vol_z > 1.0:
            score += 3
        elif vol_z > 0:
            score += 1

    # Volume percentile
    vol_pct = row.get("volume_percentile", np.nan)
    if pd.notna(vol_pct):
        if vol_pct >= 0.8:
            score += 3
        elif vol_pct >= 0.6:
            score += 2

    # Volatility regime
    vol_regime = row.get("volatility_regime", "")
    if vol_regime == "high":
        score += 2  # sweeps often better in high vol
    elif vol_regime == "normal":
        score += 1

    return min(score, max_points)


def compute_rule_scores(
    events_df: pd.DataFrame,
    config: dict[str, Any],
    levels_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute rule-based scores for all events.

    Parameters
    ----------
    events_df : DataFrame
        Event table with sweep + confirmation + feature columns.
        Must contain at minimum: event_id, direction, penetration_atr,
        wick_ratio, reclaim_atr, is_confirmed.
    config : dict
        Merged configuration dict (from load_config).
    levels_df : DataFrame, optional
        Level registry for additional level context (touch_count, etc.).

    Returns
    -------
    DataFrame
        Original events_df with added score columns:
        score_level, score_sweep, score_reclaim, score_confirmation,
        score_context, score_volume, rule_score
    """
    _validate_inputs(events_df)
    weights = _get_weights(config)
    total_weight = sum(weights.values())

    # Work on a copy to avoid modifying original
    df = events_df.copy()

    # Compute component scores
    scores_level = []
    scores_sweep = []
    scores_reclaim = []
    scores_confirmation = []
    scores_context = []
    scores_volume = []

    for _, row in df.iterrows():
        scores_level.append(_score_level(row, levels_df))
        scores_sweep.append(_score_sweep(row))
        scores_reclaim.append(_score_reclaim(row))
        scores_confirmation.append(_score_confirmation(row))
        scores_context.append(_score_context(row))
        scores_volume.append(_score_volume(row))

    df["score_level"] = scores_level
    df["score_sweep"] = scores_sweep
    df["score_reclaim"] = scores_reclaim
    df["score_confirmation"] = scores_confirmation
    df["score_context"] = scores_context
    df["score_volume"] = scores_volume

    # Compute weighted total (normalize to 100)
    df["rule_score"] = (
        df["score_level"] * weights["level"] / 20
        + df["score_sweep"] * weights["sweep"] / 25
        + df["score_reclaim"] * weights["reclaim"] / 15
        + df["score_confirmation"] * weights["confirmation"] / 20
        + df["score_context"] * weights["context"] / 10
        + df["score_volume"] * weights["volume"] / 10
    ) * (100 / total_weight)

    # Clip to 0-100
    df["rule_score"] = df["rule_score"].clip(0, 100).round(2)

    return df


def assign_score_bucket(score: float) -> str:
    """Assign a score bucket label per guide §19.4."""
    for low, high, label in SCORE_BUCKETS:
        if low <= score <= high:
            return label
    return "unknown"


def compute_bucket_report(
    scored_events: pd.DataFrame,
) -> pd.DataFrame:
    """Generate performance report by score bucket.

    Requires outcome columns: outcome_2r_h16, mfe_r_h16, mae_r_h16,
    net_result_r (from labeling pipeline).

    Returns DataFrame with columns:
        bucket, event_count, win_rate, loss_rate, ambiguous_rate,
        avg_mfe_r, avg_mae_r, avg_net_result_r, profit_factor
    """
    required_outcome_cols = ["outcome_2r_h16", "mfe_r_h16", "mae_r_h16", "net_result_r"]
    missing = [c for c in required_outcome_cols if c not in scored_events.columns]
    if missing:
        raise ScoringError(
            f"Cannot compute bucket report: missing outcome columns {missing}. "
            "Run labeling pipeline first."
        )

    df = scored_events.copy()
    df["bucket"] = df["rule_score"].apply(assign_score_bucket)

    records = []
    for bucket_label in [b[2] for b in SCORE_BUCKETS]:
        bucket_df = df[df["bucket"] == bucket_label]
        n = len(bucket_df)
        if n == 0:
            continue

        outcomes = bucket_df["outcome_2r_h16"]
        wins = (outcomes == "tp").sum()
        losses = (outcomes == "sl").sum()
        ambiguous = (outcomes == "ambiguous").sum()

        win_rate = wins / n if n > 0 else 0
        loss_rate = losses / n if n > 0 else 0
        ambiguous_rate = ambiguous / n if n > 0 else 0

        avg_mfe = bucket_df["mfe_r_h16"].mean()
        avg_mae = bucket_df["mae_r_h16"].mean()
        avg_net = bucket_df["net_result_r"].mean()

        # Profit factor = gross_profit / gross_loss (in R units)
        gross_profit = bucket_df[bucket_df["net_result_r"] > 0]["net_result_r"].sum()
        gross_loss = abs(bucket_df[bucket_df["net_result_r"] < 0]["net_result_r"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        records.append({
            "bucket": bucket_label,
            "event_count": n,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "ambiguous_rate": round(ambiguous_rate, 4),
            "avg_mfe_r": round(avg_mfe, 4) if pd.notna(avg_mfe) else None,
            "avg_mae_r": round(avg_mae, 4) if pd.notna(avg_mae) else None,
            "avg_net_result_r": round(avg_net, 4) if pd.notna(avg_net) else None,
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        })

    return pd.DataFrame(records)
