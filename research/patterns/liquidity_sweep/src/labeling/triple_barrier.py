"""Triple-barrier labeling and per-event trade simulation (guide sections 14-17).

Entry / stop / target
---------------------
For a sweep event at candle ``event_pos`` (the sweep candle, whose close time
equals ``event_time``):

* entry = open of the candle *after* the sweep (baseline, Phase 2) — the caller
  may override ``entry_pos`` when confirmation-based entry is used (Phase 5);
* stop  = sweep extreme ± ``stop_buffer_atr * ATR`` at the sweep candle
  (long: ``low - buffer*ATR``, short: ``high + buffer*ATR``, guide 14.2/14.3);
* risk  = ``entry - stop`` (long) or ``stop - entry`` (short); an event with
  non-finite or ``risk <= 0`` is invalid and must be dropped (guide 14.4);
* target = ``entry ± reward_r * risk`` (guide 14.5).

Triple-barrier walk (guide 16.2)
-------------------------------
Starting at ``entry_pos`` and scanning at most ``horizon`` candles
(``end_pos = entry_pos + horizon - 1`` inclusive — the *entry candle is the
first of the ``horizon`` bars*, exactly like the guide's ``label_long_event``),
each candle updates MFE/MAE first and then checks:

* long:  stop hit when ``low <= stop``, target hit when ``high >= target``;
* short: stop hit when ``high >= stop``, target hit when ``low <= target``.

Labels (guide 16.1)::

    1    = target hit before stop        exit_reason "target"
    0    = stop hit before target        exit_reason "stop"
    -1   = time barrier (neither hit)    exit_reason "time"
    NaN  = TP and SL both hit inside one candle (policy "ambiguous")

Same-bar policy (guide 16.3): the default ``ambiguous`` marks the event
``outcome = NaN`` / ``exit_reason = "ambiguous"`` and keeps it out of binary
training; ``conservative`` assumes stop first, ``optimistic`` assumes target
first.  ``time_barrier_result`` controls the simulated exit when the horizon
ends: ``mark_to_market`` (exit at the last candle's close) or ``zero`` (exit at
entry price).  M15 data cannot reveal intrabar order, so nothing here tries to.

Numeric codes vs. canonical tokens
----------------------------------
This module is the numeric-label core (1/0/-1/NaN, mirroring the guide's
pseudo-code).  The canonical stored dataset (``outcome_*`` columns,
``src/schema.py::OUTCOME_VALUES``) uses the tokens ``tp/sl/time/ambiguous``;
``outcome_to_token`` / ``token_to_outcome`` are the exact bijection.

Fill assumptions (auditable, guide 17): barrier exits fill exactly at the
stop/target price, time-barrier exits at close (mark-to-market) or at entry
(zero); the guide's cost model (``costs`` in ``outcome_builder``) is applied on
top of this raw geometry.  No intrabar order is invented; no look-ahead past
the horizon or past the end of the frame occurs.

Decision-bar causality (QA finding F1): the caller supplies ``event_pos`` /
``event_time``; labeling anchors the trade to the open of the candle after it
and never scans earlier bars of the same sweep run.  Run-representative dedup
rules (``deepest_penetration`` / ``strongest_reclaim``) can make that bar a
*hindsight* bar mid-run; for causal backtests generate events with
``group_rule="first"`` so the labeled decision bar equals the first sweep
bar of the run (see :mod:`src.labeling.outcome_builder`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

DIRECTION_LONG = "long"
DIRECTION_SHORT = "short"
VALID_DIRECTIONS = (DIRECTION_LONG, DIRECTION_SHORT)

EXIT_REASON_TARGET = "target"
EXIT_REASON_STOP = "stop"
EXIT_REASON_TIME = "time"
EXIT_REASON_AMBIGUOUS = "ambiguous"
EXIT_REASONS = (
    EXIT_REASON_TARGET,
    EXIT_REASON_STOP,
    EXIT_REASON_TIME,
    EXIT_REASON_AMBIGUOUS,
)

#: Numeric triple-barrier labels (guide 16.1); NaN == ambiguous.
OUTCOME_TARGET = 1.0
OUTCOME_STOP = 0.0
OUTCOME_TIME = -1.0

SAME_BAR_POLICIES = ("ambiguous", "conservative", "optimistic")
TIME_BARRIER_RESULTS = ("mark_to_market", "zero")

#: Canonical stored tokens (``src/schema.py`` ``OUTCOME_VALUES``) for the
#: numeric labels; NaN maps to "ambiguous" (handled by the helpers below).
OUTCOME_TO_TOKEN: Mapping[float, str] = {
    OUTCOME_TARGET: "tp",
    OUTCOME_STOP: "sl",
    OUTCOME_TIME: "time",
}
TOKEN_AMBIGUOUS = "ambiguous"

_OHLC = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Public helpers: numeric label <-> canonical token
# ---------------------------------------------------------------------------

def outcome_to_token(outcome: float) -> str:
    """Map a numeric label (1/0/-1/NaN) to its canonical token."""
    if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
        return TOKEN_AMBIGUOUS
    try:
        return OUTCOME_TO_TOKEN[float(outcome)]
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - guard
        raise ValueError(
            f"outcome_to_token: unknown numeric label {outcome!r}; expected "
            f"1 (tp), 0 (sl), -1 (time) or NaN (ambiguous)"
        ) from exc


def token_to_outcome(token: str) -> float:
    """Inverse of :func:`outcome_to_token`; ambiguous maps to ``NaN``."""
    for code, tok in OUTCOME_TO_TOKEN.items():
        if tok == token:
            return code
    if token == TOKEN_AMBIGUOUS:
        return float("nan")
    raise ValueError(
        f"token_to_outcome: unknown token {token!r}; expected one of "
        f"{list(OUTCOME_TO_TOKEN.values())} or {TOKEN_AMBIGUOUS!r}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_label_frame(df: pd.DataFrame, func: str = "labeling") -> None:
    """Validate a candle frame that labeling walks over.

    Requires non-empty, chronologically sorted (monotonic index) OHLC data
    plus an ``atr`` column (ATR at the sweep candle drives the stop distance).
    An unsorted frame raises because positions are row indexes and a shuffled
    frame would silently relabel the wrong candles.
    """
    missing = [c for c in (*_OHLC, "atr") if c not in df.columns]
    if missing:
        raise ValueError(f"{func}: missing required columns {missing}")
    if len(df) == 0:
        raise ValueError(f"{func}: empty DataFrame")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{func}: index must be chronologically sorted (monotonic "
            "increasing); labeling resolves positions by row index"
        )
    non_numeric = [
        c for c in (*_OHLC, "atr") if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise TypeError(f"{func}: columns {non_numeric} must be numeric")


def compute_trade_levels(
    direction: str,
    entry_price: float,
    sweep_extreme: float,
    atr_value: float,
    reward_r: float = 2.0,
    stop_buffer_atr: float = 0.10,
) -> dict[str, float]:
    """Entry/stop/target/risk for one trade (guide 14.1-14.5).

    ``sweep_extreme`` is the sweep candle's extreme (low for long, high for
    short).  Returns ``{"stop", "target", "risk"}`` with raw prices; ``risk``
    may be ``NaN`` (missing ATR) or ``<= 0`` (invalid event, guide 14.4) — the
    caller decides how to treat it.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"compute_trade_levels: direction must be long|short, got {direction!r}"
        )
    if reward_r <= 0:
        raise ValueError(
            f"compute_trade_levels: reward_r must be > 0, got {reward_r}"
        )
    if stop_buffer_atr < 0:
        raise ValueError(
            f"compute_trade_levels: stop_buffer_atr must be >= 0, "
            f"got {stop_buffer_atr}"
        )

    if math.isnan(float(atr_value)):
        return {"stop": float("nan"), "target": float("nan"), "risk": float("nan")}

    if direction == DIRECTION_LONG:
        stop = float(sweep_extreme) - stop_buffer_atr * float(atr_value)
        risk = float(entry_price) - stop
        target = float(entry_price) + reward_r * risk
    else:
        stop = float(sweep_extreme) + stop_buffer_atr * float(atr_value)
        risk = stop - float(entry_price)
        target = float(entry_price) - reward_r * risk
    return {"stop": stop, "target": target, "risk": risk}


# ---------------------------------------------------------------------------
# Core triple-barrier walk (array fast path, shared by single-event API and
# the batch builder in outcome_builder.py)
# ---------------------------------------------------------------------------

def label_event_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    times: pd.Index,
    event_pos: int,
    direction: str,
    entry_pos: int,
    horizon: int = 16,
    stop_buffer_atr: float = 0.10,
    reward_r: float = 2.0,
    same_bar_policy: str = "ambiguous",
    time_barrier_result: str = "mark_to_market",
) -> dict[str, Any] | None:
    """Triple-barrier simulation from pre-extracted float arrays.

    Batch fast path used by :func:`label_event` and by the labeling pipeline:
    numpy column arrays (one per OHLC column plus ``atr``) plus the frame's
    index as ``times``.  Semantics are identical to :func:`label_event`;
    validation is the caller's responsibility.  Returns ``None`` for invalid
    trades (entry past the end, non-finite or non-positive risk).
    """
    if entry_pos >= len(highs):
        return None

    atr_value = float(atr[event_pos])
    if direction == DIRECTION_LONG:
        sweep_extreme = float(lows[event_pos])
    else:
        sweep_extreme = float(highs[event_pos])

    entry = float(opens[entry_pos])
    levels = compute_trade_levels(
        direction,
        entry_price=entry,
        sweep_extreme=sweep_extreme,
        atr_value=atr_value,
        reward_r=reward_r,
        stop_buffer_atr=stop_buffer_atr,
    )
    stop = levels["stop"]
    target = levels["target"]
    risk = levels["risk"]
    if math.isnan(risk) or not math.isfinite(risk) or risk <= 0:
        return None

    end_pos = min(entry_pos + horizon - 1, len(highs) - 1)
    mfe = 0.0
    mae = 0.0
    outcome: float = OUTCOME_TIME
    exit_reason = EXIT_REASON_TIME
    exit_pos = end_pos
    ambiguous = False

    for pos in range(entry_pos, end_pos + 1):
        high = float(highs[pos])
        low = float(lows[pos])
        if direction == DIRECTION_LONG:
            mfe = max(mfe, high - entry)
            mae = max(mae, entry - low)
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            mfe = max(mfe, entry - low)
            mae = max(mae, high - entry)
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop and hit_target:
            if same_bar_policy == "ambiguous":
                outcome = float("nan")
                exit_reason = EXIT_REASON_AMBIGUOUS
                ambiguous = True
            elif same_bar_policy == "conservative":
                outcome = OUTCOME_STOP
                exit_reason = EXIT_REASON_STOP
            else:  # optimistic: target assumed first
                outcome = OUTCOME_TARGET
                exit_reason = EXIT_REASON_TARGET
            exit_pos = pos
            break
        if hit_stop:
            outcome = OUTCOME_STOP
            exit_reason = EXIT_REASON_STOP
            exit_pos = pos
            break
        if hit_target:
            outcome = OUTCOME_TARGET
            exit_reason = EXIT_REASON_TARGET
            exit_pos = pos
            break

    if exit_reason == EXIT_REASON_TARGET:
        exit_price = target
    elif exit_reason == EXIT_REASON_STOP:
        exit_price = stop
    elif exit_reason == EXIT_REASON_AMBIGUOUS:
        exit_price = float("nan")
    elif time_barrier_result == "zero":
        exit_price = entry
    else:  # mark_to_market
        exit_price = float(closes[exit_pos])

    return {
        "event_pos": int(event_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "risk": float(risk),
        "outcome": float(outcome),
        "exit_reason": exit_reason,
        "ambiguous": bool(ambiguous),
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_r": float(mfe / risk),
        "mae_r": float(mae / risk),
        "bars_held": int(exit_pos - entry_pos + 1),
        "exit_time": times[exit_pos],
        "exit_price": float(exit_price),
    }


def label_event(
    df: pd.DataFrame,
    event_pos: int,
    direction: str,
    horizon: int = 16,
    stop_buffer_atr: float = 0.10,
    reward_r: float = 2.0,
    *,
    same_bar_policy: str = "ambiguous",
    time_barrier_result: str = "mark_to_market",
    entry_pos: int | None = None,
) -> dict[str, Any] | None:
    """Triple-barrier label for one sweep event (either direction).

    ``event_pos`` is the row position of the *sweep candle*; its extreme (low
    for long / high for short) and its ATR anchor the stop.  ``entry_pos``
    defaults to ``event_pos + 1`` (open of the candle after the sweep,
    baseline) but may be overridden for confirmation-based entry.

    Returns ``None`` for events that cannot be traded: entry bar past the end
    of the frame, or non-finite / non-positive risk (guide 14.4).  Every other
    result carries ``outcome`` ∈ {1, 0, -1, NaN} plus the matching
    ``exit_reason`` ∈ {target, stop, time, ambiguous}, ``ambiguous``,
    ``exit_time`` / ``exit_price``, MFE/MAE (price and R), and ``entry/stop/
    target`` raw prices.
    """
    if horizon < 1 or not isinstance(horizon, int):
        raise ValueError(f"label_event: horizon must be an int >= 1, got {horizon}")
    if same_bar_policy not in SAME_BAR_POLICIES:
        raise ValueError(
            f"label_event: same_bar_policy must be one of {SAME_BAR_POLICIES}, "
            f"got {same_bar_policy!r}"
        )
    if time_barrier_result not in TIME_BARRIER_RESULTS:
        raise ValueError(
            f"label_event: time_barrier_result must be one of "
            f"{TIME_BARRIER_RESULTS}, got {time_barrier_result!r}"
        )

    validate_label_frame(df, func="label_event")
    if not isinstance(event_pos, int) or not 0 <= event_pos < len(df):
        raise ValueError(
            f"label_event: event_pos must be an int in [0, {len(df)}), "
            f"got {event_pos!r}"
        )

    resolved_entry_pos = entry_pos if entry_pos is not None else event_pos + 1
    if not isinstance(resolved_entry_pos, int) or resolved_entry_pos <= event_pos:
        raise ValueError(
            f"label_event: entry_pos must be an int > event_pos "
            f"({event_pos}), got {entry_pos!r}"
        )

    return label_event_arrays(
        highs=df["high"].to_numpy(dtype=float),
        lows=df["low"].to_numpy(dtype=float),
        opens=df["open"].to_numpy(dtype=float),
        closes=df["close"].to_numpy(dtype=float),
        atr=df["atr"].to_numpy(dtype=float),
        times=df.index,
        event_pos=event_pos,
        direction=direction,
        entry_pos=resolved_entry_pos,
        horizon=horizon,
        stop_buffer_atr=stop_buffer_atr,
        reward_r=reward_r,
        same_bar_policy=same_bar_policy,
        time_barrier_result=time_barrier_result,
    )


def label_long_event(
    df: pd.DataFrame,
    event_pos: int,
    horizon: int = 16,
    stop_buffer_atr: float = 0.10,
    reward_r: float = 2.0,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Guide section 16.2's ``label_long_event`` (numeric labels, default 2R/16)."""
    return label_event(
        df,
        event_pos,
        DIRECTION_LONG,
        horizon=horizon,
        stop_buffer_atr=stop_buffer_atr,
        reward_r=reward_r,
        **kwargs,
    )


def label_short_event(
    df: pd.DataFrame,
    event_pos: int,
    horizon: int = 16,
    stop_buffer_atr: float = 0.10,
    reward_r: float = 2.0,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Mirror of :func:`label_long_event` for bearish (short) sweeps."""
    return label_event(
        df,
        event_pos,
        DIRECTION_SHORT,
        horizon=horizon,
        stop_buffer_atr=stop_buffer_atr,
        reward_r=reward_r,
        **kwargs,
    )
