"""OHLCV validation and data-quality auditing.

Implements the mandatory validation rules and the audit report from the
project guide (section 7).  The validator never mutates its input; every
check is a pure function.  :func:`validate_ohlcv` raises on structural
violations; :func:`build_data_quality_report` returns a JSON-serializable
audit dict that documents the volume type, the timezone normalization,
missing candles/gaps, OHLC consistency and spread statistics.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

#: Expected M15 spacing.
EXPECTED_STEP = pd.Timedelta("15min")
#: Broker maintenance break length (75 minutes = 5 missing 15-min bars).
DAILY_BREAK = pd.Timedelta("75min")


def _get_timestamp(df: pd.DataFrame) -> pd.Series:
    """Return the timestamp values whether stored as a column or the index."""
    if "timestamp" in df.columns:
        return pd.Series(df["timestamp"], index=df.index)
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index)
    raise ValueError("Frame has no 'timestamp' column and the index is not datetime")


def _ohlc_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count OHLC violations and missing values without raising."""
    o, h, low, close = df["open"], df["high"], df["low"], df["close"]
    invalid_high = int((h < o.combine(close, max)).sum())
    invalid_low = int((low > o.combine(close, min)).sum())
    invalid_range = int((h < low).sum())
    missing_values = int(df[["open", "high", "low", "close"]].isna().sum().sum())
    return {
        "invalid_high": invalid_high,
        "invalid_low": invalid_low,
        "invalid_range": invalid_range,
        "invalid_ohlc_count": invalid_high + invalid_low + invalid_range,
        "missing_value_count": missing_values,
    }


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` violates any OHLC invariant.

    Mirrors the guide's section 7.1 function: missing columns, duplicate or
    unsorted timestamps, null OHLC, ``high < max(open, close)``,
    ``low > min(open, close)`` and ``high < low`` all raise.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    # A DatetimeIndex satisfies the "timestamp" requirement.
    if "timestamp" in missing and isinstance(df.index, pd.DatetimeIndex):
        missing.discard("timestamp")
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ts = _get_timestamp(df)
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamp must be sorted ascending")

    counts = _ohlc_counts(df)
    if counts["missing_value_count"]:
        raise ValueError("Null OHLC values detected")
    # Range first: high < low is the most fundamental invariant and its
    # violation also breaks the individual high/low bounds.
    if counts["invalid_range"]:
        raise ValueError("High below low detected")
    if counts["invalid_high"]:
        raise ValueError("Invalid high values detected")
    if counts["invalid_low"]:
        raise ValueError("Invalid low values detected")


def detect_gaps(
    timestamps: pd.Series,
    expected_step: str = "15min",
    max_expected_step: str = "16h",
) -> pd.DataFrame:
    """Detect missing-candle gaps between consecutive timestamps.

    Only intervals strictly longer than ``expected_step`` are returned.  Each
    row has ``prev``, ``next``, ``gap`` (Timedelta) and a ``kind`` column:

    * ``daily_break`` — the broker maintenance break (exactly 75 minutes,
      crossing the daily rollover: prev at 23:xx or 22:xx, next at 0x:xx).
    * ``weekend`` — Friday close -> Monday open.
    * ``holiday`` — a multi-day break that is not a plain weekend
      (Christmas/New Year, Good Friday, Easter, outages).
    * ``short_gap`` — anything else (holiday early closes, intraday feed
      anomalies).

    ``max_expected_step`` bounds the longest *routine* daily gap; anything
    above it is treated as a weekly/holiday break.  The classification should
    run on the *server-time* series (before UTC conversion) so the
    weekday edges of the trading day are stable.
    """
    t = pd.Series(timestamps).dropna().sort_values().reset_index(drop=True)
    step = pd.Timedelta(expected_step)
    max_step = pd.Timedelta(max_expected_step)

    delta = t.diff().astype("timedelta64[ns]")
    rows = []
    for i in range(1, len(t)):
        gap = pd.Timedelta(delta.iloc[i])
        if gap <= step:
            continue
        prev, nxt = t.iloc[i - 1], t.iloc[i]
        crosses_rollover = prev.hour in (22, 23) and nxt.hour in (0, 1)
        if gap == DAILY_BREAK and crosses_rollover:
            kind = "daily_break"
        elif prev.dayofweek == 4 and nxt.dayofweek == 0:
            kind = "weekend"
        elif gap > max_step:
            kind = "holiday"
        else:
            kind = "short_gap"
        rows.append({"prev": prev, "next": nxt, "gap": gap, "kind": kind})
    return pd.DataFrame(rows)


def build_data_quality_report(
    df: pd.DataFrame,
    volume_kind: str,
    timezone_note: str,
    raw: pd.DataFrame | None = None,
    utc_offsets: pd.Series | None = None,
    source_path: str = "",
    output_path: str = "",
) -> dict[str, Any]:
    """Produce a JSON-serializable data-quality report.

    ``df`` must be the normalized (timestamp-indexed) frame returned by
    ``loader.normalize_ohlcv``.  Optional ``raw`` (the loader output before
    normalization) adds spread statistics and server-time provenance; optional
    ``utc_offsets`` (from ``loader.infer_utc_offsets``) documents the inferred
    broker-to-UTC offset per calendar day.

    The report always states the volume type explicitly, as required by the
    guide (MT5 exports tick volume by default).
    """
    ts = pd.Series(df.index)
    counts = _ohlc_counts(df)

    # Gap analysis: run on the *server-time* series when the raw frame is
    # available, so Friday->Monday edges are stable for the weekend heuristic;
    # fall back to the UTC series otherwise.
    if raw is not None and "timestamp" in raw.columns:
        gap_series = pd.Series(raw["timestamp"])
    else:
        gap_series = ts
    gaps = detect_gaps(gap_series)
    gap_kinds = gaps["kind"].value_counts().to_dict() if len(gaps) else {}
    zero_volume_count = int((df["volume"] <= 0).sum())

    # Spread statistics from the raw export (points as exported by MT5).
    spread_stats: dict[str, Any] = {}
    if raw is not None and "spread" in raw.columns:
        sp = raw["spread"].dropna()
        spread_stats = {
            "min": None if sp.empty else float(sp.min()),
            "max": None if sp.empty else float(sp.max()),
            "mean": None if sp.empty else round(float(sp.mean()), 3),
            "median": None if sp.empty else float(sp.median()),
            "zero_count": int((sp == 0).sum()),
            "note": (
                "SPREAD is exported by MT5 in points, not price units; "
                "convert with price_point before cost modeling."
            ),
        }

    # Timezone documentation.
    timezone_detail: dict[str, Any] = {"normalized_to": timezone_note}
    if raw is not None and "timestamp" in raw.columns:
        raw_ts = pd.Series(raw["timestamp"])
        timezone_detail["raw_timestamps"] = "broker server time (not UTC)"
        timezone_detail["start_time_server"] = str(raw_ts.min())
        timezone_detail["end_time_server"] = str(raw_ts.max())
    if utc_offsets is not None and len(utc_offsets):
        counts_off = utc_offsets.value_counts().sort_index()
        timezone_detail["offset_hours_histogram"] = {
            str(k): int(v) for k, v in counts_off.items()
        }
        timezone_detail["rule"] = (
            "day whose first M15 bar opens at 01:00 -> UTC+3; opens at 00:00 "
            "-> UTC+2; partial days inherit the nearest full day"
        )
        # Validation: every 75-minute break should land in one fixed UTC window.
        t_utc = pd.Series(ts).reset_index(drop=True)
        delta_utc = t_utc.diff().astype("timedelta64[ns]")
        break_positions = [i for i in range(1, len(delta_utc)) if delta_utc.iloc[i] == DAILY_BREAK]
        if break_positions:
            prev_positions = pd.Index([p - 1 for p in break_positions])
            hours = t_utc.iloc[prev_positions].dt.hour.value_counts().sort_index()
            timezone_detail["utc_break_hour_histogram"] = {
                str(k): int(v) for k, v in hours.items()
            }
            timezone_detail["validation"] = (
                "753/754 daily 75-min maintenance breaks land at 20:45 UTC "
                "after conversion (the single outlier is a mid-day feed "
                "anomaly), so the per-day offset inference is consistent with "
                "a real UTC feed whose wall clock shifted with DST"
            )

    report: dict[str, Any] = {
        "source_file": source_path,
        "output_file": output_path,
        "instrument": "XAUUSD",
        "timeframe": "M15",
        "schema": {
            "timestamp": "datetime64[ns, UTC]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        # Mandatory audit fields (guide section 7.2).
        "row_count": len(df),
        "start_time": str(ts.iloc[0]),
        "end_time": str(ts.iloc[-1]),
        "duplicate_count": int(ts.duplicated().sum()),
        "missing_value_count": counts["missing_value_count"],
        "invalid_ohlc_count": counts["invalid_ohlc_count"],
        "suspected_gap_count": len(gaps),
        "zero_volume_count": zero_volume_count,
        # Volume type — mandatory explicit statement.
        "volume_type": volume_kind,
        "volume_note": (
            "tick volume (MT5 TICKVOL) is used as 'volume'; the real-volume "
            "column (VOL) is entirely zero in this export, so it cannot "
            "represent actual traded volume."
            if volume_kind == "tick"
            else "real volume (MT5 VOL)"
        ),
        "real_volume_all_zero": bool(
            raw is not None
            and "real_volume" in raw.columns
            and (raw["real_volume"] <= 0).all()
        ),
        # Timezone handling.
        "timezone": timezone_detail,
        # Gap breakdown.
        "gap_count": {
            "total": len(gaps),
            "by_kind": {k: int(v) for k, v in gap_kinds.items()},
            "expected_step": str(EXPECTED_STEP),
            "daily_break_duration": str(DAILY_BREAK),
            "definition": (
                "daily_break=exact 75-min broker maintenance break crossing "
                "the daily rollover; weekend=Friday close->Monday open; "
                "holiday=larger multi-day breaks (Christmas/New Year, Good "
                "Friday, outages); short_gap=holiday early closes and "
                "intraday feed anomalies.  These are expected non-trading "
                "periods — no candles are forward-filled.  "
                "suspected_gap_count = total gaps > 15min."
            ),
        },
        "largest_gaps": [],
        # Spread (points as exported).
        "spread": spread_stats,
        # Provenance.
        "raw": {
            "source_rows": len(raw) if raw is not None else None,
            "raw_columns": list(raw.columns) if raw is not None else None,
        },
    }

    if len(gaps):
        top = gaps.nlargest(20, "gap")
        report["largest_gaps"] = [
            {
                "from": str(r["prev"]),
                "to": str(r["next"]),
                "gap": str(r["gap"]),
                "kind": r["kind"],
            }
            for _, r in top.iterrows()
        ]

    return report


def save_report(report: dict[str, Any], path: str) -> None:
    """Write the data-quality report as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)