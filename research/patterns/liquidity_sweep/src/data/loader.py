"""Loading, normalization and persistence of raw MT5 OHLCV data.

The primary input is an MT5-exported tab-separated CSV with the header::

    <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>

The loader provides:

* :func:`load_ohlcv` — parse a raw MT5 export into a normalized frame
  (naive ``timestamp`` in broker server time plus ``open/high/low/close`` and
  the MT5 extra columns ``tick_volume`` / ``real_volume`` / ``spread``).
* :func:`normalize_ohlcv` — promote a volume column, convert timestamps to
  UTC, sort, deduplicate and cast to the canonical float64 schema.
* :func:`save_parquet` / :func:`read_parquet` — persist/read the processed
  file with ``timestamp`` as an explicit column.
* :func:`infer_utc_offsets` — data-driven per-day broker-to-UTC offset
  inference (see timezone note below).

Timezone handling (broker server time -> UTC)
---------------------------------------------
MT5 exports are usually in *broker server time*, not UTC.  For this dataset
the server clock follows a hybrid DST regime: on days whose first M15 bar
opens at ``01:00`` the server is on UTC+3, on days whose first bar opens at
``00:00`` it is on UTC+2.  The evidence: after subtracting the inferred
per-day offset, every daily 75-minute maintenance break lands at a fixed
``20:45 -> 22:00`` UTC window (753/754 occurrences) and weekends/holidays
align to a clean weekly calendar — the signature of a real UTC feed whose
wall-clock hours merely shifted with DST.

:func:`normalize_ohlcv` therefore converts by default
(``utc_offset_hours="auto"``).  Pass ``0`` to treat file time as UTC
verbatim, or a constant number (e.g. ``2``) for a fixed offset.  The
inference rule and its validation are fully documented in
``reports/data_quality.json``.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from ..config import resolve_path

#: Canonical output column order for a normalized OHLCV frame.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: MT5 export header -> normalized column name.
_MT5_COLUMN_MAP = {
    "DATE": "date",
    "TIME": "time",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "TICKVOL": "tick_volume",
    "VOL": "real_volume",
    "SPREAD": "spread",
}

_AUTO_OFFSET = {"auto", "infer"}


def _detect_decimal_separator(s: pd.Series) -> str:
    """Infer the decimal separator of a numeric column from its values.

    Strategy: look at how many values end with ``sep + 2 digits`` (XAUUSD
    prices are quoted to 2 decimals).  Whichever of ``,`` / ``.`` appears in
    that position more often is the decimal separator; the other is the
    thousands separator.  Falls back to ``.`` when there is no evidence.
    """
    comma_decimal = s.str.contains(r",\d{2}$", regex=True).sum()
    dot_decimal = s.str.contains(r"\.\d{2}$", regex=True).sum()
    if comma_decimal > dot_decimal:
        return ","
    return "."


def _parse_locale_numbers(s: pd.Series, decimal_sep: str) -> pd.Series:
    """Parse a numeric text series using the detected decimal separator.

    The thousands separator is the opposite character; it is stripped, then
    the decimal separator is normalized to ``.`` before ``to_numeric``.
    """
    out = s.astype(str).str.replace(" ", "", regex=False)
    if decimal_sep == ",":
        out = out.str.replace(".", "", regex=False)  # dot = thousands
        out = out.str.replace(",", ".", regex=False)
    else:
        # Dot decimal: a comma anywhere is a thousands separator.
        out = out.str.replace(",", "", regex=False)
    return pd.to_numeric(out, errors="coerce")


def _read_mt5_csv(path: str) -> pd.DataFrame:
    """Read an MT5-exported CSV (tab-separated, angle-bracketed headers).

    All columns are read as text and parsed locale-aware: the decimal
    separator (comma or dot) is inferred per column from the price columns
    (guide section 7: separator formatting depends on the export locale).
    """
    raw = pd.read_csv(path, sep="\t", dtype=str)
    # Strip angle brackets from the MT5 header style.
    raw.columns = [c.strip().strip("<>") for c in raw.columns]

    missing = [c for c in ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"] if c not in raw.columns]
    if missing:
        raise ValueError(
            f"MT5 export is missing required columns {missing}; "
            f"found columns {list(raw.columns)}"
        )

    # Locale-aware numeric parsing: a mixed comma/dot column is disambiguated
    # per column, so "1,234.56" (US) and "1.234,56" (EU) both parse correctly
    # and volumes with thousands separators are handled.
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]:
        if col not in raw.columns:
            continue
        sep = _detect_decimal_separator(raw[col].astype(str).str.replace(" ", "", regex=False))
        raw[col] = _parse_locale_numbers(raw[col], sep)

    # DATE is exported as YYYY.MM.DD in MT5; TIME as HH:MM:SS.
    date_parsed = pd.to_datetime(raw["DATE"], format="%Y.%m.%d", errors="coerce")
    time_parsed = pd.to_timedelta(raw["TIME"], errors="coerce")
    timestamp = date_parsed + time_parsed

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": raw["OPEN"].astype("float64"),
            "high": raw["HIGH"].astype("float64"),
            "low": raw["LOW"].astype("float64"),
            "close": raw["CLOSE"].astype("float64"),
        }
    )
    for src, dst in (
        ("VOL", "real_volume"),
        ("TICKVOL", "tick_volume"),
        ("SPREAD", "spread"),
    ):
        if src in raw.columns:
            out[dst] = raw[src].astype("float64")
    return out


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load raw OHLCV data from ``path`` into a normalized DataFrame.

    The returned frame retains the MT5 extra columns (``real_volume``,
    ``tick_volume``, ``spread``) and a naive ``timestamp`` in broker server
    time.  Call :func:`normalize_ohlcv` before further processing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input data file not found: {path}")
    return _read_mt5_csv(path)


def infer_utc_offsets(timestamps: pd.Series) -> pd.Series:
    """Infer the per-calendar-day broker-to-UTC offset in hours.

    Rule: a day whose first M15 bar opens at ``01:00`` server time is on
    UTC+3; a day whose first bar opens at ``00:00`` is on UTC+2 (the server
    switches between these two regimes around US/EU DST transition dates).
    Partial days inherit the offset of the nearest full day.

    Returns a Series indexed by normalized calendar day (``Timestamp``) with
    float values ``2.0``/``3.0``.
    """
    ts = pd.Series(timestamps).reset_index(drop=True)
    day = ts.dt.normalize()
    first = ts.groupby(day).first()
    start_hour = first.dt.hour

    offsets = pd.Series(index=first.index, dtype="float64")
    offsets[start_hour == 1] = 3.0
    offsets[start_hour == 0] = 2.0
    # Days with a partial morning (e.g. first/last export-day, outages)
    # inherit the offset of the nearest full trading day.
    offsets = offsets.ffill().bfill()

    if offsets.isna().any():
        raise ValueError("Could not infer a UTC offset for every calendar day")
    return offsets


def normalize_ohlcv(
    df: pd.DataFrame,
    volume_kind: str = "tick",
    timezone: str = "UTC",
    utc_offset_hours: Any = "auto",
) -> pd.DataFrame:
    """Normalize a loaded frame to the canonical processed schema.

    Parameters
    ----------
    df:
        Frame returned by :func:`load_ohlcv`.
    volume_kind:
        ``"tick"`` or ``"real"`` — which volume column to promote to
        ``volume``.  MT5 exports use tick volume by default.
    timezone:
        Target timezone label for the converted timestamps.
    utc_offset_hours:
        ``"auto"`` (default) infers a per-day offset from the daily break
        pattern; ``0`` treats file time as UTC verbatim; a number applies a
        constant offset (e.g. ``2`` for a fixed UTC+2 server).
    """
    out = df.copy()

    # Promote the selected volume column to the canonical name.
    if volume_kind == "tick" and "tick_volume" in out.columns:
        out["volume"] = out["tick_volume"]
    elif "real_volume" in out.columns:
        out["volume"] = out["real_volume"]
    else:
        out["volume"] = 0.0

    # Convert broker server time -> target timezone.
    ts = out["timestamp"]
    if isinstance(utc_offset_hours, str) and utc_offset_hours.lower() in _AUTO_OFFSET:
        offsets = infer_utc_offsets(ts)
        offset_series = ts.dt.normalize().map(offsets).astype("float64")
        ts = ts - pd.to_timedelta(offset_series, unit="h")
    elif isinstance(utc_offset_hours, (int, float)) and utc_offset_hours:
        ts = ts - pd.Timedelta(hours=float(utc_offset_hours))

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone)
    out["timestamp"] = ts

    out = (
        out.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .set_index("timestamp")
    )
    out = out[OHLCV_COLUMNS]
    out = out.astype({c: "float64" for c in OHLCV_COLUMNS})
    return out


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a normalized DataFrame to Parquet, creating parent dirs.

    The frame is written with ``timestamp`` as an explicit column (the index
    is reset if needed) so the Parquet schema matches the data contract.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    if "timestamp" not in out.columns:
        raise ValueError("Cannot persist a frame without a 'timestamp' column/index")
    out.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    """Read a normalized OHLCV Parquet file back into a DataFrame.

    Returns a frame with ``timestamp`` as a column (not the index); callers
    that need a DatetimeIndex should ``set_index("timestamp")``.
    """
    return pd.read_parquet(path)


def run_data_pipeline(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load + normalize + save processed Parquet according to ``cfg``.

    Returns the normalized DataFrame (timestamp-indexed, UTC).

    Optional ``data.resample_rule`` (e.g. ``"15min"``): when set, the
    UTC-normalized frame is causally resampled (``resampler.resample_ohlcv``)
    *before* persistence.  Each bin aggregates only closed bars fully inside
    it (open=first, high=max, low=min, close=last, volume=sum), bins labeled
    by start time per pipeline convention.  Used for M1 raw exports (BTCUSD)
    that must run as M15; XAUUSD/EURUSD configs do not set the key, so their
    behavior is unchanged.
    """
    input_path = resolve_path(cfg, cfg["data"]["input_path"])
    output_path = resolve_path(cfg, cfg["data"]["output_path"])
    volume_kind = cfg["data"].get("volume_kind", "tick")
    timezone = cfg["project"].get("timezone", "UTC")
    utc_offset = cfg["data"].get("utc_offset_hours", "auto")
    resample_rule = cfg["data"].get("resample_rule")

    raw = load_ohlcv(input_path)
    normalized = normalize_ohlcv(
        raw, volume_kind=volume_kind, timezone=timezone, utc_offset_hours=utc_offset
    )
    if resample_rule:
        from ..data.resampler import resample_ohlcv

        normalized = resample_ohlcv(normalized, resample_rule)
    save_parquet(normalized, output_path)
    return normalized


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import json

    from ..config import load_config
    from .validator import build_data_quality_report, save_report

    cfg = load_config()
    input_path = resolve_path(cfg, cfg["data"]["input_path"])
    raw = load_ohlcv(input_path)
    normalized = run_data_pipeline(cfg)
    offsets = infer_utc_offsets(raw["timestamp"])
    report = build_data_quality_report(
        normalized,
        cfg["data"].get("volume_kind", "tick"),
        cfg["project"].get("timezone", "UTC"),
        raw=raw,
        utc_offsets=offsets,
        source_path=cfg["data"]["input_path"],
        output_path=cfg["data"]["output_path"],
    )
    report_path = resolve_path(cfg, "reports/data_quality.json")
    save_report(report, report_path)
    print(json.dumps(report, indent=2, default=str))