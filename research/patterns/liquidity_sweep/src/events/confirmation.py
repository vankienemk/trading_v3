"""Confirmation detection for sweep events (guide sections 11, 13.8, Phase 5).

Confirmation is evaluated *after* a sweep and is never known at the moment of
the sweep (principle 3.2 / section 11.1).  For every sweep event this module
scans the next ``max_wait_bars`` candles (baseline 3) and reports the first
candle that satisfies one of the supported confirmation types:

* ``close_break``    — candle closes beyond the sweep extreme with
  ``body_ratio >= min_body_ratio`` and ``range/ATR >= min_range_atr`` (the
  baseline of section 11.2);
* ``displacement``   — a displacement candle: closes beyond the sweep extreme
  with ``body_ratio >= min_body_ratio`` and ``range/ATR >=
  displacement_min_range_atr`` (a strong-range variant, Phase 5);
* ``structure_break``— candle closes beyond the most recent *confirmed* swing
  pivot opposite to the sweep (short-term structure break) with the baseline
  body/range filters (Phase 5).

Run-aware anchoring (QA finding F1 / guide section 12)
------------------------------------------------------
A single level can produce a *run* of consecutive sweep candles; the event
table's representative bar may be a later bar of the run (default
``deepest_penetration`` grouping).  Decisions here therefore anchor on the
run's **first** bar (recovered causally from the per-bar sweep flags): the
confirmation window is ``[first_bar + 1, first_bar + max_wait_bars]`` and the
sweep extreme used for the break condition is the causal accumulation
``max(high[first_bar..k-1])`` (long) / ``min(low[first_bar..k-1])`` (short) —
at candle ``k``'s close only the bars *before* ``k`` are read, so a
confirmation candle can still close beyond the level it is measured against.
With ``group_rule="first"`` events the whole events+confirmation table is
truncation-invariant (locked by ``tests/test_no_lookahead.py``).

Entry timing (section 11.3)
---------------------------
``confirmation_time`` is the close time of the confirmation candle ``k``; the
entry must be at the open of candle ``k+1``, never at the open of the
confirmation candle itself.  :func:`compare_entry_strategies` implements the
Phase 5 comparison of A (entry at the open of the candle right after the sweep)
vs B (entry at the open of the candle after confirmation).

Output
------
:func:`attach_confirmations` appends the schema §6 columns
(``src.schema.CONFIRMATION_COLUMNS``) plus ``confirmation_type`` and
``confirmation_strength`` (guide section 5.4).  Unconfirmed events carry
``is_confirmed=False`` and ``NaN``/``NaT`` placeholders so setup, confirmation
and label stay separate columns (principle 3.2).

Timezone policy
---------------
Timestamps are kept consistent with the candles index: a tz-aware
(``datetime64[ns, UTC]`` per the schema / data pipeline) index produces
tz-aware ``confirmation_time`` values — including the all-``NaT`` case — and a
naive index produces naive values.  All timestamp→bar lookups inside this
module (*and* by downstream consumers against ``confirmation_time`` /
``event_time``) normalize the searched value to the candles index's
timezone-awareness, so a mixed tz-naive / tz-aware comparison can never raise
a ``TypeError`` (this is the failing mode reported in review).  Naive values
are interpreted as UTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema
from src.events.sweep_detector import detect_sweeps

#: Defaults mirror ``configs/baseline.yaml`` (confirmation section).
DEFAULT_MAX_WAIT_BARS = 3
DEFAULT_MIN_BODY_RATIO = 0.60
DEFAULT_MIN_RANGE_ATR = 0.80
DEFAULT_REQUIRE_BREAK_SWEEP_EXTREME = True
DEFAULT_DISPLACEMENT_MIN_RANGE_ATR = 1.5
DEFAULT_STRUCTURE_LEFT_BARS = 3
DEFAULT_STRUCTURE_RIGHT_BARS = 3
DEFAULT_STRUCTURE_MAX_AGE_BARS = 400
DEFAULT_BUFFER_ATR = 0.10  # configs/baseline.yaml stop.buffer_atr

CONFIRMATION_TYPES: list[str] = ["close_break", "displacement", "structure_break"]

#: Extra columns beyond schema §6, per guide section 5.4 Agent-3 output.
CONFIRMATION_EXTRA_COLUMNS: list[str] = ["confirmation_type", "confirmation_strength"]

_LONG, _SHORT = "long", "short"


def normalize_direction(direction: str) -> str:
    """Map the direction vocabulary to ``{long, short}``.

    Delegates to :func:`src.schema.normalize_event_direction` — rule 30.1's
    single implementation point (Agent 0's schema module).  Accepts both the
    event-table spelling (``bullish``/``bearish``, t4) and the trade spelling
    (``long``/``short``).
    """
    return schema.normalize_event_direction(direction)


def _sweep_flags(candles: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Causal per-bar sweep flags + ATR (recomputed; validates ``candles``)."""
    ind = config.get("indicators", {})
    liq = config.get("liquidity", {})
    swp = config.get("sweep", {})
    return detect_sweeps(
        candles,
        atr_period=ind.get("atr_period", 14),
        level_lookback=liq.get("rolling_lookback", 20),
        min_penetration_atr=swp.get("min_penetration_atr", 0.05),
        max_penetration_atr=swp.get("max_penetration_atr", 0.50),
        min_wick_ratio=swp.get("min_wick_ratio", 0.35),
        min_reclaim_atr=swp.get("min_reclaim_atr", 0.0),
    )


def run_anchor_position(flags: pd.DataFrame, pos: int, direction: str) -> int:
    """First bar of the run of consecutive same-direction sweeps containing ``pos``.

    Walks backward while the adjacent earlier bar is also a sweep flag of the
    same direction.  The anchor is known as soon as the first bar of the run
    closes, so confirmations anchored here are causal even when the event
    representative is a later (deepest-penetration) bar of the run.
    """
    col = "sweep_long" if normalize_direction(direction) == _LONG else "sweep_short"
    flags_np = flags[col].fillna(False).to_numpy()
    start = int(pos)
    while start > 0 and bool(flags_np[start - 1]):
        start -= 1
    return start


def run_extreme_at(
    candles: pd.DataFrame, first_bar: int, k: int, direction: str
) -> float:
    """Causal sweep extreme accumulated over bars ``[first_bar..k]`` (<= k only).

    ``high`` peak for long events (the level a bullish confirmation must close
    above), ``low`` trough for short events.  For the confirmation break test
    the caller passes ``k-1`` (the extreme of bars *before* the confirmation
    candle), so a single-candle sweep reproduces exactly ``high[sweep]`` /
    ``low[sweep]`` of section 11.2.
    """
    if normalize_direction(direction) == _LONG:
        return float(candles["high"].iloc[first_bar : k + 1].max())
    return float(candles["low"].iloc[first_bar : k + 1].min())


def run_opposite_extreme_at(
    candles: pd.DataFrame, first_bar: int, k: int, direction: str
) -> float:
    """Mirror extreme (stop reference, guide section 14.2/14.3).

    The low trough for long events (``stop = low_of_sweep - buffer*ATR``) and
    the high peak for short events (``stop = high_of_sweep + buffer*ATR``),
    accumulated over bars ``[first_bar..k]``; all bars are ``<= k`` and known at
    the entry decision, so the stop placement is causal.
    """
    if normalize_direction(direction) == _LONG:
        return float(candles["low"].iloc[first_bar : k + 1].min())
    return float(candles["high"].iloc[first_bar : k + 1].max())


def _is_swing_pivot(series: np.ndarray, i: int, left_bars: int, right_bars: int, direction: str) -> bool:
    """Strict fractal: pivot is beyond its ``left_bars``+``right_bars`` neighbours."""
    if direction == _LONG:
        return series[i] > series[i - left_bars : i].max() and series[i] > series[i + 1 : i + 1 + right_bars].max()
    return series[i] < series[i - left_bars : i].min() and series[i] < series[i + 1 : i + 1 + right_bars].min()


def structure_level_at(
    candles: pd.DataFrame,
    before_pos: int,
    direction: str,
    left_bars: int = DEFAULT_STRUCTURE_LEFT_BARS,
    right_bars: int = DEFAULT_STRUCTURE_RIGHT_BARS,
    max_age_bars: int = DEFAULT_STRUCTURE_MAX_AGE_BARS,
) -> float:
    """Price of the most recent *confirmed* swing pivot strictly before ``before_pos``.

    A pivot at candle ``i`` requires ``left_bars`` lower (higher) neighbours on
    each side and is only *known* at ``i + right_bars`` (its confirmation bar).
    Only pivots with ``i + right_bars < before_pos`` are usable, so the result
    is causal.  Returns ``NaN`` when no confirmed pivot is available.  This is
    the minimal local structure definition used by the ``structure_break``
    confirmation type; the canonical swing registry is Agent 2's
    ``src.liquidity.swing_levels`` (task t8).
    """
    series = (
        candles["high"].to_numpy()
        if normalize_direction(direction) == _LONG
        else candles["low"].to_numpy()
    )
    side = normalize_direction(direction)
    latest: float | None = None
    first = max(left_bars, before_pos - max_age_bars)
    for i in range(first, before_pos - right_bars):
        if _is_swing_pivot(series, i, left_bars, right_bars, side):
            latest = float(series[i])
    return float("nan") if latest is None else latest


def _confirmation_strength(body_ratio: float, range_atr: float) -> float:
    """Quality score in ``[0, 1]``: halves of normalized body and range.

    ``strength = 0.5 * min(body_ratio, 1) + 0.5 * min(range_atr / 2, 1)`` — a
    full-body, 2+ ATR candle scores 1.0.  Documented in the module docstring;
    used by human review (t7) and scoring (t13).
    """
    return float(
        0.5 * min(max(body_ratio, 0.0), 1.0)
        + 0.5 * min(max(range_atr, 0.0) / 2.0, 1.0)
    )


def _as_index_timestamp(index: pd.Index, ts) -> pd.Timestamp:
    """Normalize ``ts`` to the index's timezone-awareness for lookups.

    A tz-aware index paired with a naive value treats the value as UTC
    (``tz_localize(index.tz)`` after converting to UTC); a naive index paired
    with a tz-aware value drops the tz after converting to UTC.  Two different
    aware timezones are converted to the index's.  This makes every timestamp
    comparison in the module (and by consumers of ``confirmation_time`` /
    ``event_time``) TypeError-free regardless of input framing.
    """
    if not isinstance(index, pd.DatetimeIndex) or not isinstance(ts, pd.Timestamp):
        return ts
    if index.tz is None:
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("UTC").tz_localize(None)  # naive index == UTC
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(index.tz)  # aware index, naive value == UTC
    return ts.tz_convert(index.tz)


def _lookup_position(candles: pd.DataFrame, index: pd.Index, ts) -> int:
    """Bar position of ``ts`` in ``candles``, tz-normalized; raises if absent."""
    locs = index.get_indexer([_as_index_timestamp(index, ts)])
    if locs[0] == -1:
        raise ValueError(
            f"timestamp {ts!r} is not present in the candles index "
            "(tz-normalized to the index's timezone-awareness)"
        )
    return int(locs[0])


def _event_position(candles: pd.DataFrame, events: pd.DataFrame, row) -> int:
    if "bar_index" in events.columns and not pd.isna(row.get("bar_index")):
        return int(row["bar_index"])
    return _lookup_position(candles, candles.index, row["event_time"])


def attach_confirmations(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    config: dict | None = None,
    *,
    max_wait_bars: int | None = None,
    min_body_ratio: float | None = None,
    min_range_atr: float | None = None,
    require_break_sweep_extreme: bool | None = None,
    confirmation_types: list[str] | None = None,
    displacement_min_range_atr: float = DEFAULT_DISPLACEMENT_MIN_RANGE_ATR,
    structure_left_bars: int = DEFAULT_STRUCTURE_LEFT_BARS,
    structure_right_bars: int = DEFAULT_STRUCTURE_RIGHT_BARS,
    structure_max_age_bars: int = DEFAULT_STRUCTURE_MAX_AGE_BARS,
) -> pd.DataFrame:
    """Attach schema §6 confirmation columns to a copy of the event table.

    ``candles`` is the sorted OHLCV frame the events were detected on; ``events``
    must carry ``event_time`` and ``direction`` (``bullish``/``bearish`` or
    ``long``/``short``).  ``config`` follows ``configs/baseline.yaml``
    (``confirmation``, ``sweep``, ``indicators``, ``liquidity`` sections); each
    keyword overrides the config / baseline default.  If
    ``confirmation.enabled`` is false every event is reported unconfirmed.

    The scan follows section 11.2: the first candle ``k`` in
    ``[first_bar+1, first_bar+max_wait_bars]`` that satisfies any configured
    type (types evaluated in list order, candles in ascending order) is the
    confirmation candle.  Returns ``events`` (copy) with the confirmation
    columns appended; the input frames are never mutated.
    """
    cfg = dict(config or {})
    conf = cfg.get("confirmation", {})
    max_wait = (
        conf.get("max_wait_bars", DEFAULT_MAX_WAIT_BARS)
        if max_wait_bars is None
        else max_wait_bars
    )
    body_min = (
        conf.get("min_body_ratio", DEFAULT_MIN_BODY_RATIO)
        if min_body_ratio is None
        else min_body_ratio
    )
    range_min = (
        conf.get("min_range_atr", DEFAULT_MIN_RANGE_ATR)
        if min_range_atr is None
        else min_range_atr
    )
    req_break = (
        conf.get("require_break_sweep_extreme", DEFAULT_REQUIRE_BREAK_SWEEP_EXTREME)
        if require_break_sweep_extreme is None
        else require_break_sweep_extreme
    )
    types = list(confirmation_types or CONFIRMATION_TYPES)
    for t in types:
        if t not in CONFIRMATION_TYPES:
            raise ValueError(
                f"attach_confirmations: unknown confirmation type {t!r}; "
                f"supported: {CONFIRMATION_TYPES}"
            )
    enabled = conf.get("enabled", True)

    required = ["event_time", "direction"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(
            f"attach_confirmations: events table missing required columns {missing}"
        )
    if len(events) == 0:
        out = events.copy()
        ts_dtype = candles.index.dtype if isinstance(candles.index, pd.DatetimeIndex) else "object"
        for col, dtype in (
            ("is_confirmed", "bool"),
            ("confirmation_time", ts_dtype),
            ("confirmation_delay_bars", "float64"),
            ("confirmation_close", "float64"),
            ("confirmation_range_atr", "float64"),
            ("confirmation_body_ratio", "float64"),
            ("confirmation_volume", "float64"),
            ("confirmation_type", "object"),
            ("confirmation_strength", "float64"),
        ):
            out[col] = pd.Series(dtype=dtype, index=events.index)
        return out

    flags = _sweep_flags(candles, cfg)
    n = len(candles)
    if max_wait < 0:
        raise ValueError(f"attach_confirmations: max_wait_bars must be >= 0, got {max_wait}")
    if not 0.0 <= body_min <= 1.0:
        raise ValueError(f"attach_confirmations: min_body_ratio must be in [0, 1], got {body_min}")
    if range_min < 0.0:
        raise ValueError(f"attach_confirmations: min_range_atr must be >= 0, got {range_min}")
    if displacement_min_range_atr < 0.0:
        raise ValueError(
            "attach_confirmations: displacement_min_range_atr must be >= 0, "
            f"got {displacement_min_range_atr}"
        )

    rows: list[dict] = []
    for _, row in events.iterrows():
        direction = normalize_direction(row["direction"])
        pos = _event_position(candles, events, row)
        anchor = run_anchor_position(flags, pos, direction)

        found_k: int | None = None
        found_type: str | None = None
        if enabled:
            for k in range(anchor + 1, min(anchor + max_wait + 1, n)):
                candle = candles.iloc[k]
                o, h, lo, c = (candle["open"], candle["high"], candle["low"], candle["close"])
                rng = h - lo
                atr_k = flags["atr"].iloc[k]
                if not np.isfinite(atr_k) or rng <= 0:
                    continue
                body_ratio = abs(c - o) / rng
                range_atr = rng / atr_k

                extreme = run_extreme_at(candles, anchor, k - 1, direction)
                broke_extreme = (c > extreme) if direction == _LONG else (c < extreme)
                broke_structure = False
                if "structure_break" in types:
                    # Structure must pre-date the sweep run: use pivots known
                    # at or before the run's first bar.
                    level = structure_level_at(
                        candles,
                        anchor + 1,
                        direction,
                        left_bars=structure_left_bars,
                        right_bars=structure_right_bars,
                        max_age_bars=structure_max_age_bars,
                    )
                    if np.isfinite(level):
                        broke_structure = (c > level) if direction == _LONG else (c < level)

                for t in types:
                    ok = False
                    if t == "close_break":
                        ok = (not req_break or broke_extreme) and (
                            body_ratio >= body_min and range_atr >= range_min
                        )
                    elif t == "displacement":
                        ok = broke_extreme and (
                            body_ratio >= body_min
                            and range_atr >= displacement_min_range_atr
                        )
                    elif t == "structure_break":
                        ok = broke_structure and (
                            body_ratio >= body_min and range_atr >= range_min
                        )
                    if ok:
                        found_k, found_type = k, t
                        break
                if found_k is not None:
                    break

        if found_k is None:
            rows.append(
                {
                    "is_confirmed": False,
                    "confirmation_time": pd.NaT,
                    "confirmation_delay_bars": np.nan,
                    "confirmation_close": np.nan,
                    "confirmation_range_atr": np.nan,
                    "confirmation_body_ratio": np.nan,
                    "confirmation_volume": np.nan,
                    "confirmation_type": np.nan,
                    "confirmation_strength": np.nan,
                }
            )
        else:
            candle = candles.iloc[found_k]
            rng = candle["high"] - candle["low"]
            body_ratio = abs(candle["close"] - candle["open"]) / rng
            range_atr = rng / flags["atr"].iloc[found_k]
            vol = candle["volume"] if "volume" in candles.columns else np.nan
            rows.append(
                {
                    "is_confirmed": True,
                    "confirmation_time": candles.index[found_k],
                    "confirmation_delay_bars": found_k - anchor,
                    "confirmation_close": float(candle["close"]),
                    "confirmation_range_atr": float(range_atr),
                    "confirmation_body_ratio": float(body_ratio),
                    "confirmation_volume": float(vol),
                    "confirmation_type": found_type,
                    "confirmation_strength": _confirmation_strength(body_ratio, range_atr),
                }
            )

    confirm_df = pd.DataFrame(rows, index=events.index)
    # Keep the tz-awareness of the candles index.  A naive datetime column
    # (pandas builds one when every row is NaT) would break comparisons
    # against tz-aware boundary timestamps, so normalize explicitly:
    # naive values are read as UTC (timezone-policy docstring).
    if isinstance(candles.index, pd.DatetimeIndex):
        utc_col = pd.to_datetime(confirm_df["confirmation_time"], utc=True)
        tz = candles.index.tz
        confirm_df["confirmation_time"] = (
            utc_col.dt.tz_convert(tz) if tz is not None else utc_col.dt.tz_localize(None)
        )
    out = events.copy()
    for col in schema.CONFIRMATION_COLUMNS + CONFIRMATION_EXTRA_COLUMNS:
        out[col] = confirm_df[col]
    return out


def build_entry_schedule(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    config: dict | None = None,
    *,
    mode: str = "after_confirmation",
) -> pd.DataFrame:
    """Per-event entry timing (guide sections 11.3 / 14.1).

    Returns one row per executable entry: ``event_id``, ``direction``,
    ``sweep_anchor_time`` (first bar of the sweep run), ``is_confirmed``,
    ``confirmation_time``, ``entry_bar``, ``entry_time`` and ``entry_price``
    (the **open of the candle after** the decision candle — never the open of a
    confirmation candle itself, section 11.3).

    ``mode`` selects the decision point:

    * ``after_confirmation`` (default) — entry at the open of the candle after
      the confirmation candle; unconfirmed events are dropped;
    * ``after_sweep`` — entry at the open of the candle right after the sweep
      anchor (section 14.1, strategy A).

    Only the config ``entry`` cost placeholders (``slippage_price`` /
    ``spread_price``, default 0) are applied; the full cost model is Agent 5's
    labeling contract.
    """
    if mode not in ("after_confirmation", "after_sweep"):
        raise ValueError(
            f"build_entry_schedule: mode must be 'after_confirmation' or "
            f"'after_sweep', got {mode!r}"
        )
    cfg = dict(config or {})
    required = ["event_time", "direction"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(
            f"build_entry_schedule: events table missing required columns {missing}"
        )
    if mode == "after_confirmation" and "is_confirmed" not in events.columns:
        raise ValueError(
            "build_entry_schedule: mode='after_confirmation' needs the "
            "confirmation columns — run attach_confirmations first"
        )

    costs = cfg.get("entry", {})
    slip = float(costs.get("slippage_price", 0.0))
    spread = float(costs.get("spread_price", 0.0))
    flags = _sweep_flags(candles, cfg)
    n = len(candles)

    rows: list[dict] = []
    for _, row in events.iterrows():
        direction = normalize_direction(row["direction"])
        pos = _event_position(candles, events, row)
        anchor = run_anchor_position(flags, pos, direction)
        is_long = direction == _LONG

        if mode == "after_sweep":
            entry_bar = anchor + 1
        else:
            if not bool(row.get("is_confirmed", False)) or pd.isna(row.get("confirmation_time")):
                continue
            conf_bar = _lookup_position(candles, candles.index, row["confirmation_time"])
            entry_bar = conf_bar + 1
        if entry_bar >= n:
            continue
        entry_price = float(candles["open"].iloc[entry_bar])
        entry_price = entry_price + slip + spread if is_long else entry_price - slip - spread
        rows.append(
            {
                "event_id": row.get("event_id", np.nan),
                "direction": direction,
                "sweep_anchor_time": candles.index[anchor],
                "is_confirmed": bool(row.get("is_confirmed", False)),
                "confirmation_time": row.get("confirmation_time", pd.NaT),
                "entry_bar": int(entry_bar),
                "entry_time": candles.index[entry_bar],
                "entry_price": float(entry_price),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id", "direction", "sweep_anchor_time", "is_confirmed",
                "confirmation_time", "entry_bar", "entry_time", "entry_price",
            ]
        )
    return pd.DataFrame(rows)


def _strategy_trades(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    config: dict,
    flags: pd.DataFrame,
    *,
    reward_r: float,
    horizon_bars: int,
    buffer_atr: float,
) -> tuple[list[dict], list[dict]]:
    """Per-event simulated trades for A (after sweep) and B (after confirmation).

    Entry is always the **open of the candle after** the decision bar (section
    11.3 / 14.1); the stop is the sweep extreme at the sweep (section 14.2/14.3:
    ``sweep_extreme -/+ buffer_atr * ATR[sweep]``) — never placed at the open of
    a decision candle.  Outcome is computed with a fixed ``reward_r`` target and
    a ``horizon_bars`` cap: first touch of stop → ``-1R``, first touch of target
    → ``+reward_r`` (same-bar stop priority, conservative), otherwise
    mark-to-market at the horizon close.  MFE/MAE are max favorable/adverse
    excursions over the scanned bars, in R.  Costs: only the config entry
    placeholders (``entry.slippage_price`` / ``entry.spread_price``, default 0)
    are applied; the full cost model is Agent 5's labeling contract (task t6).
    """
    costs = config.get("entry", {})
    slip = float(costs.get("slippage_price", 0.0))
    spread = float(costs.get("spread_price", 0.0))

    a_trades: list[dict] = []
    b_trades: list[dict] = []
    n = len(candles)
    for _, row in events.iterrows():
        direction = normalize_direction(row["direction"])
        pos = _event_position(candles, events, row)
        anchor = run_anchor_position(flags, pos, direction)
        is_long = direction == _LONG

        extreme = run_opposite_extreme_at(candles, anchor, anchor, direction)
        atr_anchor = flags["atr"].iloc[anchor]
        if not np.isfinite(atr_anchor):
            continue
        if is_long:
            stop = extreme - buffer_atr * atr_anchor
        else:
            stop = extreme + buffer_atr * atr_anchor

        candidates = [("after_sweep", anchor + 1)]
        if bool(row.get("is_confirmed", False)) and not pd.isna(row.get("confirmation_time")):
            conf_pos = _lookup_position(candles, candles.index, row["confirmation_time"])
            candidates.append(("after_confirmation", conf_pos + 1))

        for strategy, entry_idx in candidates:
            if entry_idx >= n:
                continue
            entry_price = float(candles["open"].iloc[entry_idx])
            entry_price = entry_price + slip + spread if is_long else entry_price - slip - spread
            risk = abs(entry_price - stop)
            if risk <= 0 or not np.isfinite(risk):
                continue
            target = entry_price + reward_r * risk if is_long else entry_price - reward_r * risk

            last = min(entry_idx + horizon_bars, n)
            outcome: float
            mfe, mae = 0.0, 0.0
            hit = False
            for j in range(entry_idx, last):
                hi, lo = candles["high"].iloc[j], candles["low"].iloc[j]
                if is_long:
                    mfe = max(mfe, (hi - entry_price) / risk)
                    mae = min(mae, (lo - entry_price) / risk)
                else:
                    mfe = max(mfe, (entry_price - lo) / risk)
                    mae = min(mae, (entry_price - hi) / risk)
                if (lo <= stop) if is_long else (hi >= stop):
                    outcome, hit = -1.0, True
                    break
                if (hi >= target) if is_long else (lo <= target):
                    outcome, hit = reward_r, True
                    break
            if not hit:
                outcome = (
                    (candles["close"].iloc[last - 1] - entry_price) / risk
                    if is_long
                    else (entry_price - candles["close"].iloc[last - 1]) / risk
                )

            trade = {
                "event_id": row.get("event_id", np.nan),
                "entry_idx": entry_idx,
                "entry_price": entry_price,
                "stop": stop,
                "target": target,
                "outcome_r": float(outcome),
                "mfe_r": float(mfe),
                "mae_r": float(mae),
                "risk_atr": float(risk / atr_anchor),
                "entry_delay": int(entry_idx - anchor),
            }
            (a_trades if strategy == "after_sweep" else b_trades).append(trade)
    return a_trades, b_trades


def compare_entry_strategies(
    candles: pd.DataFrame,
    confirmed_events: pd.DataFrame,
    config: dict | None = None,
    *,
    reward_r: float = 1.0,
    horizon_bars: int = 32,
    buffer_atr: float | None = None,
) -> pd.DataFrame:
    """Phase 5 comparison: A (entry right after the sweep) vs B (after confirmation).

    ``confirmed_events`` must already carry the confirmation columns from
    :func:`attach_confirmations`.  Metrics per strategy (guard: rows with no
    executable trades are omitted):

    * ``event_count`` — executable entries;
    * ``win_rate`` — share of outcomes ``> 0``;
    * ``expectancy_r`` — mean outcome in R;
    * ``avg_stop_distance_atr`` — mean ``risk / ATR[sweep]``;
    * ``mfe_mean_r`` / ``mae_mean_r`` — mean max favorable / adverse excursion;
    * ``avg_entry_delay_bars`` — bars from the sweep anchor to the entry open
      (A: 1; B: confirmation delay + 1).

    The comparison is *decision-causal*: every entry/stop is fixed from bars up
    to the decision candle; MFE/MAE/outcome are retrospective analysis of what
    happened after entry (the same role the labeling stage plays for the full
    backtest).  Stop mode must be ``sweep_extreme`` (config ``stop.mode``).
    """
    cfg = dict(config or {})
    if "is_confirmed" not in confirmed_events.columns or "confirmation_time" not in confirmed_events.columns:
        raise ValueError(
            "compare_entry_strategies: confirmed_events needs the confirmation "
            "columns — run attach_confirmations first"
        )
    stop_cfg = cfg.get("stop", {})
    stop_mode = stop_cfg.get("mode", "sweep_extreme")
    if stop_mode != "sweep_extreme":
        raise ValueError(
            f"compare_entry_strategies: unsupported stop.mode {stop_mode!r} "
            "(only 'sweep_extreme' is implemented)"
        )
    buf = stop_cfg.get("buffer_atr", DEFAULT_BUFFER_ATR) if buffer_atr is None else buffer_atr
    if reward_r <= 0:
        raise ValueError(f"compare_entry_strategies: reward_r must be > 0, got {reward_r}")
    if horizon_bars < 1:
        raise ValueError(f"compare_entry_strategies: horizon_bars must be >= 1, got {horizon_bars}")

    flags = _sweep_flags(candles, cfg)
    a_trades, b_trades = _strategy_trades(
        candles, confirmed_events, cfg, flags,
        reward_r=reward_r, horizon_bars=horizon_bars, buffer_atr=buf,
    )

    def _metrics(trades: list[dict]) -> dict | None:
        if not trades:
            return None
        outs = np.array([t["outcome_r"] for t in trades])
        return {
            "event_count": len(trades),
            "win_rate": float((outs > 0).mean()),
            "expectancy_r": float(outs.mean()),
            "avg_stop_distance_atr": float(np.mean([t["risk_atr"] for t in trades])),
            "mfe_mean_r": float(np.mean([t["mfe_r"] for t in trades])),
            "mae_mean_r": float(np.mean([t["mae_r"] for t in trades])),
            "avg_entry_delay_bars": float(np.mean([t["entry_delay"] for t in trades])),
        }

    rows = {}
    m_a, m_b = _metrics(a_trades), _metrics(b_trades)
    if m_a is not None:
        rows["after_sweep"] = m_a
    if m_b is not None:
        rows["after_confirmation"] = m_b
    return pd.DataFrame.from_dict(rows, orient="index")