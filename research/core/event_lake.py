"""
event_lake.py — Append-only PatternEvent store (§7)

Upgrades the FeatureStore into a first-class persistence product
(REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §7):

    event_lake/
    ├── events/{pattern_name}/{symbol}/{yyyy-mm}.parquet   # append-only, full PatternEvent
    ├── outcomes/{event_id}.parquet                        # forward outcomes, join by event_id
    └── metrics/{model_id}.parquet                         # rolling metrics cho LifecycleManager

Principles (§7.1): EVERY PatternEvent is persisted — including events rejected
by threshold, deduped by CorrelationManager, or discarded as stale — carrying
``attributes.discard_reason``; forward outcomes (return, MFE/MAE, hypothetical
PnL minus costs) are appended later by the :class:`OutcomeTracker` periodic job.
This module also ships the PSI drift monitor (spec §5.2/§5.3) exposed for
LifecycleManager and ``rebuild_dataset`` (§7.3.1) so retrain never needs to
re-run a detector.

Append-only invariant: rows are never mutated or deleted.  A partition file is
re-written (read existing + concat new rows + write) only to *append*; re-appending
an existing ``event_id`` raises :class:`EventLakeError`.

Cost model (§6.5): per-symbol ``{"spread": .., "commission": .., "slippage": ..}``
in price units; hypothetical PnL subtracts the total per-unit cost.

Drift thresholds (§5.2/§5.3): PSI < 0.1 healthy; 0.1-0.2 moderate; >= 0.2 drift on
``min_drift_features`` core features -> ``requires_retrain``; any feature >= 0.25
-> ``demote_to_degraded``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from research.core.config_hash import canonical_json
from research.core.contracts import PatternEvent

# ---------------------------------------------------------------------------
# Layout (§7.2)
# ---------------------------------------------------------------------------
EVENT_LAKE_DIRNAME = "event_lake"
EVENTS_SUBDIR = "events"
OUTCOMES_SUBDIR = "outcomes"
METRICS_SUBDIR = "metrics"
PARTITION_DATE_FORMAT = "%Y-%m"

#: Exit reasons produced by the outcome tracker's triple-barrier walk.
EXIT_TARGET_HIT = "target_hit"
EXIT_STOP_HIT = "stop_hit"
EXIT_TIMEOUT = "timeout"
EXIT_NO_DATA = "no_data"

#: Hypothetical PnL is per-unit (price units); a missing per-symbol cost entry
#: contributes zero cost (spec §6.5 stores costs in the dataset row — callers
#: pass the configured per-symbol cost model explicitly).
DEFAULT_COSTS: dict[str, dict[str, float]] = {}

_EVENT_COLUMNS: list[str] = [
    "event_id",
    "pattern_name",
    "pattern_version",
    "symbol",
    "timeframe",
    "direction",
    "detect_time",
    "confirm_time",
    "entry_time",
    "known_at",
    "entry_price",
    "stop_price",
    "target_price",
    "rule_score",
    "model_prob",
    "config_hash",
    "feature_schema_version",
    "structure_levels_json",
    "attributes_json",
    "lifecycle_state",
    "confluence_group_id",
]

_OUTCOME_COLUMNS: list[str] = [
    "event_id",
    "pattern_name",
    "symbol",
    "timeframe",
    "entry_time",
    "horizon",
    "trade_direction",
    "exit_reason",
    "exit_price",
    "forward_return_r",
    "mfe_r",
    "mae_r",
    "mfe",
    "mae",
    "gross_pnl",
    "costs_total",
    "pnl_net",
    "pnl_net_r",
    "window_truncated",
    "computed_at",
]

_METRICS_COLUMNS: list[str] = [
    "asof",
    "pf",
    "win_rate",
    "n_trades",
    "max_psi",
    "psi_retrain",
    "lifecycle_state",
]


class EventLakeError(Exception):
    """Append-only violation, malformed layout argument or bad row data."""


# ---------------------------------------------------------------------------
# Timestamp + JSON helpers (canonical, deterministic)
# ---------------------------------------------------------------------------

def _as_utc(ts: Any) -> pd.Timestamp:
    """Coerce *ts* to a tz-aware UTC :class:`pd.Timestamp` (NaT stays NaT)."""
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return cast("pd.Timestamp", t)
    return t if t.tz is not None else t.tz_localize("UTC")


def _as_utc_optional(ts: Any) -> pd.Timestamp:
    """Like :func:`_as_utc` but maps ``None`` to NaT."""
    if ts is None or pd.isna(ts):
        return cast("pd.Timestamp", pd.NaT)
    return _as_utc(ts)


def _read_ts(value: Any) -> pd.Timestamp | None:
    """Parquet-cell -> optional timestamp (NaT -> None)."""
    if value is None or pd.isna(value):
        return None
    t = pd.Timestamp(value)
    return t if t.tz is not None else t.tz_localize("UTC")


def _read_mandatory_ts(value: Any) -> pd.Timestamp:
    """Parquet-cell -> timestamp; NaT when the cell is empty."""
    ts = _read_ts(value)
    return ts if ts is not None else cast("pd.Timestamp", pd.NaT)


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain JSON types."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _flatten_dict(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Flatten nested attribute dicts into dotted primitive columns."""
    out: dict[str, Any] = {}
    if not isinstance(value, dict) or depth >= 3:
        return out
    for key, val in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            out.update(_flatten_dict(val, name, depth + 1))
        elif isinstance(val, (int, float, str, bool)):
            out[name] = val
        elif isinstance(val, pd.Timestamp):
            out[name] = val.isoformat()
    return out


# ---------------------------------------------------------------------------
# PatternEvent <-> frame serialization
# ---------------------------------------------------------------------------

def _event_to_row(ev: PatternEvent) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "pattern_name": ev.pattern_name,
        "pattern_version": ev.pattern_version,
        "symbol": ev.symbol,
        "timeframe": ev.timeframe,
        "direction": ev.direction,
        "detect_time": _as_utc(ev.detect_time),
        "confirm_time": _as_utc_optional(ev.confirm_time),
        "entry_time": _as_utc_optional(ev.entry_time),
        "known_at": _as_utc_optional(ev.known_at),
        "entry_price": float(ev.entry_price),
        "stop_price": float(ev.stop_price),
        "target_price": float("nan")
        if ev.target_price is None
        else float(ev.target_price),
        "rule_score": float(ev.rule_score),
        "model_prob": float("nan") if ev.model_prob is None else float(ev.model_prob),
        "config_hash": ev.config_hash,
        "feature_schema_version": ev.feature_schema_version,
        "structure_levels_json": canonical_json(_json_safe(ev.structure_levels)),
        "attributes_json": canonical_json(_json_safe(ev.attributes)),
        "lifecycle_state": ev.lifecycle_state,
        "confluence_group_id": ev.confluence_group_id or "",
    }


def events_to_frame(events: Sequence[PatternEvent]) -> pd.DataFrame:
    """Serialize PatternEvents into the canonical event frame (one row each).

    Every event goes through :func:`_event_to_row` — including rejected /
    deduped / stale events whose ``attributes.discard_reason`` is preserved in
    ``attributes_json`` (§7.1).
    """
    rows = [_event_to_row(ev) for ev in events]
    return pd.DataFrame(rows, columns=_EVENT_COLUMNS)


def _row_to_event(row: Any) -> PatternEvent:
    def _f(col: str) -> float:
        v = row.get(col, float("nan"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def _opt_float(col: str) -> float | None:
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(fv) else fv

    attrs_raw = row.get("attributes_json", "")
    attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) and attrs_raw else {}
    levels_raw = row.get("structure_levels_json", "")
    levels = json.loads(levels_raw) if isinstance(levels_raw, str) and levels_raw else {}
    group_id = row.get("confluence_group_id", "")
    return PatternEvent(
        event_id=str(row["event_id"]),
        pattern_name=str(row["pattern_name"]),
        pattern_version=str(row["pattern_version"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        direction=str(row["direction"]),
        detect_time=_read_mandatory_ts(row["detect_time"]),
        confirm_time=_read_ts(row.get("confirm_time")),
        entry_time=_read_ts(row.get("entry_time")),
        known_at=_read_ts(row.get("known_at")),
        entry_price=_f("entry_price"),
        stop_price=_f("stop_price"),
        target_price=_opt_float("target_price"),
        rule_score=_f("rule_score"),
        model_prob=_opt_float("model_prob"),
        config_hash=str(row.get("config_hash", "")),
        feature_schema_version=str(row.get("feature_schema_version", "")),
        structure_levels=dict(levels) if isinstance(levels, dict) else {},
        attributes=dict(attrs) if isinstance(attrs, dict) else {},
        lifecycle_state=str(row.get("lifecycle_state", "")),
        confluence_group_id=None if not group_id else str(group_id),
    )


def events_from_frame(frame: pd.DataFrame) -> list[PatternEvent]:
    """Reconstruct PatternEvent dataclasses from an event frame."""
    return [_row_to_event(row) for _, row in frame.iterrows()]


def empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLUMNS)


# ---------------------------------------------------------------------------
# Outcome model (forward outcomes + hypothetical PnL)
# ---------------------------------------------------------------------------

@dataclass
class EventOutcome:
    """Forward outcome for one event (spec §7.1 / §7.2 ``outcomes/``).

    ``forward_return_r`` is the *realized* barrier/timeout return in R
    multiples; ``mfe_r`` / ``mae_r`` are the fixed-window excursions (same
    convention as the sweep plugin's ``labeling.excursions``); ``pnl_net`` is
    the hypothetical gross PnL minus per-unit costs (spread + commission +
    slippage, §6.5).
    """

    event_id: str
    pattern_name: str
    symbol: str
    timeframe: str
    entry_time: pd.Timestamp
    horizon: int
    trade_direction: str  # "long" | "short" (market direction from the event)
    exit_reason: str
    exit_price: float
    forward_return_r: float
    mfe_r: float
    mae_r: float
    mfe: float
    mae: float
    gross_pnl: float
    costs_total: float
    pnl_net: float
    pnl_net_r: float
    window_truncated: bool
    computed_at: pd.Timestamp


def outcomes_to_frame(outcomes: Sequence[EventOutcome]) -> pd.DataFrame:
    rows = [
        {
            "event_id": oc.event_id,
            "pattern_name": oc.pattern_name,
            "symbol": oc.symbol,
            "timeframe": oc.timeframe,
            "entry_time": _as_utc(oc.entry_time),
            "horizon": int(oc.horizon),
            "trade_direction": oc.trade_direction,
            "exit_reason": oc.exit_reason,
            "exit_price": float(oc.exit_price),
            "forward_return_r": float(oc.forward_return_r),
            "mfe_r": float(oc.mfe_r),
            "mae_r": float(oc.mae_r),
            "mfe": float(oc.mfe),
            "mae": float(oc.mae),
            "gross_pnl": float(oc.gross_pnl),
            "costs_total": float(oc.costs_total),
            "pnl_net": float(oc.pnl_net),
            "pnl_net_r": float(oc.pnl_net_r),
            "window_truncated": bool(oc.window_truncated),
            "computed_at": _as_utc(oc.computed_at),
        }
        for oc in outcomes
    ]
    return pd.DataFrame(rows, columns=_OUTCOME_COLUMNS)


def outcomes_from_frame(frame: pd.DataFrame) -> list[EventOutcome]:
    out: list[EventOutcome] = []
    for _, row in frame.iterrows():
        out.append(
            EventOutcome(
                event_id=str(row["event_id"]),
                pattern_name=str(row["pattern_name"]),
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                entry_time=_read_mandatory_ts(row["entry_time"]),
                horizon=int(row["horizon"]),
                trade_direction=str(row["trade_direction"]),
                exit_reason=str(row["exit_reason"]),
                exit_price=float(row["exit_price"]),
                forward_return_r=float(row["forward_return_r"]),
                mfe_r=float(row["mfe_r"]),
                mae_r=float(row["mae_r"]),
                mfe=float(row["mfe"]),
                mae=float(row["mae"]),
                gross_pnl=float(row["gross_pnl"]),
                costs_total=float(row["costs_total"]),
                pnl_net=float(row["pnl_net"]),
                pnl_net_r=float(row["pnl_net_r"]),
                window_truncated=bool(row["window_truncated"]),
                computed_at=_read_mandatory_ts(row["computed_at"]),
            )
        )
    return out


def empty_outcome_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTCOME_COLUMNS)


# ---------------------------------------------------------------------------
# Event Lake (writer + reader, append-only)
# ---------------------------------------------------------------------------

class EventLake:
    """Append-only event/outcome/metrics store on the §7.2 parquet layout.

    ``root`` defaults to ``trading_v3/event_lake``.  Every mutation is an
    append: re-appending an existing ``event_id`` (events/outcomes) or
    ``asof`` (metrics) raises :class:`EventLakeError`.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[2] / EVENT_LAKE_DIRNAME
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Path helpers (§7.2)
    # ------------------------------------------------------------------
    def events_partition_path(
        self, pattern: str, symbol: str, month: str
    ) -> Path:
        return self.root / EVENTS_SUBDIR / pattern / symbol / f"{month}.parquet"

    def outcome_path(self, event_id: str) -> Path:
        return self.root / OUTCOMES_SUBDIR / f"{event_id}.parquet"

    def metrics_path(self, model_id: str) -> Path:
        return self.root / METRICS_SUBDIR / f"{model_id}.parquet"

    # ------------------------------------------------------------------
    # Events — append-only writer + readers
    # ------------------------------------------------------------------
    def append_events(self, events: Sequence[PatternEvent]) -> int:
        """Append events to their month partitions (append-only).

        Raises :class:`EventLakeError` when any ``event_id`` already exists in
        its partition (rows are never overwritten) or when ``detect_time`` is
        missing.
        """
        if not events:
            return 0
        frame = events_to_frame(list(events))
        for _, row in frame.iterrows():
            detect = _as_utc(row["detect_time"])
            if pd.isna(detect):
                raise EventLakeError(
                    f"event {row['event_id']}: detect_time required for "
                    "month partitioning"
                )

        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for _, row in frame.iterrows():
            month = pd.Timestamp(row["detect_time"]).strftime(
                PARTITION_DATE_FORMAT
            )
            key = (str(row["pattern_name"]), str(row["symbol"]), month)
            groups.setdefault(key, []).append(dict(row))

        appended = 0
        for (pattern, symbol, month), rows in groups.items():
            path = self.events_partition_path(pattern, symbol, month)
            incoming = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
            if path.exists():
                existing = pd.read_parquet(path)
                dup = set(incoming["event_id"].astype(str)) & set(
                    existing["event_id"].astype(str)
                )
                if dup:
                    raise EventLakeError(
                        "append-only violation — event_id(s) already "
                        f"persisted: {sorted(dup)}"
                    )
                merged = pd.concat([existing, incoming], ignore_index=True)
            else:
                merged = incoming
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path, index=False)
            appended += len(incoming)
        return appended

    def read_events(
        self,
        pattern: str | None = None,
        symbol: str | None = None,
        from_ts: pd.Timestamp | None = None,
        to_ts: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read events, optionally filtered by pattern/symbol/time window.

        The time filter applies on ``detect_time`` (causal partition key).
        """
        if symbol is not None and pattern is None:
            raise EventLakeError(
                "read_events: symbol filter requires pattern (layout is "
                "events/{pattern}/{symbol}/)"
            )
        base = self.root / EVENTS_SUBDIR
        if pattern is not None:
            base = base / pattern
        if symbol is not None:
            base = base / symbol
        files = sorted(base.rglob("*.parquet")) if base.exists() else []
        frames = [pd.read_parquet(f) for f in files]
        if not frames:
            return empty_event_frame()
        out = pd.concat(frames, ignore_index=True)
        if from_ts is not None or to_ts is not None:
            dt = pd.to_datetime(out["detect_time"], utc=True)
            mask = pd.Series(True, index=out.index, dtype=bool)
            if from_ts is not None:
                mask &= dt >= from_ts
            if to_ts is not None:
                mask &= dt <= to_ts
            out = out[mask]
        return out.reset_index(drop=True)

    def events(
        self,
        pattern: str | None = None,
        symbol: str | None = None,
        from_ts: pd.Timestamp | None = None,
        to_ts: pd.Timestamp | None = None,
    ) -> list[PatternEvent]:
        return events_from_frame(
            self.read_events(pattern, symbol, from_ts, to_ts)
        )

    def event_ids(
        self,
        pattern: str | None = None,
        symbol: str | None = None,
    ) -> list[str]:
        frame = self.read_events(pattern, symbol)
        if frame.empty:
            return []
        return sorted(frame["event_id"].astype(str).tolist())

    # ------------------------------------------------------------------
    # Outcomes — one parquet per event_id (§7.2)
    # ------------------------------------------------------------------
    def append_outcomes(self, outcomes: Sequence[EventOutcome]) -> int:
        if not outcomes:
            return 0
        for oc in outcomes:
            path = self.outcome_path(oc.event_id)
            if path.exists():
                raise EventLakeError(
                    f"append-only violation — outcome for event_id "
                    f"{oc.event_id!r} already persisted"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            outcomes_to_frame([oc]).to_parquet(path, index=False)
        return len(outcomes)

    def read_outcomes(
        self, event_ids: Sequence[str] | None = None
    ) -> pd.DataFrame:
        base = self.root / OUTCOMES_SUBDIR
        if event_ids is not None:
            paths = [
                self.outcome_path(eid)
                for eid in event_ids
                if self.outcome_path(eid).exists()
            ]
        else:
            paths = sorted(base.rglob("*.parquet")) if base.exists() else []
        frames = [pd.read_parquet(p) for p in paths]
        if not frames:
            return empty_outcome_frame()
        return pd.concat(frames, ignore_index=True).reset_index(drop=True)

    def outcomes(
        self, event_ids: Sequence[str] | None = None
    ) -> list[EventOutcome]:
        return outcomes_from_frame(self.read_outcomes(event_ids))

    def pending_event_ids(
        self,
        pattern: str | None = None,
        symbol: str | None = None,
    ) -> list[str]:
        """Event ids that have no outcome row yet (feed the tracker job)."""
        all_ids = self.event_ids(pattern, symbol)
        if not all_ids:
            return []
        have = set(self.read_outcomes().get("event_id", pd.Series(dtype=str)).astype(str))
        return [eid for eid in all_ids if eid not in have]

    # ------------------------------------------------------------------
    # Metrics — rolling metrics per model_id (§5.5 / §7.2)
    # ------------------------------------------------------------------
    def append_metrics(self, model_id: str, rows: pd.DataFrame) -> int:
        """Append rolling-metric rows for *model_id* (append-only by ``asof``).

        ``rows`` must contain an ``asof`` timestamp column (one snapshot per
        timestamp); duplicate ``asof`` raises :class:`EventLakeError`.
        """
        if rows.empty:
            return 0
        if "asof" not in rows.columns:
            raise EventLakeError("append_metrics: rows need an 'asof' column")
        incoming = rows.copy()
        incoming["asof"] = pd.to_datetime(incoming["asof"], utc=True)
        path = self.metrics_path(model_id)
        if path.exists():
            existing = pd.read_parquet(path)
            existing_dt = pd.to_datetime(existing["asof"], utc=True)
            dup = set(incoming["asof"]) & set(existing_dt)
            if dup:
                raise EventLakeError(
                    f"append-only violation — metrics snapshot(s) for "
                    f"{model_id!r} already exist at {sorted(dup)}"
                )
            merged = pd.concat([existing, incoming], ignore_index=True)
        else:
            merged = incoming
        merged = merged.sort_values("asof").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        return len(incoming)

    def read_metrics(self, model_id: str) -> pd.DataFrame:
        path = self.metrics_path(model_id)
        if not path.exists():
            return pd.DataFrame(columns=_METRICS_COLUMNS)
        frame = pd.read_parquet(path)
        return frame.sort_values("asof").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Attribute flattening + feature discovery (drift / retrain support)
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_attributes(
        frame: pd.DataFrame, drop_json: bool = True
    ) -> pd.DataFrame:
        """Expand ``attributes_json`` into primitive columns (§7.3 mining).

        Nested dicts become dotted columns (``features.atr1``); base event
        column names win collisions (attribute key suffixed ``_attr``).
        """
        if frame.empty:
            return frame.copy()
        base_cols = set(_EVENT_COLUMNS)
        rows_out: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            d = dict(row)
            attrs_raw = d.get("attributes_json", "")
            attrs = (
                json.loads(attrs_raw)
                if isinstance(attrs_raw, str) and attrs_raw
                else {}
            )
            if isinstance(attrs, dict):
                for name, val in _flatten_dict(attrs).items():
                    col = name if name not in base_cols else f"{name}_attr"
                    d[col] = val
            if drop_json:
                d.pop("attributes_json", None)
            rows_out.append(d)
        out = pd.DataFrame(rows_out)
        return out

    def feature_columns(
        self,
        pattern: str | None = None,
        symbol: str | None = None,
    ) -> list[str]:
        """Discover numeric feature columns stored in event attributes."""
        frame = self.read_events(pattern, symbol)
        flat = self.flatten_attributes(frame)
        return [
            str(col)
            for col in flat.columns
            if col not in _EVENT_COLUMNS and flat[col].dtype.kind in ("f", "i", "u", "b")
        ]

    # ------------------------------------------------------------------
    # Query API (§7.3.1): rebuild a training dataset without re-running a
    # detector — events + flattened features + outcome labels + costs.
    # ------------------------------------------------------------------
    def rebuild_dataset(
        self,
        pattern: str,
        symbol: str,
        from_ts: pd.Timestamp | None = None,
        to_ts: pd.Timestamp | None = None,
        require_completed_outcome: bool = True,
        flatten_attributes: bool = True,
    ) -> pd.DataFrame:
        """Reassemble a labeled dataset from the lake (spec §7.3.1).

        Joins events and outcomes on ``event_id``.  With
        ``require_completed_outcome`` (default) only rows whose forward window
        fully elapsed (no truncation, exit_reason != ``no_data``) are kept —
        the only rows fit for training.  Outcome column ``entry_time`` /
        ``pattern_name`` / ... collide with event columns and get the
        ``_outcome`` suffix; other outcome metrics keep their names.
        """
        events = self.read_events(pattern, symbol, from_ts, to_ts)
        if events.empty:
            return empty_event_frame()
        if flatten_attributes:
            events = self.flatten_attributes(events)
        eids = events["event_id"].astype(str).tolist()
        oc = self.read_outcomes(eids)
        if oc.empty:
            merged = events.assign(
                horizon=pd.NA,
                trade_direction=pd.NA,
                exit_reason=pd.NA,
                exit_price=float("nan"),
                forward_return_r=float("nan"),
                mfe_r=float("nan"),
                mae_r=float("nan"),
                mfe=float("nan"),
                mae=float("nan"),
                gross_pnl=float("nan"),
                costs_total=float("nan"),
                pnl_net=float("nan"),
                pnl_net_r=float("nan"),
                window_truncated=pd.NA,
                computed_at=pd.NA,
            )
        else:
            merged = events.merge(
                oc,
                on="event_id",
                how="left",
                suffixes=("", "_outcome"),
            )
        if require_completed_outcome:
            trunc = merged["window_truncated"]
            merged = merged[
                trunc.notna() & ~trunc.fillna(True) & (merged["exit_reason"] != EXIT_NO_DATA)
            ]
        return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outcome tracker — periodic job: forward outcomes + hypothetical PnL (§7.1)
# ---------------------------------------------------------------------------

def _resolve_costs_total(
    symbol: str, costs: Mapping[str, Mapping[str, float]] | None
) -> float:
    if not costs:
        return 0.0
    cfg = costs.get(symbol)
    if not cfg:
        return 0.0
    return float(cfg.get("spread", 0.0)) + float(cfg.get("commission", 0.0)) + float(
        cfg.get("slippage", 0.0)
    )


def _entry_position(
    index: pd.DatetimeIndex, entry_ts: pd.Timestamp
) -> int:
    """Position of the first bar at/after *entry_ts* (UTC-naive compare)."""
    idx: pd.DatetimeIndex = index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    entry = entry_ts
    if entry.tz is not None:
        entry = entry.tz_convert("UTC").tz_localize(None)
    times = idx.to_numpy()
    return int(np.searchsorted(times, np.datetime64(entry), side="left"))


def _compute_single_outcome(
    ev: PatternEvent,
    highs: np.ndarray[Any, Any],
    lows: np.ndarray[Any, Any],
    closes: np.ndarray[Any, Any],
    index: pd.DatetimeIndex,
    horizon: int,
    costs_total: float,
    computed_at: pd.Timestamp,
) -> EventOutcome:
    entry_ts = ev.entry_time if ev.entry_time is not None else ev.known_at_ts
    direction = "long" if ev.direction == "bullish" else "short"
    entry = float(ev.entry_price)
    stop = float(ev.stop_price)
    target = (
        float(ev.target_price) if ev.target_price is not None else float("nan")
    )
    risk = abs(entry - stop) if math.isfinite(stop) else float("nan")
    n = len(highs)
    pos = _entry_position(index, entry_ts)

    nan_metrics: dict[str, Any] = dict(
        event_id=ev.event_id,
        pattern_name=ev.pattern_name,
        symbol=ev.symbol,
        timeframe=ev.timeframe,
        entry_time=entry_ts,
        horizon=horizon,
        trade_direction=direction,
        exit_reason=EXIT_NO_DATA,
        exit_price=float("nan"),
        forward_return_r=float("nan"),
        mfe_r=float("nan"),
        mae_r=float("nan"),
        mfe=float("nan"),
        mae=float("nan"),
        gross_pnl=float("nan"),
        costs_total=costs_total,
        pnl_net=float("nan"),
        pnl_net_r=float("nan"),
        window_truncated=True,
        computed_at=computed_at,
    )
    if pos >= n or pd.isna(entry_ts):
        return EventOutcome(**nan_metrics)

    end = min(pos + horizon - 1, n - 1)  # inclusive last window bar
    window_truncated = end - pos + 1 < horizon
    w_high = highs[pos : end + 1]
    w_low = lows[pos : end + 1]
    w_close = closes[pos : end + 1]

    if direction == "long":
        mfe = float(np.max(w_high)) - entry
        mae = entry - float(np.min(w_low))
    else:
        mfe = entry - float(np.min(w_low))
        mae = float(np.max(w_high)) - entry

    # Triple-barrier walk: post-entry bars, conservative intra-bar ordering
    # (both hit within a bar -> counted as stop hit).
    exit_reason = EXIT_TIMEOUT
    exit_price = float(w_close[-1])
    for i in range(pos + 1, end + 1):
        hi = float(highs[i])
        lo = float(lows[i])
        if direction == "long":
            hit_tp = math.isfinite(target) and hi >= target
            hit_sl = math.isfinite(stop) and lo <= stop
        else:
            hit_tp = math.isfinite(target) and lo <= target
            hit_sl = math.isfinite(stop) and hi >= stop
        if hit_tp and hit_sl:
            exit_reason, exit_price = EXIT_STOP_HIT, stop
        elif hit_tp:
            exit_reason, exit_price = EXIT_TARGET_HIT, target
        elif hit_sl:
            exit_reason, exit_price = EXIT_STOP_HIT, stop
        if exit_reason != EXIT_TIMEOUT:
            break

    gross = (exit_price - entry) if direction == "long" else (entry - exit_price)
    r_div = risk if math.isfinite(risk) and risk > 0 else float("nan")

    def _r(val: float) -> float:
        return float(val / r_div) if math.isfinite(r_div) else float("nan")

    return EventOutcome(
        event_id=ev.event_id,
        pattern_name=ev.pattern_name,
        symbol=ev.symbol,
        timeframe=ev.timeframe,
        entry_time=entry_ts,
        horizon=horizon,
        trade_direction=direction,
        exit_reason=exit_reason,
        exit_price=exit_price,
        forward_return_r=_r(gross),
        mfe_r=_r(mfe),
        mae_r=_r(mae),
        mfe=mfe,
        mae=mae,
        gross_pnl=gross,
        costs_total=costs_total,
        pnl_net=gross - costs_total,
        pnl_net_r=_r(gross - costs_total),
        window_truncated=window_truncated,
        computed_at=computed_at,
    )


def _as_frame_with_index(candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Validate an OHLCV frame and return a DatetimeIndex-ed copy."""
    for col in ("open", "high", "low", "close"):
        if col not in candles.columns:
            raise EventLakeError(
                f"outcome tracker: candles frame missing '{col}' column"
            )
    index = pd.DatetimeIndex(candles.index)
    if index.has_duplicates:
        raise EventLakeError("outcome tracker: candles index must be unique")
    out = candles.sort_index()
    return out, pd.DatetimeIndex(out.index)


class OutcomeTracker:
    """Computes forward outcomes + hypothetical PnL (minus costs, §6.5).

    ``candles`` is the (datetime-indexed OHLC) frame *including* future bars
    relative to each event — by design the tracker is a post-hoc labeling job
    (matched to the sweep plugin's fixed-window convention: window =
    ``entry_pos .. entry_pos + horizon - 1`` inclusive).
    """

    def __init__(
        self,
        lake: EventLake,
        horizon: int = 16,
        costs: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.lake = lake
        self.horizon = horizon
        self.costs: Mapping[str, Mapping[str, float]] = (
            dict(costs) if costs is not None else dict(DEFAULT_COSTS)
        )

    def compute_outcomes(
        self,
        events: Sequence[PatternEvent],
        candles: pd.DataFrame,
        horizon: int | None = None,
        costs: Mapping[str, Mapping[str, float]] | None = None,
    ) -> list[EventOutcome]:
        frame, index = _as_frame_with_index(candles)
        h = self.horizon if horizon is None else horizon
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        cost_cfg = self.costs if costs is None else costs
        computed_at = _as_utc(pd.Timestamp.now(tz="UTC"))
        highs = frame["high"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
        return [
            _compute_single_outcome(
                ev,
                highs,
                lows,
                closes,
                index,
                h,
                _resolve_costs_total(ev.symbol, cost_cfg),
                computed_at,
            )
            for ev in events
        ]

    def run(
        self,
        candles_provider: Callable[[str, str, str], pd.DataFrame],
        horizon: int | None = None,
        costs: Mapping[str, Mapping[str, float]] | None = None,
    ) -> int:
        """Periodic job: compute outcomes for every pending event and append.

        ``candles_provider(pattern_name, symbol, timeframe) -> DataFrame`` is
        how the job fetches market data (spec §9.1 ``fetch``).  Re-running the
        job is a no-op for already-computed events (append-only lake).
        """
        pending = self.lake.pending_event_ids()
        if not pending:
            return 0
        frame = self.lake.read_events()
        frame = frame[frame["event_id"].astype(str).isin(pending)]
        if frame.empty:
            return 0
        events = events_from_frame(frame)

        groups: dict[tuple[str, str, str], list[PatternEvent]] = {}
        for ev in events:
            groups.setdefault((ev.pattern_name, ev.symbol, ev.timeframe), []).append(ev)

        outcomes: list[EventOutcome] = []
        for (pattern, symbol, timeframe), evs in groups.items():
            candles = candles_provider(pattern, symbol, timeframe)
            outcomes.extend(
                self.compute_outcomes(evs, candles, horizon=horizon, costs=costs)
            )
        self.lake.append_outcomes(outcomes)
        return len(outcomes)


# ---------------------------------------------------------------------------
# Drift monitor — PSI per feature (live vs train) for LifecycleManager (§5.3)
# ---------------------------------------------------------------------------

def population_stability_index(
    expected: Sequence[float] | np.ndarray[Any, Any],
    actual: Sequence[float] | np.ndarray[Any, Any],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between two feature samples.

    ``expected`` defines the reference bins (train distribution); ``actual``
    is the live sample.  Returns 0.0 (no drift evidence) when either sample is
    too small or ``expected`` has zero variance — a periodic job must not blow
    up on degenerate slices.
    """
    exp = np.asarray(expected, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = np.asarray(actual, dtype=float)
    act = act[np.isfinite(act)]
    if len(exp) < 2 or len(act) < 1:
        return 0.0
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_hist, _ = np.histogram(exp, bins=edges)
    act_hist, _ = np.histogram(act, bins=edges)
    exp_sum = float(exp_hist.sum())
    act_sum = float(act_hist.sum())
    exp_pct = exp_hist / exp_sum if exp_sum > 0 else exp_hist
    act_pct = act_hist / act_sum if act_sum > 0 else act_hist
    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


@dataclass
class FeatureDrift:
    """PSI result for one feature."""

    feature: str
    psi: float
    severity: str  # "none" | "moderate" | "drift"


@dataclass
class DriftReport:
    """§5.3 drift verdict consumed by LifecycleManager."""

    per_feature: list[FeatureDrift]
    drift_threshold: float = 0.2
    demote_threshold: float = 0.25
    min_drift_features: int = 2

    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.per_feature if f.severity == "drift"]

    @property
    def n_drift(self) -> int:
        return len(self.drifted_features())

    @property
    def requires_retrain(self) -> bool:
        """PSI >= 0.2 on >= ``min_drift_features`` core features (§5.3)."""
        return self.n_drift >= self.min_drift_features

    @property
    def demote_to_degraded(self) -> bool:
        """Any single feature PSI >= 0.25 -> live -> degraded (§5.3)."""
        return any(f.psi >= self.demote_threshold for f in self.per_feature)

    def summary(self) -> str:
        if not self.per_feature:
            return "drift: no features measured"
        worst = max(self.per_feature, key=lambda f: f.psi)
        verdict = "RETRAIN" if self.requires_retrain else (
            "DEGRADE" if self.demote_to_degraded else "ok"
        )
        return (
            f"drift[{verdict}]: {self.n_drift}/{len(self.per_feature)} "
            f"features drifted (>= {self.drift_threshold}); worst "
            f"{worst.feature}={worst.psi:.3f}"
        )


class DriftMonitor:
    """PSI per core feature (live vs train) — spec §5.2/§5.3.

    Thresholds follow §5.3: PSI < 0.1 none, 0.1-0.2 moderate, >= 0.2 drift;
    ``requires_retrain`` when >= ``min_drift_features`` features drift;
    ``demote_to_degraded`` when any feature PSI >= 0.25.
    """

    def __init__(
        self,
        bins: int = 10,
        drift_threshold: float = 0.2,
        demote_threshold: float = 0.25,
        min_drift_features: int = 2,
        epsilon: float = 1e-6,
    ) -> None:
        self.bins = bins
        self.drift_threshold = drift_threshold
        self.demote_threshold = demote_threshold
        self.min_drift_features = min_drift_features
        self.epsilon = epsilon

    def _severity(self, psi: float) -> str:
        if psi < 0.1:
            return "none"
        if psi < self.drift_threshold:
            return "moderate"
        return "drift"

    def feature_psi(
        self,
        train: pd.DataFrame,
        live: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> dict[str, float]:
        present = [c for c in feature_cols if c in train.columns and c in live.columns]
        results: dict[str, float] = {}
        for col in present:
            results[col] = population_stability_index(
                train[col].to_numpy(dtype=float),
                live[col].to_numpy(dtype=float),
                bins=self.bins,
                epsilon=self.epsilon,
            )
        return results

    def check(
        self,
        train: pd.DataFrame,
        live: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> DriftReport:
        """Compute a §5.3 drift report over *feature_cols*."""
        present = [c for c in feature_cols if c in train.columns and c in live.columns]
        if not present:
            raise EventLakeError(
                "DriftMonitor.check: none of the requested feature columns "
                f"{list(feature_cols)} exist in train/live frames"
            )
        per_feature = [
            FeatureDrift(
                feature=col,
                psi=population_stability_index(
                    train[col].to_numpy(dtype=float),
                    live[col].to_numpy(dtype=float),
                    bins=self.bins,
                    epsilon=self.epsilon,
                ),
                severity=self._severity(
                    population_stability_index(
                        train[col].to_numpy(dtype=float),
                        live[col].to_numpy(dtype=float),
                        bins=self.bins,
                        epsilon=self.epsilon,
                    )
                ),
            )
            for col in present
        ]
        return DriftReport(
            per_feature=per_feature,
            drift_threshold=self.drift_threshold,
            demote_threshold=self.demote_threshold,
            min_drift_features=self.min_drift_features,
        )

    def check_lake_drift(
        self,
        lake: EventLake,
        pattern: str,
        symbol: str,
        feature_cols: Sequence[str],
        train_from: pd.Timestamp,
        train_to: pd.Timestamp,
        live_from: pd.Timestamp,
        live_to: pd.Timestamp,
    ) -> DriftReport:
        """Periodic drift job: train vs live feature distributions straight
        from the lake (spec §7.3.3) — no detector, no re-labeling."""
        train = lake.rebuild_dataset(
            pattern,
            symbol,
            from_ts=train_from,
            to_ts=train_to,
            require_completed_outcome=False,
        )
        live = lake.rebuild_dataset(
            pattern,
            symbol,
            from_ts=live_from,
            to_ts=live_to,
            require_completed_outcome=False,
        )
        return self.check(train, live, feature_cols)


__all__ = [
    "DEFAULT_COSTS",
    "EVENTS_SUBDIR",
    "EVENT_LAKE_DIRNAME",
    "EXIT_NO_DATA",
    "EXIT_STOP_HIT",
    "EXIT_TARGET_HIT",
    "EXIT_TIMEOUT",
    "METRICS_SUBDIR",
    "OUTCOMES_SUBDIR",
    "DriftMonitor",
    "DriftReport",
    "EventLake",
    "EventLakeError",
    "EventOutcome",
    "FeatureDrift",
    "OutcomeTracker",
    "events_from_frame",
    "events_to_frame",
    "outcomes_from_frame",
    "outcomes_to_frame",
    "population_stability_index",
]