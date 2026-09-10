"""
runner.py — Multi-Pattern Backtest Runner (SPEC v1.1 §12)

Replays the *live* multi-pattern code path against historical OHLCV:

    detect (pattern_registry detectors) -> validate_causality (§3.4)
        -> attach model_prob (t1 tier-2 artifacts, when available)
        -> CorrelationManager.group (§4.2 — SAME class imported from
           live.engine, no duplicated grouping logic)
        -> §4.4 exposure caps (CorrelationManager.check_exposure_caps)
        -> per-group representative trade simulated forward with costs
           (spread / commission / slippage)

Backtest ≡ live: the runner calls the exact ``live.engine``
``CorrelationManager`` / ``CorrelationConfig`` / ``ResolvedGroup`` objects
the MultiPatternEngine uses, so identical input yields identical group
decisions (pinned by the parity test in tests/test_multi_backtest.py).

Causality (§3.4): every ``PatternEvent`` is consumed at its ``known_at``
only — detection runs ONCE over the full (warmed-up) frame with each
detector's own causal feature computation, then events are replayed in
``known_at`` order.  No future bar is ever read to decide or size a trade.

CLI::

    python -m research.multi_backtest.runner \\
        --symbols XAUUSD,EURUSD \\
        --patterns liquidity_sweep,double_bottom,double_top \\
        --start 2023-01-01 --end 2026-09-03 --mode dedup \\
        --out research/multi_backtest/reports
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from live.engine.pattern_registry import get_registry
from research.core.contracts import (
    LIFECYCLE_LIVE,
    PatternEvent,
)

# ---------------------------------------------------------------------------
# §12 mandate: reuse the SAME correlation code path as live — import only.
# (The runtime bridge imports the identical live classes; a static import is
# avoided so mypy strict stays scoped to research/multi_backtest per handoff
# lesson #2 — live/ carries pre-existing non-strict code, out of t4 scope.)
# ---------------------------------------------------------------------------
from research.multi_backtest.live_corr import (
    CorrelationConfig,
    CorrelationManager,
    ResolvedGroup,
)

logger = logging.getLogger("multi_backtest.runner")

#: Package root (trading_v3/research/multi_backtest/).
_ROOT = Path(__file__).resolve().parent
#: Default candle-data directory (self-contained copies of the M15 parquets).
DATA_DIR = _ROOT / "data"
#: Default report output directory.
REPORTS_DIR = _ROOT / "reports"

#: Default warm-up bars before the report window (indicators + pattern
#: geometry need history; detection stays causal — no bar ≥ known_at used).
DEFAULT_WARMUP_BARS = 1500
#: Default forward horizon (bars) when an event carries no model horizon.
DEFAULT_HORIZON_BARS = 72

#: §4.4 cap defaults (risk-normalised units) — mirror CorrelationConfig.
DEFAULT_MAX_TOTAL_RISK = 1.0
DEFAULT_MAX_CLUSTER_RISK = 0.75

#: Per-symbol default trading costs (bps).  A bps = 1e-4 of price per side.
DEFAULT_COSTS: dict[str, dict[str, float]] = {
    "XAUUSD": {"spread_bps": 0.7, "commission_bps": 0.5, "slippage_bps": 1.0},
    "EURUSD": {"spread_bps": 0.5, "commission_bps": 0.5, "slippage_bps": 1.0},
    "__default__": {"spread_bps": 1.0, "commission_bps": 0.5, "slippage_bps": 1.0},
}

PATTERNS_XAUUSD: list[str] = ["liquidity_sweep", "double_bottom", "double_top"]


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Trading costs per symbol (bps of price, applied per side: entry+exit)."""

    spread_bps: float = 1.0
    commission_bps: float = 0.5
    slippage_bps: float = 1.0

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.commission_bps + self.slippage_bps

    def cost_r(self, entry_price: float, risk: float) -> float:
        """Two-sided cost converted into R-multiples of the trade risk.

        cost_R = 2 sides * (total_bps / 1e4) * entry_price / risk.
        ``0.0`` when the risk distance is non-positive (trade not executable).
        """
        if risk <= 0 or not np.isfinite(risk) or not np.isfinite(entry_price):
            return 0.0
        per_side = (self.total_bps / 1e4) * abs(entry_price)
        return 2.0 * per_side / risk

    def to_dict(self) -> dict[str, float]:
        return {
            "spread_bps": self.spread_bps,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
        }

    @classmethod
    def for_symbol(cls, symbol: str, overrides: dict[str, Any] | None = None) -> CostConfig:
        spec = DEFAULT_COSTS.get(symbol, DEFAULT_COSTS["__default__"])
        merged = dict(spec)
        if overrides:
            extra = overrides.get(symbol) or overrides.get("__default__")
            if isinstance(extra, dict):
                for k in ("spread_bps", "commission_bps", "slippage_bps"):
                    if k in extra and extra[k] is not None:
                        merged[k] = float(extra[k])
        return cls(
            spread_bps=float(merged["spread_bps"]),
            commission_bps=float(merged["commission_bps"]),
            slippage_bps=float(merged["slippage_bps"]),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _as_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _index_pos(index: pd.Index[Any], ts: str | pd.Timestamp, side: str) -> int:
    """Bar position of the first timestamp >= ts (``side='left'``) or > ts
    (``'right'``).  pandas-stubs type the DatetimeIndex searchsorted return as
    a union — cast to ``Any`` keeps mypy strict satisfied."""
    from typing import Literal, cast

    left_right: Literal["left", "right"] = "left" if side == "left" else "right"
    pos = cast(Any, index.searchsorted(_as_utc(ts), side=left_right))
    return int(pos)


def load_symbol_frame(
    symbol: str,
    data_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    sim_extra_bars: int = DEFAULT_HORIZON_BARS,
) -> pd.DataFrame:
    """Load a symbol's M15 OHLCV frame (DatetimeIndex, sorted, deduped).

    Reads the self-contained parquet under ``research/multi_backtest/data/``;
    falls back to the pattern plugin dataset locations.  When ``start`` is
    given, ``warmup_bars`` of leading history are retained before it so
    indicators / pattern geometry are already computed (still causal).  The
    right edge is extended by ``sim_extra_bars`` beyond ``end`` so forward
    simulation of the last report events has bars (events are still consumed
    at ``known_at <= end`` only).
    """
    data_dir = data_dir or DATA_DIR
    cands = [
        data_dir / f"{symbol.lower()}_m15.parquet",
        _ROOT.parents[1] / "research" / "patterns" / "liquidity_sweep" / "data"
        / "processed" / f"{symbol.lower()}_m15.parquet",
    ]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No M15 parquet for {symbol} — expected one of "
            + ", ".join(str(p) for p in cands)
        )
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype(float)
    if start is not None:
        lo = max(0, _index_pos(df.index, start, "left") - warmup_bars)
        df = df.iloc[lo:]
    if end is not None:
        pos = _index_pos(df.index, end, "right")
        # extend right by sim_extra_bars (M15 cadence) for forward simulation
        pos = min(len(df), pos + sim_extra_bars)
        df = df.iloc[:pos]
    return df


# ---------------------------------------------------------------------------
# Tier-2 model attachment (t1 artifacts, when available)
# ---------------------------------------------------------------------------

def _artifact_dir(pattern: str, symbol: str, timeframe: str) -> Path | None:
    """Locate ``models/{pattern}_{symbol}_{tf}_v1`` under the pattern plugin.

    Returns ``None`` when the pattern has no trained artifact (e.g. LSW, or
    any pattern on a symbol t1 did not train — XAUUSD-only artifacts).
    """
    tf_dir = f"{pattern}_{symbol.lower()}_{timeframe.lower()}_v1"
    pkg = _ROOT.parent / "patterns" / pattern / "artifacts" / "models" / tf_dir
    if (pkg / "model.pkl").exists() and (pkg / "features.json").exists():
        return pkg
    return None


def make_tier2_scorer(
    pattern: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> Callable[[list[PatternEvent]], dict[str, float]] | None:
    """Build the §6.4 scorer closure over t1 artifacts (model.pkl envelope).

    Features are built with the trainer's own causal ``build_feature_frame``
    (bars ≤ confirm bar) so training == inference contract holds; the
    calibrated classifier scores rows in ``envelope["feature_names"]`` order.

    Returns ``None`` when no artifact is available for (pattern, symbol, TF).
    """
    import joblib  # type: ignore[import-untyped]

    art = _artifact_dir(pattern, symbol, timeframe)
    if art is None:
        return None
    try:
        envelope = joblib.load(str(art / "model.pkl"))
    except Exception as exc:  # corrupt artifact → never crash the backtest
        logger.warning("multi_backtest: artifact %s unreadable: %s", art, exc)
        return None
    model = envelope.get("model")
    feature_names = list(envelope.get("feature_names") or [])
    if model is None or not feature_names:
        return None

    from research.core.walkforward_trainer import build_feature_frame

    hmm_cols = [c for c in feature_names if c.startswith("hmm_")]

    def scorer(events: list[PatternEvent]) -> dict[str, float]:
        try:
            X, _names = build_feature_frame(df, events)
        except Exception as exc:
            logger.warning("multi_backtest: feature build failed for %s: %s", pattern, exc)
            return {}
        if X.empty:
            return {}
        # A model trained WITH hmm_* features reads them straight from the
        # emitted event attributes (guide §4.2B / requirements §4.2).  Old
        # models (no hmm_*) never reach here — reindex keeps them unmodified.
        if hmm_cols:
            attrs_by_id = {
                str(getattr(ev, "event_id", "")): getattr(ev, "attributes", {}) or {}
                for ev in events
            }
            for col in hmm_cols:
                X[col] = [
                    _attr_num(attrs_by_id.get(str(eid), {}).get(col))
                    for eid in X.index
                ]
        X = X.reindex(columns=feature_names).fillna(0.0).astype(float)
        try:
            proba = model.predict_proba(X.to_numpy(dtype=float))
        except Exception as exc:
            logger.warning("multi_backtest: predict failed for %s: %s", pattern, exc)
            return {}
        col = 1 if proba.shape[1] >= 2 else 0
        return {str(eid): float(p) for eid, p in zip(X.index, proba[:, col])}

    return scorer


def _attr_num(value: Any) -> float:
    """Numeric coercion of an attribute value; NaN when missing/unusable."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_bt_slot(
    assignment: BacktestAssignment,
    shared_plugin: Any = None,
    config_source: dict[str, Any] | None = None,
) -> Any:
    """Build a :class:`live.engine.regime_wiring.RegimeSlot` for a backtest
    assignment (or ``None`` when no regime wiring is wanted — default OFF).

    Mirrors the live ``MultiPatternEngine._assign_regime_slots`` path so
    live ≡ backtest share the exact same emitter/gate decisions (requirements
    §5): the resolution rules, the rule map, and the default-``None`` slot
    are all identical.
    """
    from live.engine.regime_wiring import (
        RegimeSlot,
        resolve_regime_config,
        resolve_regime_wiring,
    )

    if assignment.regime_states is not None:
        slot = RegimeSlot(
            plugin=assignment.regime_plugin or shared_plugin,
            emitter_enabled=assignment.emitter_enabled,
            gate_enabled=assignment.gate_enabled,
            rules=assignment.regime_filter or {},
            hmm_config=assignment.regime_config or {},
            assignment_id=assignment.assignment_id,
            pattern_name=assignment.pattern_name,
            model_feature_list=assignment.model_feature_list,
            model_optional_plugins=assignment.model_optional_plugins,
        )
        slot.set_states(list(assignment.regime_states))
        return slot
    plugin = assignment.regime_plugin or shared_plugin
    if plugin is None:
        return None
    src = config_source or {}
    resolved = dict(assignment.regime_config) if assignment.regime_config else (
        resolve_regime_config(
            # plugin section carries the master ``enabled`` switch — pass it
            # through so backtest wiring honours plugin.enabled from YAML
            # (mirrors MultiPatternEngine._assign_regime_slots).
            src.get("plugin") if isinstance(src.get("plugin"), dict) else None,
            src.get("hmm") if isinstance(src.get("hmm"), dict) else None,
            src.get("regime_filter") if isinstance(src.get("regime_filter"), dict) else None,
            assignment.pattern_name,
        )
        if src
        else {}
    )
    master = resolved.get("enabled")
    if master is None:
        resolved["enabled"] = bool(
            assignment.emitter_enabled
            or assignment.gate_enabled
            or resolved.get("rules")
            or (assignment.regime_filter is not None)
        )
    resolved["emitter_enabled"] = bool(
        assignment.emitter_enabled or resolved.get("emitter_enabled", False)
    )
    resolved["gate_enabled"] = bool(
        assignment.gate_enabled or resolved.get("gate_enabled", False)
    )
    if assignment.regime_filter is not None:
        resolved["rules"] = assignment.regime_filter
    return resolve_regime_wiring(
        plugin,
        resolved,
        assignment.assignment_id,
        assignment.pattern_name,
        model_feature_list=assignment.model_feature_list,
        model_optional_plugins=assignment.model_optional_plugins,
    )


def attach_regime_features(
    assignment: Any,
    events: list[PatternEvent],
    df: pd.DataFrame,
    slot: Any,
) -> bool:
    """§4.2 feature emitter for a backtest assignment (causal, default OFF).

    When the slot's emitter OR hard gate is active, computes the causal
    states once (cached on the slot) so the replay loop's gate can use them;
    injects hmm_* attributes only when the feature emitter is enabled.  The
    state at each event's known_at is looked up causally — never future bars;
    never changes an old model's inference.

    Returns ``True`` when scoring may proceed, ``False`` when the feature
    emitter was ACTIVE but computation failed and the model REQUIRES the
    plugin (fail-closed §6.3) — the events are stamped
    ``discard_reason="hmm_unavailable"`` and the caller must skip tier-2
    scoring (mirrors ``MultiPatternEngine._run_assignment``).
    """
    if not events or slot is None:
        return True
    if not (slot.emitter_active() or slot.gate_active()):
        return True
    try:
        states = slot.states()
        if states is None:
            states = slot.plugin.predict(df, slot.hmm_config or None)
            slot.set_states(list(states))
        if slot.emitter_active():
            slot.emitter.attach(events, states, df)
    except Exception as exc:
        logger.warning("multi_backtest: regime computation failed for %s: %s",
                       assignment.pattern_name, exc)
        if slot.emitter_active() and _bt_model_requires_plugin(assignment):
            for e in events:
                e.attributes["discard_reason"] = "hmm_unavailable"  # §6.2 / §7.1
            return False
    return True


def _bt_model_requires_plugin(assignment: Any) -> bool:
    """§6.3 runner-side mirror of the live engine's ``_model_requires_plugin``:
    the model was trained WITH ``hmm_*`` features and lists a required
    ``hmm_regime`` optional plugin (registry §4.3)."""
    from live.engine.feature_emitter import requires_hmm_regime

    has_hmm = any(str(f).startswith("hmm_") for f in (assignment.model_feature_list or []))
    return has_hmm and requires_hmm_regime(assignment.model_optional_plugins or [])


def gate_not_blocked(rep: PatternEvent, slot: Any, df: pd.DataFrame) -> bool:
    """Requirements §5 hard gate on a representative event (backtest side).

    Uses the SAME :func:`live.engine.hard_gate.is_allowed` the live
    engine calls (shared function → live ≡ backtest, pinned by the parity
    test).  A blocked event is annotated ``discard_reason="regime_blocked"``.
    """
    from live.engine.feature_emitter import state_at_confirm_bar
    from live.engine.hard_gate import is_allowed

    if slot is None or not slot.gate_active():
        return True
    states = slot.states()
    regime = state_at_confirm_bar(states, df, rep) if states else None
    allowed, _reason = is_allowed(rep, regime, slot.gate_rules())
    if not allowed:
        rep.attributes["discard_reason"] = "regime_blocked"
    return allowed


# ---------------------------------------------------------------------------
# Detection (mirrors MultiPatternEngine._run_assignment §9.1)
# ---------------------------------------------------------------------------

def detect_all(
    df: pd.DataFrame,
    assignments: list[Any],
    shared_plugin: Any = None,
    config_source: dict[str, Any] | None = None,
) -> list[PatternEvent]:
    """Run every assignment's detector once on the frame.

    Each detector is invoked with the full (causal) frame exactly like the
    live engine; events are stamped with their assignment lifecycle +
    timeframe, causality-validated (§3.4), HMM-regime feature-emitter
    attached (default OFF), and tier-2 scored when an artifact exists.
    Returns ALL events (consumed later at ``known_at``).
    """
    events_all: list[PatternEvent] = []
    for a in assignments:
        detector = getattr(a, "detector", None)
        if detector is None:
            continue
        # resolve the regime slot ONCE per assignment (shared live≡backtest rules)
        slot = _resolve_bt_slot(a, shared_plugin=shared_plugin, config_source=config_source)
        a._regime_slot = slot
        try:
            events = detector.detect(df, a.config)
        except Exception as exc:
            logger.warning("multi_backtest: %s detect failed: %s", a.pattern_name, exc)
            continue
        events = [e for e in events if e is not None]
        for e in events:
            e.lifecycle_state = a.state
            if e.timeframe in (None, ""):
                e.timeframe = a.timeframe
        try:
            detector.validate_causality(events)
        except Exception as exc:
            logger.warning(
                "multi_backtest: causality violation in %s — events dropped: %s",
                a.pattern_name, exc,
            )
            events = []
        # §6.3 fail-closed (mirror of live MultiPatternEngine): a model trained
        # WITH hmm_* + required hmm_regime must not be scored when the slot is
        # unavailable (no plugin wired / inactive).  Stamp the events and skip
        # the scorer so model_prob stays None (event persisted to the lake with
        # discard_reason — spec §7.1).
        if events and slot is None and _bt_model_requires_plugin(a):
            for e in events:
                e.attributes["discard_reason"] = "hmm_unavailable"  # §6.2 / §7.1
            logger.warning(
                "multi_backtest: %s requires hmm_regime but no regime plugin "
                "is wired — events marked hmm_unavailable (fail-closed §6.3)",
                a.pattern_name,
            )
        else:
            # HMM regime feature emitter (requirements §4.2) — only when enabled.
            # Returns False (→ skip scorer) when the emitter was active but
            # failed and the model requires the plugin (§6.3).
            scoring_ok = attach_regime_features(a, events, df, slot)
            scorer = getattr(a, "scorer", None)
            if scorer is not None and events and scoring_ok:
                try:
                    proba = scorer(events)
                    for e in events:
                        if e.event_id in proba and proba[e.event_id] is not None:
                            e.model_prob = float(proba[e.event_id])
                except Exception as exc:
                    logger.warning("multi_backtest: scorer failed for %s: %s",
                                   a.pattern_name, exc)
        events_all.extend(events)
    return events_all


@dataclass
class BacktestAssignment:
    """One symbol x pattern assignment (duck-compatible with the live
    ``PatternAssignment`` fields the engine accesses: detector / config /
    state / timeframe / scorer / risk_fraction / max_entry_drift_atr /
    current_market_price).  Kept local so runner.py stays mypy-strict clean
    without importing the legacy (non-strict) ``signal_engine_v2`` module.

    HMM regime fields (requirements v1.0 §4/§5) are optional and default OFF
    so a legacy backtest stays byte-identical; when ``regime_plugin`` is a
    fitted :class:`research.regime.CausalGaussianHMM` the emitter + hard gate
    run through the SAME research.regime functions the live engine uses.
    """

    assignment_id: str
    pattern_name: str
    timeframe: str
    state: str = LIFECYCLE_LIVE
    detector: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    scorer: Callable[[list[PatternEvent]], dict[str, float]] | None = None
    risk_fraction: float = 1.0
    max_entry_drift_atr: float = 0.0
    current_market_price: float | None = None
    # --- HMM regime plugin (default OFF) ---
    regime_plugin: Any = None
    regime_config: dict[str, Any] = field(default_factory=dict)
    regime_filter: dict[str, Any] | None = None
    model_feature_list: list[str] = field(default_factory=list)
    model_optional_plugins: list[dict[str, Any]] = field(default_factory=list)
    emitter_enabled: bool = False
    gate_enabled: bool = False
    regime_states: list[Any] | None = None


def build_assignments(
    symbol: str,
    patterns: list[str],
    timeframe: str = "M15",
    df: pd.DataFrame | None = None,
    state: str = LIFECYCLE_LIVE,
    attach_models: bool = True,
    max_entry_drift_atr: float = 0.0,
    # --- HMM regime plugin (requirements v1.0; default OFF) ---
    regime_plugin: Any = None,
    regime_config: dict[str, Any] | None = None,
    regime_filter: dict[str, Any] | None = None,
    model_optional_plugins: list[dict[str, Any]] | None = None,
) -> list[BacktestAssignment]:
    """Build one live-compatible assignment per pattern (detector from the
    pattern registry).

    Tier-2 scorers are attached from the t1 artifacts when available (only
    XAUUSD-trained patterns on the matching symbol).  ``max_entry_drift_atr=0``
    disables the §3.3 live drift guard by default (backtest executes at the
    recorded entry).  HMM regime wiring is OFF unless ``regime_plugin`` is a
    fitted regime plugin (then the emitter/hard-gate flags + rules apply).
    """
    registry = get_registry()
    assignments: list[BacktestAssignment] = []
    for pattern in patterns:
        cls = registry.get(pattern)
        if cls is None:
            logger.warning("multi_backtest: pattern %r not registered — skipped", pattern)
            continue
        # default config from a fresh instance (get_default_config is an
        # instance method), then override symbol/timeframe for the backtest
        probe = cls()
        with_cfg = (
            probe.get_default_config()
            if hasattr(probe, "get_default_config")
            else {}
        )
        config = dict(with_cfg)
        config["symbol"] = symbol
        config["timeframe"] = timeframe
        detector = cls(config)  # type: ignore[call-arg]  # detectors accept config
        scorer = None
        if attach_models and df is not None:
            scorer = make_tier2_scorer(pattern, symbol, timeframe, df)
        assignments.append(
            BacktestAssignment(
                assignment_id=f"{symbol}-{pattern}",
                pattern_name=pattern,
                timeframe=timeframe,
                state=state,
                detector=detector,
                config=config,
                scorer=scorer,
                max_entry_drift_atr=max_entry_drift_atr,
                regime_plugin=regime_plugin,
                regime_config=dict(regime_config) if regime_config else {},
                regime_filter=dict(regime_filter) if regime_filter else None,
                model_optional_plugins=list(model_optional_plugins) if model_optional_plugins else [],
            )
        )
    return assignments


# ---------------------------------------------------------------------------
# Trade simulation (causal forward walk)
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """One executed (or cap-rejected) representative trade."""

    event_id: str
    pattern_name: str
    symbol: str
    direction: str
    known_at: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float | None
    risk: float
    model_prob: float | None
    rule_score: float
    group_id: str | None
    n_patterns: int
    confluence_score: float
    risk_fraction: float
    asset_atr: float | None
    # forward outcome
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    gross_r: float = 0.0
    cost_r: float = 0.0
    net_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    executed: bool = True
    cap_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "pattern_name": self.pattern_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "known_at": str(self.known_at),
            "entry_time": str(self.entry_time),
            "entry_price": round(float(self.entry_price), 6),
            "stop_price": round(float(self.stop_price), 6),
            "target_price": (
                round(float(self.target_price), 6) if self.target_price is not None else None
            ),
            "risk": round(float(self.risk), 6),
            "model_prob": self.model_prob,
            "rule_score": float(self.rule_score),
            "group_id": self.group_id,
            "n_patterns": int(self.n_patterns),
            "confluence_score": round(float(self.confluence_score), 4),
            "risk_fraction": float(self.risk_fraction),
            "exit_time": str(self.exit_time) if self.exit_time is not None else None,
            "exit_price": (
                round(float(self.exit_price), 6) if self.exit_price is not None else None
            ),
            "exit_reason": self.exit_reason,
            "gross_r": round(float(self.gross_r), 4),
            "cost_r": round(float(self.cost_r), 4),
            "net_r": round(float(self.net_r), 4),
            "mfe_r": round(float(self.mfe_r), 4),
            "mae_r": round(float(self.mae_r), 4),
            "executed": bool(self.executed),
            "cap_reason": self.cap_reason,
        }


def _bar_pos(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    return _index_pos(df.index, ts, "left")


def simulate_trade(
    df: pd.DataFrame,
    ev: PatternEvent,
    costs: CostConfig,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
) -> BacktestTrade:
    """Simulate one event forward (bars after entry) with costs.

    Triple-barrier + fixed horizon convention (identical to the §6.3
    label_events rule): target hit -> +R_target; stop hit -> -1.0R; neither ->
    exit at close[entry + horizon].  Only bars AFTER the entry bar are read
    (strictly forward, no lookahead).
    """
    direction = str(ev.direction).lower()
    entry_price = float(ev.entry_price)
    stop_price = float(ev.stop_price)
    target_price = ev.target_price
    risk = abs(entry_price - stop_price)
    known_at = pd.Timestamp(ev.known_at_ts)
    entry_time = pd.Timestamp(ev.entry_time) if ev.entry_time is not None else known_at
    entry_bar = min(_bar_pos(df, entry_time), len(df) - 1)
    horizon = max(1, int(horizon_bars))
    end_bar = min(entry_bar + 1 + horizon, len(df) - 1)

    trade = BacktestTrade(
        event_id=ev.event_id,
        pattern_name=str(ev.pattern_name),
        symbol=str(ev.symbol),
        direction=direction,
        known_at=known_at,
        entry_time=df.index[entry_bar],
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk=risk,
        model_prob=(
            float(ev.model_prob) if ev.model_prob is not None else None
        ),
        rule_score=float(ev.rule_score or 0.0),
        group_id=ev.confluence_group_id,
        n_patterns=1,
        confluence_score=0.0,
        risk_fraction=1.0,
        asset_atr=_event_atr(ev),
    )
    if risk <= 0 or not np.isfinite(risk):
        trade.exit_reason = "invalid_risk"
        trade.executed = False
        trade.cap_reason = "risk<=0"
        return trade

    is_long = direction in ("bullish", "long", "buy")
    target_r = (
        (target_price - entry_price) / risk if target_price is not None and is_long
        else (entry_price - target_price) / risk if target_price is not None
        else float("inf")
    )
    if target_r < 0:
        target_r = float("inf")

    mfe_r = 0.0
    mae_r = 0.0
    exit_bar = end_bar
    exit_reason = "horizon"
    exit_price = float(df["close"].iloc[end_bar])
    for b in range(entry_bar + 1, end_bar + 1):
        high = float(df["high"].iloc[b])
        low = float(df["low"].iloc[b])
        mfe_r = max(mfe_r, (high - entry_price) / risk if is_long else (entry_price - low) / risk)
        mae_r = max(mae_r, (entry_price - low) / risk if is_long else (high - entry_price) / risk)
        if mfe_r >= target_r:
            exit_bar = b
            exit_reason = "target"
            exit_price = float(target_price) if target_price is not None else high
            break
        if mae_r >= 1.0:
            exit_bar = b
            exit_reason = "stop"
            exit_price = stop_price
            break
    if exit_reason == "target":
        gross_r = target_r
    elif exit_reason == "stop":
        gross_r = -1.0
    else:
        gross_r = (
            (exit_price - entry_price) / risk if is_long
            else (entry_price - exit_price) / risk
        )

    cost_r = costs.cost_r(entry_price, risk)
    trade.mfe_r = float(mfe_r)
    trade.mae_r = float(mae_r)
    trade.gross_r = float(gross_r)
    trade.cost_r = float(cost_r)
    trade.net_r = float(gross_r - cost_r)
    trade.exit_time = df.index[exit_bar]
    trade.exit_price = float(exit_price)
    trade.exit_reason = exit_reason
    return trade


def _event_atr(ev: PatternEvent) -> float | None:
    """Best-known ATR for an event (used by §4.4 caps), mirroring
    ``default_atr_resolver`` key order."""
    import math

    attrs = getattr(ev, "attributes", {}) or {}
    for key in ("atr_value", "atr"):
        v = attrs.get(key)
        if v is not None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return f
    levels = getattr(ev, "structure_levels", {}) or {}
    v = levels.get("atr")
    if v is not None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isfinite(f):
            return f
    return None


def _rejected_trade(
    rep: PatternEvent,
    group: Any,
    risk_fraction: float,
    reason: str,
) -> BacktestTrade:
    """Build a non-executed BacktestTrade for gate/hmm_unavailable rejections.

    Shared by the ``regime_blocked`` (hard gate §5) and ``hmm_unavailable``
    (fail-closed §6.3) paths so the rejected trade is recorded with its
    discard reason (Event Lake §7.1 keeps the event anyway).
    """
    return BacktestTrade(
        event_id=rep.event_id,
        pattern_name=str(rep.pattern_name),
        symbol=str(rep.symbol),
        direction=str(rep.direction),
        known_at=pd.Timestamp(rep.known_at_ts),
        entry_time=(
            pd.Timestamp(rep.entry_time)
            if rep.entry_time is not None
            else pd.Timestamp(rep.known_at_ts)
        ),
        entry_price=(
            float(rep.entry_price)
            if rep.entry_price and rep.entry_price == rep.entry_price
            else 0.0
        ),
        stop_price=(
            float(rep.stop_price)
            if rep.stop_price and rep.stop_price == rep.stop_price
            else 0.0
        ),
        target_price=rep.target_price,
        risk=0.0,
        model_prob=float(rep.model_prob) if rep.model_prob is not None else None,
        rule_score=float(rep.rule_score or 0.0),
        group_id=rep.confluence_group_id,
        n_patterns=group.n_patterns if group is not None else 1,
        confluence_score=group.confluence_score if group is not None else 0.0,
        risk_fraction=risk_fraction,
        asset_atr=_event_atr(rep),
        executed=False,
        cap_reason=reason,
    )


# ---------------------------------------------------------------------------
# Portfolio replay — grouping (live code path) + §4.4 caps
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """One symbol's backtest: events, groups, trades, equity, decisions."""

    symbol: str
    timeframe: str
    events: list[PatternEvent]
    groups: list[ResolvedGroup]
    trades: list[BacktestTrade]
    rejected: list[BacktestTrade]
    corr_config: CorrelationConfig
    costs: CostConfig
    mode: str

    @property
    def executed_trades(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.executed]


# ---------------------------------------------------------------------------
# Grouping — the live §4.2 code path (no duplicated logic)
# ---------------------------------------------------------------------------

def group_events(
    events: list[PatternEvent],
    corr_cfg: CorrelationConfig | None = None,
) -> list[ResolvedGroup]:
    """Partition events into correlation groups with the SAME
    ``CorrelationManager`` the live ``MultiPatternEngine`` uses (§4.2/§4.3).

    This is the single grouping call of the runner — a parity test asserts
    it reaches the same decisions as the live engine on identical input.
    """
    corr_cfg = corr_cfg or CorrelationConfig(mode="dedup", bar_seconds=900.0)
    manager = CorrelationManager(corr_cfg)
    return list(manager.group(events, config=corr_cfg))


def run_symbol_backtest(
    df: pd.DataFrame,
    assignments: list[Any],
    corr_cfg: CorrelationConfig | None = None,
    costs: CostConfig | None = None,
    symbol_override: str | None = None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    # --- HMM regime plugin (requirements v1.0; default OFF) ---
    shared_plugin: Any = None,
    config_source: dict[str, Any] | None = None,
) -> PortfolioResult:
    """Full §12 replay for one symbol on an already-warmed OHLCV frame.

    1. detect all assignments (causal, once);
    2. ``CorrelationManager.group`` over ALL events — the exact live
       grouping class/config (window-gated union-find, §4.2);
    3. replay groups in ``known_at`` order; per group:
       apply the HMM regime hard gate (when a rule applies) then §4.4
       exposure caps against currently-open trades, then simulate the
       representative trade forward (only bars > entry) with costs.
    """
    corr_cfg = corr_cfg or CorrelationConfig(mode="dedup", bar_seconds=900.0)
    manager = CorrelationManager(corr_cfg)

    events = detect_all(df, assignments, shared_plugin=shared_plugin,
                        config_source=config_source)
    events = [e for e in events if e is not None]
    events.sort(key=lambda e: (pd.Timestamp(e.known_at_ts), e.event_id))

    symbol = symbol_override or (
        events[0].symbol if events else str(assignments[0].config.get("symbol", "XAUUSD"))
    )
    timeframe = str(assignments[0].timeframe or "M15")
    costs = costs or CostConfig.for_symbol(symbol)

    # §4.2 — same grouping call the live engine performs each scan.
    groups = manager.group(events, config=corr_cfg)

    # owner lookup (event -> assignment) for risk_fraction/state
    owner_by_pattern: dict[str, Any] = {}
    for a in assignments:
        pname = str(getattr(a.detector, "name", None) or a.pattern_name or "").lower()
        owner_by_pattern[pname] = a

    trades: list[BacktestTrade] = []
    rejected: list[BacktestTrade] = []
    open_positions: list[dict[str, Any]] = []

    for g in sorted(
        (g for g in groups if g.representative is not None),
        key=lambda g: (pd.Timestamp(g.representative.known_at_ts), g.representative.event_id),
    ):
        rep = g.representative
        if rep is None:
            continue
        owner = owner_by_pattern.get(str(rep.pattern_name).lower())
        risk_fraction = float(getattr(owner, "risk_fraction", 1.0)) if owner is not None else 1.0

        # §6.3 fail-closed (mirror of live engine): an event stamped
        # hmm_unavailable (required hmm model without a working plugin) must
        # not be simulated — record as rejected so the Event Lake / report
        # still carry the discard_reason (spec §7.1).
        if rep.attributes.get("discard_reason") == "hmm_unavailable":
            rejected_unavailable = _rejected_trade(
                rep, g, risk_fraction, "hmm_unavailable",
            )
            rejected.append(rejected_unavailable)
            continue

        # HMM regime hard gate (requirements §5, shared is_allowed with live) —
        # a disallowed regime blocks the trade before simulation; the event is
        # annotated discard_reason="regime_blocked" (Event Lake §7.1).
        slot = getattr(owner, "_regime_slot", None) if owner is not None else None
        if not gate_not_blocked(rep, slot, df):
            rejected_blocked = _rejected_trade(
                rep, g, risk_fraction, "regime_blocked",
            )
            rejected.append(rejected_blocked)
            continue

        trade = simulate_trade(df, rep, costs, horizon_bars=horizon_bars)
        trade.n_patterns = g.n_patterns
        trade.confluence_score = g.confluence_score
        trade.risk_fraction = risk_fraction

        # drop positions that closed at/before this trade's entry bar
        open_positions = [
            p
            for p in open_positions
            if p.get("exit_time") is None
            or pd.Timestamp(p["exit_time"]) > trade.entry_time
        ]

        # §4.4 exposure caps (mandatory, same code as live RiskGuard path)
        ok, reason = manager.check_exposure_caps(
            symbol=symbol,
            direction=rep.direction,
            proposed_risk=risk_fraction,
            proposed_entry=rep.entry_price,
            proposed_atr=trade.asset_atr,
            open_positions=open_positions,
            config=corr_cfg,
        )
        if not ok:
            trade.executed = False
            trade.cap_reason = reason
            rejected.append(trade)
            # cap-rejected trades do not open positions
            continue
        trade.executed = True
        trades.append(trade)
        open_positions.append(
            {
                "direction": rep.direction,
                "entry_price": rep.entry_price,
                "risk": risk_fraction,
                "position_size": risk_fraction,
                "exit_time": trade.exit_time,
            }
        )

    return PortfolioResult(
        symbol=symbol,
        timeframe=timeframe,
        events=events,
        groups=groups,
        trades=trades,
        rejected=rejected,
        corr_config=corr_cfg,
        costs=costs,
        mode=corr_cfg.mode,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_costs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid --costs JSON: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def main(argv: list[str] | None = None) -> int:
    """CLI entry — runs N patterns over the configured symbols/period and
    writes the §12 5-item report for each symbol (report.py)."""
    from research.multi_backtest.report import generate_symbol_report

    parser = argparse.ArgumentParser(
        description="Multi-pattern backtest runner (§12) — live code path."
    )
    parser.add_argument("--symbols", default="XAUUSD,EURUSD",
                        help="comma-separated symbols (default XAUUSD,EURUSD)")
    parser.add_argument("--patterns", default=",".join(PATTERNS_XAUUSD),
                        help="comma-separated pattern names (default LSW,DB,DT)")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--start", default="2023-01-01", help="report window start")
    parser.add_argument("--end", default="2026-09-03", help="report window end")
    parser.add_argument("--mode", default="dedup",
                        choices=("dedup", "confluence", "independent"))
    parser.add_argument("--corr-time-window", type=float, default=5.0)
    parser.add_argument("--corr-price-atr", type=float, default=0.5)
    parser.add_argument("--bar-seconds", type=float, default=900.0,
                        help="seconds per bar of the larger TF (M15=900)")
    parser.add_argument("--cap-total-risk", type=float, default=2.0,
                        help="§4.4 max_total_risk_per_symbol (units; live default "
                             "1.0 fits max_open_positions=1, backtest default 2.0 "
                             "allows 2 concurrent units)")
    parser.add_argument("--cap-cluster-risk", type=float, default=1.0,
                        help="§4.4 max_direction_cluster_risk (units; must be >= "
                             "per-trade risk_fraction=1.0 so a lone trade passes, "
                             "live default 0.75 assumes risk<=0.75 units)")
    parser.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON_BARS)
    parser.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS)
    parser.add_argument("--costs", default=None,
                        help='JSON e.g. \'{"XAUUSD":{"spread_bps":1.0}}\'')
    parser.add_argument("--attach-models", action="store_true", default=True,
                        help="attach t1 tier-2 model_prob when artifacts exist")
    parser.add_argument("--no-models", action="store_false", dest="attach_models")
    parser.add_argument("--out", default=str(REPORTS_DIR),
                        help="report output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    patterns = [p.strip().lower() for p in args.patterns.split(",") if p.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_overrides = _parse_costs(args.costs)

    corr_cfg = CorrelationConfig(
        mode=args.mode,
        corr_time_window=args.corr_time_window,
        corr_price_window_atr=args.corr_price_atr,
        bar_seconds=args.bar_seconds,
        max_total_risk_per_symbol=args.cap_total_risk,
        max_direction_cluster_risk=args.cap_cluster_risk,
    )

    written: list[Path] = []
    for symbol in symbols:
        logger.info("Loading %s M15 …", symbol)
        df = load_symbol_frame(
            symbol, start=args.start, end=args.end, warmup_bars=args.warmup_bars,
        )
        logger.info("Building assignments for %s (%d bars) …", symbol, len(df))
        assignments = build_assignments(
            symbol, patterns, timeframe=args.timeframe,
            df=df if args.attach_models else None,
            attach_models=args.attach_models,
        )
        if not assignments:
            logger.error("No assignments built for %s — check pattern registry", symbol)
            return 2
        costs = CostConfig.for_symbol(symbol, cost_overrides)
        result = run_symbol_backtest(
            df, assignments, corr_cfg=corr_cfg, costs=costs,
            symbol_override=symbol, horizon_bars=args.horizon_bars,
        )
        md_path, json_path = generate_symbol_report(
            result, df, out_dir=out_dir,
            start=pd.Timestamp(args.start), end=pd.Timestamp(args.end),
        )
        written.extend([md_path, json_path])
        logger.info("%s: %d events, %d trades (written %s)",
                    symbol, len(result.events), len(result.executed_trades), md_path)

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())