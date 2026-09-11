"""
signal_engine_v2.py — Multi-Symbol Signal Engine for Paper Trading V2

Key changes from signal_engine.py (v1):
1. Reads per-symbol config from research/configs/symbols/{SYMBOL}.yaml.
2. Validates status field: only "validated" gets a running engine.
3. Loads per-symbol model artifacts from artifacts/models/{SYMBOL}/.
4. Loads per-symbol feature schema from artifacts/feature_schemas/{SYMBOL}/.
5. Maintains separate levels_cache per symbol.
6. Entry=next_open_after_confirmation. Stop=sweep_extreme+buffer_atr*ATR.
   Target=entry+reward_r*(entry-stop).
7. No-lookahead preserved (shift(1) in reused modules).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

# Ensure the research pipeline src/ directory is importable.
# All required modules (sweep_detector_v2, confirmation, deduplication,
# features, liquidity, scoring) live under research/patterns/liquidity_sweep/src/.
_RESEARCH = Path(__file__).resolve().parent.parent.parent / "research" / "patterns" / "liquidity_sweep"
if str(_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_RESEARCH))

from src.events.confirmation import (  # noqa: E402  (bootstrap import)
    attach_confirmations,
    run_anchor_position,
    run_opposite_extreme_at,
)
from src.events.deduplication import (  # noqa: E402  (bootstrap import)
    select_deduplicated_events,
)
from src.events.sweep_detector_v2 import (  # noqa: E402  (bootstrap import)
    _candidate_rows_v2,
    detect_sweeps_v2,
)
from src.features.feature_pipeline import (  # noqa: E402  (bootstrap import)
    build_event_features,
)
from src.features.registry import (  # noqa: E402  (bootstrap import)
    registered_feature_names,
)

# The research feature registry no longer exports a precomputed
# SIGNAL_FEATURE_NAMES constant; resolve the ordered feature-name list once
# at import time (identical contract for model inference below).
SIGNAL_FEATURE_NAMES = registered_feature_names()
from src.liquidity.level_registry import (  # noqa: E402  (bootstrap import)
    build_liquidity_levels,
)
from src.scoring.rule_score import compute_rule_scores  # noqa: E402  (bootstrap import)

logger = logging.getLogger("signal_engine_v2")

# --- rule_score scales per pattern (B2, bug summary) ------------------------
# Plugins do not share one scale: liquidity_sweep emits 0..100, every other
# plugin clamps to [0, 1].  Only LSW is declared here; anything else is
# auto-detected by magnitude in _normalize_rule_score.
_RULE_SCORE_SCALE: dict[str, float] = {"liquidity_sweep": 100.0}

# Optional per-pattern minimum model probability (B1).  Empty by default =
# no probability gate (historical behaviour).  Opt in either here or via a
# per-assignment ``min_model_prob`` key in the pattern/symbol config.
_DEFAULT_MIN_MODEL_PROB: dict[str, float] = {}

# Centralized system logger (GUI_REWORK §5)
from live.logging.logger_v2 import log as syslog  # noqa: E402  (bootstrap import)

# Model Registry access (GUI_REWORK §2)
from live.state.shared_app_state_v2 import (  # noqa: E402  (bootstrap import)
    SYMBOL_CONFIG_DIR,
    ModelRegistry,
    get_state,
)

# Paths — relative to trading_v3/ root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # trading_v3/
# Single-source the per-symbol config location with the Risk Guard and the
# Execution Layer (canonical constant in shared_app_state_v2) so order-sizing
# params never diverge between modules.
_SYMBOL_CONFIG_DIR = SYMBOL_CONFIG_DIR
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
_MODELS_BASE = _ARTIFACTS_DIR / "eurusd" / "models"
_FEATURE_SCHEMAS_BASE = _RESEARCH / "artifacts" / "feature_schemas"
_PIPELINE_V2_MODEL_DIR = _RESEARCH / "pipeline_v2" / "artifacts" / "models"

# --- v1.1 multi-pattern integration (§2.2, §4, §5, §9) ---------------------
from live.engine.correlation_manager import (  # noqa: E402
    CorrelationConfig,
    CorrelationManager,
    ResolvedGroup,
)
from live.engine.pattern_registry import (  # noqa: E402
    get_registry,
    order_comment_for_event,
)
from research.core.contracts import (  # noqa: E402
    LIFECYCLE_LIVE,
    LIFECYCLE_SHADOW,
    PatternEvent,
)
from research.core.dedupe import drop_opposite_overlap  # noqa: E402


class SymbolNotValidatedError(ValueError):
    """Symbol status is not 'validated'."""

class SymbolConfigNotFoundError(FileNotFoundError):
    """Per-symbol YAML config does not exist."""

class ModelArtifactNotFoundError(FileNotFoundError):
    """Per-symbol model/calibrator artifact missing."""


@dataclass(frozen=True)
class SignalCandidate:
    """One executable signal produced by the signal engine."""
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime
    event_time: datetime
    event_id: str
    rule_score: float
    model_prob: float
    combined_score: float
    penetration_atr: float
    wick_ratio: float
    reclaim_atr: float
    h1_trend: int
    confirmation_delay_bars: float
    confirmation_range_atr: float
    atr_value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- v1.1 multi-pattern lineage (§9.2/§4) — optional, back-compat defaults -
    pattern_name: str = ""
    pattern_version: str = ""
    pattern_short: str = ""
    order_comment: str = ""
    confluence_score: float = 0.0
    confluence_group_id: str | None = None
    assignment_id: str = ""
    risk_fraction: float = 1.0
    lifecycle_state: str = "live"


# ---------------------------------------------------------------------------
# Per-symbol config loader
# ---------------------------------------------------------------------------

def load_symbol_config(symbol: str) -> dict:
    """Load per-symbol YAML from research/configs/symbols/{symbol}.yaml."""
    cfg_path = _SYMBOL_CONFIG_DIR / f"{symbol}.yaml"
    if not cfg_path.exists():
        raise SymbolConfigNotFoundError(
            f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty or invalid config at {cfg_path}")
    return cfg


# ---------------------------------------------------------------------------
# Status validation (spec section 1 - critical safety gate)
# ---------------------------------------------------------------------------

def _validate_status(symbol_cfg: dict, symbol_name: str = "") -> None:
    """Check status field; raise if not 'validated'."""
    status = symbol_cfg.get("status")
    if status is None:
        status = symbol_cfg.get("symbol", {}).get("status")
    if status is None:
        raise SymbolNotValidatedError(
            f"Symbol '{symbol_name}' config has no 'status' field. "
            f"Spec requires status: validated|candidate|rejected.")
    status_str = str(status).lower().strip()
    if status_str != "validated":
        raise SymbolNotValidatedError(
            f"Symbol '{symbol_name}' status is '{status_str}', not 'validated'. "
            f"Signal Engine will NOT run. Only validated symbols go live.")


# ---------------------------------------------------------------------------
# Per-symbol model / calibrator / feature schema loader
# ---------------------------------------------------------------------------

def _load_symbol_model_and_calibrator(
    symbol: str,
    model_path_override: str | None = None,
    calibrator_path_override: str | None = None,
) -> tuple:
    """Load model + calibrator, preferring explicit paths from ModelRegistry.

    When *model_path_override* is given (from ModelInfo._resolved_paths),
    load from those exact paths.  Otherwise fall back to legacy per-symbol
    artifacts/models/{symbol}/ directory.
    """
    if model_path_override is not None:
        model_path = Path(model_path_override)
        calibrator_path = Path(calibrator_path_override) if calibrator_path_override else _MODELS_BASE / symbol / "calibrator.pkl"
    else:
        model_dir = _MODELS_BASE / symbol
        model_path = model_dir / "model.pkl"
        calibrator_path = model_dir / "calibrator.pkl"

    if not model_path.exists():
        if model_path_override is not None:
            raise ModelArtifactNotFoundError(
                f"Model artifact not found at ModelRegistry path: {model_path}")
        fallback = _PIPELINE_V2_MODEL_DIR / "model.pkl"
        if fallback.exists():
            logger.warning("Per-symbol model for %s not found - using fallback %s",
                           symbol, fallback)
            model_path = fallback
            calibrator_path = _PIPELINE_V2_MODEL_DIR / "calibrator.pkl"
        else:
            raise ModelArtifactNotFoundError(
                f"Model artifact not found for '{symbol}'. Train via Onboarding Wizard.")

    import joblib
    logger.info("Loading model for %s from %s", symbol, model_path)
    raw = joblib.load(str(model_path))
    model = raw["model"] if isinstance(raw, dict) else raw

    logger.info("Loading calibrator for %s from %s", symbol, calibrator_path)
    calibrator = joblib.load(str(calibrator_path))

    base = getattr(model, "estimator", None)
    if base is None:
        raise ValueError(f"model.pkl for {symbol} has no .estimator")
    n_feat = getattr(base, "n_features_in_", 0)
    logger.info("Model for %s expects %d features", symbol, n_feat)
    return model, calibrator


def _load_symbol_feature_schema(symbol: str,
                                 schema_path_override: str | None = None) -> dict | None:
    """Load feature schema from explicit path (ModelRegistry) or legacy location.

    When *schema_path_override* is given, load from that exact path.
    Otherwise try artifacts/feature_schemas/{symbol}/features.json.
    """
    if schema_path_override is not None:
        schema_path = Path(schema_path_override)
    else:
        schema_path = _FEATURE_SCHEMAS_BASE / symbol / "features.json"
    if not schema_path.exists():
        return None
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Extract per-symbol pipeline parameters
# ---------------------------------------------------------------------------

def _extract_pipeline_params(symbol_cfg: dict) -> dict:
    """Merge pipeline_overrides with frozen defaults into a flat param dict."""
    ov = symbol_cfg.get("pipeline_overrides", {})
    s = ov.get("sweep", {})
    c = ov.get("confirmation", {})
    i = ov.get("indicators", {})
    liq = ov.get("liquidity", {})
    st = ov.get("stop", {})

    return {
        "v2_nguoc_trend": s.get("v2_nguoc_trend", True),
        "v2_target_r": s.get("v2_target_r", 3.0),
        "min_penetration_atr": s.get("min_penetration_atr", 0.05),
        "max_penetration_atr": s.get("max_penetration_atr", 0.20),
        "min_wick_ratio": s.get("min_wick_ratio", 0.35),
        "min_reclaim_atr": s.get("min_reclaim_atr", 0.00),
        "cooldown_bars": s.get("cooldown_bars", 4),
        "group_rule": s.get("group_rule", "first"),
        "max_wait_bars": c.get("max_wait_bars", 3),
        "require_break_sweep_extreme": c.get("require_break_sweep_extreme", True),
        "min_body_ratio": c.get("min_body_ratio", 0.60),
        "min_range_atr": c.get("min_range_atr", 0.80),
        "atr_period": i.get("atr_period", 14),
        "rolling_lookback": liq.get("rolling_lookback", 20),
        "buffer_atr": st.get("buffer_atr", 0.10),
    }


# ---------------------------------------------------------------------------
# Model probability helper
# ---------------------------------------------------------------------------

def _compute_model_prob(
    features_df: pd.DataFrame, model: Any, calibrator: Any,
) -> pd.Series:
    """Compute calibrated probability of the positive (win) class."""
    if features_df.empty:
        return pd.Series(dtype="float64")
    available = [c for c in SIGNAL_FEATURE_NAMES if c in features_df.columns]
    # Only registered *numeric* features are part of the model input contract:
    # the research training step drops categorical registry columns (e.g.
    # level_type, volatility_regime) before fitting (src.modeling.train).
    # Mirror that selection here so live inference sees exactly the columns
    # the model expects and never tries to cast string values to float.
    available = [c for c in available if pd.api.types.is_numeric_dtype(features_df[c])]
    X = features_df[available].astype("float64").fillna(0.0).values
    try:
        proba = model.predict_proba(X)
    except Exception:
        proba = calibrator.predict_proba(X)
    if proba.shape[1] >= 2:
        return pd.Series(proba[:, 1], index=features_df.index)
    return pd.Series(proba[:, 0], index=features_df.index)


# ---------------------------------------------------------------------------
# Signal statistics (live per-symbol Trigger / Found / Pass counters)
# ---------------------------------------------------------------------------

def _record_signal_stat(symbol: str, stat: str, count: int = 1) -> None:
    """Thread-safely bump one per-symbol scan statistic and log it at INFO.

    *stat* is one of ``"trigger"`` / ``"found"`` / ``"pass"`` and maps to the
    matching ``SharedAppState.increment_*`` method.  The counters only tally
    decisions already taken on **closed** bars inside the per-bar scan —
    they never read future data and never influence detection/scoring, so
    no-lookahead and safety behavior is preserved.

    A failure here must never break the live scan flow, hence the guard.
    """
    try:
        state = get_state()
        getattr(state, f"increment_{stat}")(symbol, count)
        totals = state.get_signal_stats(symbol)
        syslog(
            "INFO", "SignalEngine", symbol,
            f"Signal stats — {stat} +{count} "
            f"(trigger={totals['trigger']}, found={totals['found']}, "
            f"pass={totals['pass']})",
            extra={"stat": stat, "delta": count, "totals": totals},
        )
    except Exception as exc:  # stats instrumentation must never break trading
        logger.warning("Signal stats update failed for %s (%s): %s",
                       symbol, stat, exc)


# ---------------------------------------------------------------------------
# Core per-bar scanning (one symbol)
# ---------------------------------------------------------------------------

def _check_new_bar(
    symbol: str,
    mcp_client: Any,
    pipeline_params: dict,
    model: Any,
    calibrator: Any,
    levels_cache: dict | None = None,
) -> list[SignalCandidate]:
    """Core per-bar scanning for one symbol (no-lookahead enforced)."""
    if levels_cache is None:
        levels_cache = {}

    # 1. Fetch candles
    candles_raw = mcp_client.get_chart_history(symbol)
    if candles_raw is None or (isinstance(candles_raw, pd.DataFrame) and candles_raw.empty):
        logger.warning("No candles returned for %s", symbol)
        return []
    candles = candles_raw.rename(columns=str.lower).copy()
    required = {"open", "high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Missing columns for {symbol}: {required - set(candles.columns)}")
    if not isinstance(candles.index, pd.DatetimeIndex):
        candles.index = pd.to_datetime(candles.index)
    if not candles.index.is_monotonic_increasing:
        candles = candles.sort_index()
    if "volume" not in candles.columns:
        candles["volume"] = 0

    # --- Record the newest closed candle this scan actually processed ---
    # Polling-layer gating metadata: the signal polling engine reads this
    # (via the engine closure) so a symbol's engine is only invoked for a
    # genuinely new closed M15 candle and Trigger counts distinct bars.
    # The reserved key never collides with per-symbol liquidity levels.
    levels_cache["__last_processed_bar_time__"] = candles.index[-1]

    # --- Signal statistics: Trigger (one M15 candle check per scan run) ---
    # Candles are available and about to be scanned — count this M15 candle
    # check/trigger for the symbol.  Only closed-bar data is used here.
    _record_signal_stat(symbol, "trigger")

    # Log: new bar / candle data received
    last_close = float(candles["close"].iloc[-1]) if not candles.empty else 0.0
    last_time = str(candles.index[-1]) if not candles.empty else ""
    syslog("INFO", "SignalEngine", symbol,
           f"New M15 candle closed @ {last_close:.2f} → running detection",
           extra={"last_close": last_close, "candle_time": last_time, "n_candles": len(candles)})

    # 2. V2 Sweep detection (per-symbol params)
    sweep_out = detect_sweeps_v2(
        candles,
        atr_period=pipeline_params["atr_period"],
        level_lookback=pipeline_params["rolling_lookback"],
        min_penetration_atr=pipeline_params["min_penetration_atr"],
        max_penetration_atr=pipeline_params["max_penetration_atr"],
        min_wick_ratio=pipeline_params["min_wick_ratio"],
        v2_nguoc_trend=pipeline_params["v2_nguoc_trend"],
    )
    candidates = _candidate_rows_v2(sweep_out)
    if candidates.empty:
        syslog("DEBUG", "SignalEngine", symbol, "No sweep found on new candle")
        return []

    n_sweeps = len(candidates)

    # --- Signal statistics: Found (patterns the rule machine detected) ---
    # detect_sweeps_v2 / _candidate_rows_v2 produced n_sweeps candidate
    # rows — the rule machine filter output for this scan.
    _record_signal_stat(symbol, "found", n_sweeps)

    syslog("INFO", "SignalEngine", symbol,
           f"Sweep detected ({'bullish' if n_sweeps > 0 else 'bearish'}, {n_sweeps} candidate(s)) → scoring",
           extra={"n_candidates": n_sweeps, "pipeline_params": str(pipeline_params)})

    # 3. Deduplicate
    deduped = select_deduplicated_events(
        candidates,
        cooldown_bars=pipeline_params["cooldown_bars"],
        group_rule=pipeline_params["group_rule"],
    )
    if deduped.empty:
        return []
    deduped = deduped.sort_values(["event_time", "direction"], kind="stable")
    deduped["event_id"] = [f"{symbol}-V2-{i:06d}" for i in range(len(deduped))]
    deduped["v2_target_r"] = pipeline_params["v2_target_r"]

    # 4. Apply confirmation
    confirmed = attach_confirmations(
        candles, deduped, {},
        max_wait_bars=pipeline_params["max_wait_bars"],
        min_body_ratio=pipeline_params["min_body_ratio"],
        min_range_atr=pipeline_params["min_range_atr"],
        require_break_sweep_extreme=pipeline_params["require_break_sweep_extreme"],
    )
    confirmed_events = confirmed[confirmed["is_confirmed"]].copy()
    if confirmed_events.empty:
        return []

    # 5. Build feature matrix (per-symbol levels cache)
    if symbol not in levels_cache:
        levels_cache[symbol] = build_liquidity_levels(candles, {})
    levels = levels_cache[symbol]
    features_df = build_event_features(candles, levels, confirmed_events, {})
    if features_df.empty:
        return []

    # 6. Compute scores
    scored = compute_rule_scores(confirmed_events, {}, levels)
    rule_scores = dict(zip(scored["event_id"], scored["rule_score"]))
    model_probs = _compute_model_prob(features_df, model, calibrator)
    model_prob_dict = dict(zip(features_df["event_id"], model_probs))

    # 7. Compute entry / stop / target
    buffer_atr = pipeline_params["buffer_atr"]
    reward_r = pipeline_params["v2_target_r"]
    atr_series = sweep_out["atr"]

    results: list[SignalCandidate] = []
    for _, ev in confirmed_events.iterrows():
        eid = ev["event_id"]
        direction = "long" if str(ev["direction"]).lower() in ("bullish", "long") else "short"
        is_long = direction == "long"
        bar_pos = candles.index.get_indexer([ev["event_time"]])[0]
        anchor = run_anchor_position(sweep_out, bar_pos, str(ev["direction"]))
        conf_time = ev["confirmation_time"]
        conf_bar = candles.index.get_indexer([conf_time])[0]
        entry_bar = conf_bar + 1
        if entry_bar >= len(candles):
            continue
        entry_price = float(candles["open"].iloc[entry_bar])
        atr_sweep = float(atr_series.iloc[anchor])
        extreme = run_opposite_extreme_at(candles, anchor, anchor, str(ev["direction"]))
        if is_long:
            stop_price = extreme - buffer_atr * atr_sweep
            target_price = entry_price + reward_r * (entry_price - stop_price)
        else:
            stop_price = extreme + buffer_atr * atr_sweep
            target_price = entry_price - reward_r * (stop_price - entry_price)

        rule_score = rule_scores.get(eid, 0.0)
        model_prob = model_prob_dict.get(eid, 0.5)
        # B2: this legacy path is sweep-only (0..100 scale) — normalize
        # explicitly instead of hard-coding the divisor.
        combined_score = _combine_score(
            _normalize_rule_score("liquidity_sweep", rule_score), model_prob)

        results.append(SignalCandidate(
            symbol=symbol, direction=direction,
            entry_price=round(entry_price, 5),
            stop_price=round(stop_price, 5),
            target_price=round(target_price, 5),
            entry_time=candles.index[entry_bar].to_pydatetime(),
            event_time=ev["event_time"].to_pydatetime(),
            event_id=eid,
            rule_score=round(rule_score, 2),
            model_prob=round(model_prob, 4),
            combined_score=round(combined_score, 4),
            penetration_atr=round(float(ev["penetration_atr"]), 4),
            wick_ratio=round(float(ev["wick_ratio"]), 4),
            reclaim_atr=round(float(ev["reclaim_atr"]), 4),
            h1_trend=int(ev.get("h1_trend", 0)),
            confirmation_delay_bars=float(ev.get("confirmation_delay_bars", np.nan)),
            confirmation_range_atr=float(ev.get("confirmation_range_atr", np.nan)),
            atr_value=round(atr_sweep, 5),
            metadata={
                "config_source": "per_symbol",
                "level_id": str(ev.get("level_id", "")),
                "level_price": float(ev.get("level_price", 0)),
            },
        ))

    # Log: signal generation summary
    if results:
        # --- Signal statistics: Pass (signals emitted this scan) ---
        # Every SignalCandidate above survived the rule-score + ML model
        # probability round and was appended to *results*.
        _record_signal_stat(symbol, "pass", len(results))

        dir_label = results[0].direction
        syslog("INFO", "SignalEngine", symbol,
               f"Signal generated: {len(results)} candidate(s), direction={dir_label}, "
               f"best score={max(r.combined_score for r in results):.4f}",
               extra={"n_signals": len(results),
                      "directions": list(set(r.direction for r in results)),
                      "scores": [r.combined_score for r in results]})
    return results


# ---------------------------------------------------------------------------
# Public factory: create_symbol_engine
# ---------------------------------------------------------------------------

def create_symbol_engine(
    symbol: str,
    mcp_client: Any,
    symbol_cfg_override: dict | None = None,
    model_override: Any | None = None,
    calibrator_override: Any | None = None,
) -> Callable[[], list[SignalCandidate]]:
    """Create a per-symbol signal engine closure.

    Flow:
    1. Load per-symbol config from research/configs/symbols/{symbol}.yaml.
    2. Validate status field (only 'validated' proceeds).
    3. Read ``model_id`` from config → look up ModelRegistry → load
       model.pkl / calibrator.pkl / feature_schema from resolved paths.
       If model_id is missing or not found → log ERROR and raise.
    4. Extract pipeline parameters.
    5. Return a zero-arg callable ``check_new_bar() -> list[SignalCandidate]``.

    Parameters
    ----------
    symbol : str
        Instrument name (e.g. "XAUUSD", "EURUSD").
    mcp_client : Any
        MCP client with get_chart_history(symbol) returning OHLCV DataFrame.
    symbol_cfg_override : dict, optional
        Pre-loaded symbol config (for testing; skips file read).
    model_override : Any, optional
        Pre-loaded model (for testing; skips load).
    calibrator_override : Any, optional
        Pre-loaded calibrator (for testing; skips load).

    Returns
    -------
    callable
        A zero-argument function ``check_new_bar() -> list[SignalCandidate]``.

    Raises
    ------
    SymbolConfigNotFoundError
        If the per-symbol YAML is missing.
    SymbolNotValidatedError
        If the symbol status is not 'validated'.
    ModelArtifactNotFoundError
        If model artifacts are missing (no fallback).

    Usage
    -----
    >>> eng = create_symbol_engine("XAUUSD", mcp_client)
    >>> signals = eng()
    """
    # --- 1. Load per-symbol config ---
    if symbol_cfg_override is not None:
        symbol_cfg = symbol_cfg_override
    else:
        symbol_cfg = load_symbol_config(symbol)

    # --- 2. Validate status ---
    _validate_status(symbol_cfg, symbol)

    logger.info("Symbol '%s' status=validated — creating engine", symbol)
    syslog("INFO", "SignalEngine", symbol, "Creating engine — status=validated")

    # --- 3. Read model_id and load via Model Registry ---
    model_id = symbol_cfg.get("model_id") or symbol_cfg.get("symbol", {}).get("model_id")
    if model_override is not None and calibrator_override is not None:
        model, calibrator = model_override, calibrator_override
        syslog("INFO", "SignalEngine", symbol, "Using model override (test mode)")
        feature_schema = None
    elif model_id:
        registry = ModelRegistry.get_instance()
        model_info = registry.get_model(model_id)
        if model_info is None:
            msg = f"model_id='{model_id}' not found in Model Registry — skipping symbol"
            logger.error(msg)
            syslog("ERROR", "SignalEngine", symbol, msg)
            raise SymbolNotValidatedError(msg)

        resolved = model_info._resolved_paths
        model_path_str = resolved.get("model_path", "")
        calibrator_path_str = resolved.get("calibrator_path", "")
        schema_path_str = resolved.get("feature_schema", "")

        if not model_path_str or not Path(model_path_str).is_file():
            msg = f"Model artifact missing for model_id='{model_id}' at {model_path_str} — skipping symbol"
            logger.error(msg)
            syslog("ERROR", "SignalEngine", symbol, msg, extra={"model_id": model_id})
            raise ModelArtifactNotFoundError(msg)

        model, calibrator = _load_symbol_model_and_calibrator(
            symbol,
            model_path_override=model_path_str,
            calibrator_path_override=calibrator_path_str,
        )
        feature_schema = _load_symbol_feature_schema(
            symbol,
            schema_path_override=schema_path_str if schema_path_str else None,
        )

        model_ident = getattr(model, "__class__.__name__", str(type(model).__name__))
        syslog("INFO", "SignalEngine", symbol,
               f"Model loaded from ModelRegistry — {model_id} ({model_ident})",
               extra={"model_id": model_id, "model_type": model_ident})
    else:
        msg = "No model_id in config and no model override — skipping symbol"
        logger.error(msg)
        syslog("ERROR", "SignalEngine", symbol, msg)
        raise SymbolNotValidatedError(msg)

    # --- 4. Extract pipeline parameters ---
    pipeline_params = _extract_pipeline_params(symbol_cfg)

    # --- 5. Load feature schema (optional) — skip if already loaded via ModelRegistry ---
    if feature_schema is None:  # not yet loaded (e.g. test override path)
        feature_schema = _load_symbol_feature_schema(symbol)
    if feature_schema is not None:
        logger.info("Loaded per-symbol feature schema for %s (%d fields)",
                    symbol, len(feature_schema))
        syslog("INFO", "SignalEngine", symbol,
               f"Feature schema loaded ({len(feature_schema)} fields)")

    # Per-symbol levels cache (separate from every other symbol)
    levels_cache: dict = {}

    # --- 6. Return bound closure (with M15 new-bar gating metadata) ---
    def check_new_bar() -> list[SignalCandidate]:
        return _check_new_bar(
            symbol, mcp_client, pipeline_params, model, calibrator, levels_cache,
        )

    def peek_latest_bar_time():
        """Return the timestamp of the newest closed M15 candle available for
        this symbol right now (``None`` when candles are unavailable).

        Fetches chart history exactly like ``_check_new_bar`` but performs NO
        detection and records NO statistics.  The polling engine calls this
        once per symbol per cycle to skip unchanged frames between M15 closes,
        so Trigger counts distinct closed candles and the same event is never
        re-detected/re-emitted on an unchanged frame.
        """
        try:
            raw = mcp_client.get_chart_history(symbol)
            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                return None
            df = raw.rename(columns=str.lower).copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            return None if df.empty else df.index[-1]
        except Exception as exc:  # peek must never break the polling loop
            logger.warning("Cannot peek latest bar time for %s: %s", symbol, exc)
            return None

    def get_last_bar_time():
        """Timestamp of the newest closed candle the last scan processed
        (``None`` before the first successful scan)."""
        return levels_cache.get("__last_processed_bar_time__")

    # Attach metadata for introspection / gating
    check_new_bar.symbol = symbol
    check_new_bar.config = symbol_cfg
    check_new_bar.model = model
    check_new_bar.calibrator = calibrator
    check_new_bar.pipeline_params = pipeline_params
    check_new_bar.peek_latest_bar_time = peek_latest_bar_time
    check_new_bar.get_last_bar_time = get_last_bar_time
    return check_new_bar


# ---------------------------------------------------------------------------
# v1.1 Multi-Pattern Engine — §9.1 flow (keeps the legacy single-pattern
# ``create_symbol_engine`` interface above untouched).
# ---------------------------------------------------------------------------
#
# Spec §9.1 (multi-pattern):
#   symbol config (assignments[])                  # symbol x pattern x TF x state
#     for each assignment (state in {live, shadow}):
#       df = fetch(tf=assignment.timeframe)        # TF-per-assignment (§3.5)
#       events = detector.detect(df, config)
#       validate_causality(events)                 # runtime check (§3.4)
#       events -> attach model_prob
#     CorrelationManager.group(all events)         # §4.2 -> dedup/confluence/independent
#     shadow assignments -> record lake, NO signal
#     live               -> Exposure caps §4.4 -> PendingSignal
#     §3.3: at entry, if current_price moved > max_entry_drift_atr*ATR from
#           entry_price -> drop/stale (no signal, avoid chasing price).

@dataclass
class PatternAssignment:
    """One symbol x pattern x timeframe x state assignment (spec §3.5, §5.5)."""
    assignment_id: str
    pattern_name: str
    timeframe: str
    state: str = LIFECYCLE_LIVE          # "live" | "shadow" (lifecycle state)
    detector: Any = None                 # BasePatternDetector instance
    config: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    # Optional tier-2 inference — ``scorer(events, df)->dict[event_id->prob]``
    scorer: Callable[[list[PatternEvent]], dict[str, float]] | None = None
    feature_schema_version: str = ""
    risk_fraction: float = 1.0
    # §3.3 guard config
    max_entry_drift_atr: float = 1.5
    # Real market price at entry-check time (from a live quote/candle close).
    # ``None`` -> the guard cannot run (engine decides per fail_closed_drift).
    current_market_price: float | None = None
    # --- HMM regime plugin (requirements v1.0 §4.3, §5; default OFF) -----
    # A fitted regime plugin attached to THIS assignment (overrides the
    # engine-level shared plugin when set).  ``None`` -> no regime wiring.
    regime_plugin: Any = None
    # Resolved regime wiring config (see live.engine.regime_wiring).  ``None`` /
    # {} -> feature emitter + hard gate disabled (legacy behaviour).
    regime_config: dict[str, Any] = field(default_factory=dict)
    # Hard-gate rules map (per-pattern); ``None`` -> gate off.
    regime_filter: dict[str, Any] | None = None
    # Registry §4.3 metadata used for fail-closed (§6.3) — exact train-time
    # feature_list + optional_plugins (``hmm_regime required=true``).
    model_feature_list: list[str] = field(default_factory=list)
    model_optional_plugins: list[dict[str, Any]] = field(default_factory=list)
    # Explicit emitter/gate activation (rarely set directly — normally derived
    # from regime_config). Kept for testing and engine-level overrides.
    emitter_enabled: bool = False
    gate_enabled: bool = False
    # Precomputed regime states (list[RegimeState], aligned to df index).
    # ``None`` -> engine computes them from regime_plugin when needed.
    regime_states: list[Any] | None = None


class MultiPatternEngine:
    """Orchestrates multiple pattern assignments for one symbol (§9.1).

    Each call to :meth:`check_new_bar` fetches the full (already-closed) OHLCV
    frame for the symbol, runs every **live/shadow** assignment through its
    detector + causality validation + tier-2 score, feeds ALL events through the
    :class:`CorrelationManager`, applies the §3.3 entry-drift guard, and for
    **live** events maps each resolved group to exactly ONE
    :class:`SignalCandidate`.  **Shadow** assignments produce events + optional
    lake records but never a SignalCandidate (no order).

    The legacy ``mcp_client.get_chart_history(symbol)`` adapter is reused so a
    :class:`MCPCandleSource` (polling engine) or a stub satisfies ``candle_fn``.
    """

    def __init__(
        self,
        symbol: str,
        assignments: list[PatternAssignment],
        candle_fn: Callable[[str], pd.DataFrame],
        correlation_cfg: CorrelationConfig | None = None,
        keep_last_events: bool = True,
        # --- HMM regime plugin (requirements v1.0, default OFF) ----------
        # Shared fitted regime plugin; each assignment may override with its
        # own ``regime_plugin``.  When both are None, no regime wiring runs
        # and behaviour is identical to a legacy engine (golden unchanged).
        regime_plugin: Any = None,
        regime_config_source: dict[str, Any] | None = None,
        # --- rework §1.3.3: opposite-direction overlap guard (default OFF) --
        # When ON, events whose structure is claimed by a contrary-direction
        # event are dropped before grouping.  The backtest runner exposes the
        # same switch (`run_symbol_backtest(opposite_overlap_guard=...)`) so
        # live and backtest stay identical (§12 parity).
        opposite_overlap_guard: bool = False,
    ) -> None:
        if not assignments:
            raise ValueError("MultiPatternEngine requires >= 1 assignment")
        self.symbol = symbol
        self.assignments = assignments
        self.candle_fn = candle_fn
        self.corr_manager = CorrelationManager(correlation_cfg)
        self.corr_config = correlation_cfg or CorrelationConfig()
        self.keep_last_events = keep_last_events
        self.regime_plugin = regime_plugin
        self.opposite_overlap_guard = bool(opposite_overlap_guard)
        # introspection metadata (mirrors the single-pattern engine closure)
        self.config: dict[str, Any] = {"assignments": [a.assignment_id for a in assignments]}
        self._last_events: list[PatternEvent] = []
        self._last_candidates: list[SignalCandidate] = []
        self._event_owner: dict[str, PatternAssignment] = {}
        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._levels_cache: dict[str, dict[str, Any]] = {}
        # Resolve each assignment's regime slot (emitter + hard gate).
        self._assign_regime_slots(regime_plugin, regime_config_source)

    # ------------------------------------------------------------------
    # Candle access + per-assignment fetch
    # ------------------------------------------------------------------
    def _assign_regime_slots(
        self, shared_plugin: Any, config_source: dict[str, Any] | None
    ) -> None:
        """Resolve a RegimeSlot per assignment (emitter + hard gate).

        Default OFF: a slot is only built when the assignment opts into the
        plugin (its own ``regime_plugin`` or the shared one) AND the resolved
        config enables it.  Legacy assignments get ``None`` → no wiring.
        """
        from live.engine.regime_wiring import (
            RegimeSlot,
            resolve_regime_config,
            resolve_regime_wiring,
        )

        self._regime_slots: dict[str, Any] = {}
        for a in self.assignments:
            if a.regime_states is not None:
                # explicit precomputed states — wrap a minimal slot
                slot = RegimeSlot(
                    plugin=a.regime_plugin or shared_plugin,
                    emitter_enabled=a.emitter_enabled,
                    gate_enabled=a.gate_enabled,
                    rules=a.regime_filter or {},
                    hmm_config=a.regime_config or {},
                    assignment_id=a.assignment_id,
                    pattern_name=a.pattern_name,
                    model_feature_list=a.model_feature_list,
                    model_optional_plugins=a.model_optional_plugins,
                )
                if hasattr(slot, "set_states"):
                    slot.set_states(list(a.regime_states))
                self._regime_slots[a.assignment_id] = slot
                continue
            plugin = a.regime_plugin or shared_plugin
            if plugin is None:
                self._regime_slots[a.assignment_id] = None
                continue
            src = config_source or {}
            resolved = dict(a.regime_config) if a.regime_config else (
                resolve_regime_config(
                    # plugin section carries the master ``enabled`` switch —
                    # pass it through so engine-level config_source wiring
                    # honours plugin.enabled from the hmm_regime YAML.
                    src.get("plugin") if isinstance(src.get("plugin"), dict) else None,
                    src.get("hmm") if isinstance(src.get("hmm"), dict) else None,
                    src.get("regime_filter") if isinstance(src.get("regime_filter"), dict) else None,
                    a.pattern_name,
                )
                if src
                else {}
            )
            # master switch: explicit ``enabled`` (from the plugin YAML) wins;
            # otherwise any explicit emitter/gate flag enables the slot.
            master = resolved.get("enabled")
            if master is None:
                resolved["enabled"] = bool(
                    a.emitter_enabled
                    or a.gate_enabled
                    or resolved.get("rules")
                    or (a.regime_filter is not None)
                )
            resolved["emitter_enabled"] = bool(
                a.emitter_enabled or resolved.get("emitter_enabled", False)
            )
            resolved["gate_enabled"] = bool(
                a.gate_enabled or resolved.get("gate_enabled", False)
            )
            if a.regime_filter is not None:
                resolved["rules"] = a.regime_filter
            self._regime_slots[a.assignment_id] = resolve_regime_wiring(
                plugin,
                resolved,
                a.assignment_id,
                a.pattern_name,
                model_feature_list=a.model_feature_list,
                model_optional_plugins=a.model_optional_plugins,
            )

    def _regime_slot(self, a: PatternAssignment) -> Any:
        return self._regime_slots.get(a.assignment_id)

    def _fetch(self, timeframe: str) -> pd.DataFrame:
        cache_key = f"{self.symbol}:{timeframe or 'default'}"
        if cache_key in self._bar_cache:
            return self._bar_cache[cache_key]
        df = self.candle_fn(self.symbol)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns for {self.symbol}: {required - set(df.columns)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        self._bar_cache[cache_key] = df
        return df

    # ------------------------------------------------------------------
    def check_new_bar(self) -> list[SignalCandidate]:
        """Run the §9.1 multi-pattern scan for this symbol.

        Returns the live `SignalCandidate`s (0 for purely-shadow symbols /
        no actionable events).  Shadow assignments contribute events to the
        correlation grouping so a live signal overlapping a shadow one de-dups
        correctly, but shadow events never emit an order.
        """
        self._last_events = []
        self._last_candidates = []
        self._event_owner.clear()

        # 1. fetch once per (unique) timeframe
        # 2. run each assignment
        events_by_aid: dict[str, list[PatternEvent]] = {}
        per_df: dict[str, pd.DataFrame] = {}
        for a in self.assignments:
            if a.state not in (LIFECYCLE_LIVE, LIFECYCLE_SHADOW):
                continue  # degraded/retired -> no new signals (§5.4)
            df = self._fetch(a.timeframe)
            if df.empty:
                continue
            per_df[a.assignment_id] = df
            evs = self._run_assignment(a, df)
            events_by_aid[a.assignment_id] = evs
            for e in evs:
                self._event_owner[e.event_id] = a
            self._last_events.extend(evs)

        if not self._last_events:
            return []

        # 2b. Opposite-direction overlap guard (rework spec §1.3.3).
        #     §4 grouping can never merge contrary events (they are different
        #     ideas by construction), so a double top and a double bottom read
        #     off ONE shared middle swing would both survive and the engine
        #     could open opposing trades on the same structure.  Drop the
        #     weaker reading before grouping so downstream sees one idea.
        #
        #     OFF by default so the runner/live parity contract (§12) holds:
        #     the backtest harness must apply the SAME filter to keep
        #     "backtest ≡ live".  Enable both sides together via
        #     `opposite_overlap_guard` — see docs/pattern_rework_resolution.md.
        if self.opposite_overlap_guard:
            before = len(self._last_events)
            self._last_events = drop_opposite_overlap(self._last_events)
            if len(self._last_events) != before:
                logger.info(
                    "MultiPatternEngine %s: opposite-direction overlap guard "
                    "dropped %d of %d events (shared structure, rework §1.3.3)",
                    self.symbol, before - len(self._last_events), before,
                )

        # 3. CorrelationManager.group over ALL assignments (dedup/confluence)
        groups = self.corr_manager.group(self._last_events, config=self.corr_config)

        candidates: list[SignalCandidate] = []
        for g in groups:
            if g.representative is None:
                continue
            rep = g.representative
            # find owning assignment for state/config (by event_id first)
            aid = self._event_owner.get(rep.event_id) or _assignment_of(self.assignments, rep)
            if aid is None:
                continue
            if aid.state == LIFECYCLE_SHADOW:
                # §5.2 shadow: record only -> NO signal
                self._record_shadow(rep, g, aid)
                continue

            # 4. §3.3 entry-drift guard (avoid chasing price)
            if not _entry_drift_ok(rep, aid, per_df.get(aid.assignment_id), self.symbol):
                rep.attributes["discard_reason"] = "stale"  # §3.3 / §7.1
                continue

            # 5a. §6.3 fail-closed: a model trained WITH hmm_* features and a
            #     required hmm_regime plugin must NOT emit a signal when the
            #     plugin is inactive / not wired — reindexing would give NaN
            #     for every hmm_* column and the model_prob would be meaningless.
            #     The same suppression applies when the emitter was active but
            #     computation failed in _run_assignment (event stamped
            #     discard_reason="hmm_unavailable").
            rslot = self._regime_slot(aid)
            if (rslot is None and _model_requires_plugin(aid)) or rep.attributes.get("discard_reason") == "hmm_unavailable":
                if rep.attributes.get("discard_reason") != "hmm_unavailable":
                    rep.attributes["discard_reason"] = "hmm_unavailable"  # §6.2 / §7.1
                logger.warning(
                    "MultiPatternEngine %s: %s requires hmm_regime (%s) but the "
                    "plugin is inactive/unavailable — signal suppressed (fail-closed §6.3)",
                    self.symbol, aid.pattern_name, aid.model_id,
                )
                continue

            # 5. HMM regime hard gate (requirements §5, guide §8): when the
            #    assignment's slot gate is active and the pattern has a rule,
            #    block events whose regime is disallowed / too-low confidence.
            #    A blocked event keeps discard_reason="regime_blocked" (it is
            #    still persisted to the Event Lake — spec §7.1).
            if rslot is not None and rslot.gate_active():
                allowed, _reason = _gate_allowed(
                    rep, rslot, per_df.get(aid.assignment_id),
                )
                if not allowed:
                    rep.attributes["discard_reason"] = "regime_blocked"  # §7.1
                    continue

            cand = self._group_to_candidate(g, aid, per_df.get(aid.assignment_id))
            if cand is not None:
                candidates.append(cand)

        self._last_candidates = candidates
        if candidates:
            _record_signal_stat(self.symbol, "pass", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    def _run_assignment(self, a: PatternAssignment, df: pd.DataFrame) -> list[PatternEvent]:
        """detect -> validate_causality -> attach model_prob for one assignment."""
        if a.detector is None:
            return []
        try:
            events = a.detector.detect(df, a.config)
        except Exception as exc:
            # Fail-open technically, fail-closed in signals (§2.2).  A detector
            # exception must not crash the whole symbol scan.
            logger.warning("MultiPatternEngine %s: %s detect failed: %s",
                           self.symbol, a.pattern_name, exc)
            return []
        events = [e for e in events if e is not None]
        # stamp assignment lifecycle onto the event (contract §2.1 lifecycle_state)
        for e in events:
            e.lifecycle_state = a.state
            if e.timeframe in (None, ""):
                e.timeframe = a.timeframe
        # runtime causality check (§3.4) — a violation drops the batch (fail
        # closed in signals) + alerts the GUI log tab.
        try:
            a.detector.validate_causality(events)
        except Exception as exc:
            logger.warning("MultiPatternEngine %s: causality violation in %s: %s "
                           "— events dropped", self.symbol, a.pattern_name, exc)
            syslog("WARN", "SignalEngine", self.symbol,
                   f"Causality violation in {a.pattern_name}: {exc} — events dropped",
                   extra={"pattern_name": a.pattern_name, "assignment_id": a.assignment_id})
            events = []
        # HMM regime plugin (requirements §4.2 / §5, guide §5 step 4): when the
        # assignment's slot needs the regime (emitter AND/OR hard gate), compute
        # the causal states ONCE and cache them on the slot; then inject hmm_*
        # attributes only when the feature emitter is enabled.  ``None`` events
        # are skipped; a failure must never crash the scan (fail closed).
        slot = self._regime_slot(a)
        emitter_failed = False
        if events and slot is not None and (slot.emitter_active() or slot.gate_active()):
            try:
                states = slot.states()
                if states is None:
                    states = slot.plugin.predict(df, slot.hmm_config or None)
                    slot.set_states(list(states))
                if slot.emitter_active():
                    slot.emitter.attach(events, states, df)
            except Exception as exc:
                logger.warning("MultiPatternEngine %s: regime computation failed for %s: %s",
                               self.symbol, a.pattern_name, exc)
                # §6.2/§6.3 fail-closed: when the feature emitter is ACTIVE but
                # computation failed, the hmm_* features are unavailable — a
                # model trained WITH hmm_* must not be scored on NaNs.  Stamp
                # the events and skip tier-2 scoring (model_prob stays None →
                # candidate fallback 0.5 / gate suppression in check_new_bar).
                if slot.emitter_active() and _model_requires_plugin(a):
                    for e in events:
                        e.attributes["discard_reason"] = "hmm_unavailable"
                    emitter_failed = True
        # tier-2 calibrated probability (§9.1 attach model_prob)
        if a.scorer is not None and not emitter_failed:
            try:
                proba = a.scorer(events)
                for e in events:
                    if e.event_id in proba and proba[e.event_id] is not None:
                        e.model_prob = float(proba[e.event_id])
            except Exception as exc:
                logger.warning("MultiPatternEngine %s: scorer failed for %s: %s",
                               self.symbol, a.pattern_name, exc)
        return events

    def _group_to_candidate(
        self, g: ResolvedGroup, a: PatternAssignment,
        df: pd.DataFrame | None,
    ) -> SignalCandidate | None:
        rep = g.representative
        if rep is None:
            return None
        direction = "long" if rep.direction in ("bullish", "buy", "long") else "short"
        conf_time = pd.Timestamp(rep.known_at_ts)
        entry_price = (rep.entry_price if not _nan(rep.entry_price)
                       else _price_at(df, pd.Timestamp(rep.known_at_ts)))
        if _nan(entry_price):
            return None
        stop = rep.stop_price if not _nan(rep.stop_price) else 0.0
        target = rep.target_price if rep.target_price is not None else 0.0
        rule_score = float(rep.rule_score or 0.0)
        model_prob = float(rep.model_prob) if _finite(rep.model_prob) else 0.5  # type: ignore[arg-type]
        # B1 (bug summary): optional minimum-probability gate. Off unless a
        # threshold is configured (per-assignment YAML ``min_model_prob`` or
        # the module default map) so the historical behaviour is preserved
        # while a low-probability event can no longer pass silently when the
        # operator HAS opted in.
        prob_min = _resolve_min_model_prob(a)
        if prob_min is not None and model_prob < prob_min:
            rep.attributes["discard_reason"] = "low_probability"   # §7.1
            logger.info(
                "signal_engine: %s %s discarded — model_prob %.4f < min_model_prob %.4f",
                self.symbol, rep.event_id, model_prob, prob_min,
            )
            return None
        short = get_registry().short_name(rep.pattern_name)
        return SignalCandidate(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry_price, 5),
            stop_price=round(stop, 5),
            target_price=round(target, 5),
            entry_time=conf_time.to_pydatetime(),
            event_time=pd.Timestamp(rep.known_at_ts).to_pydatetime(),
            event_id=rep.event_id,
            rule_score=round(rule_score, 2),
            model_prob=round(model_prob, 4),
            # B2: per-pattern normalization (LSW 0..100, DB/DT 0..1)
            combined_score=_combine_score(
                _normalize_rule_score(rep.pattern_name, rule_score), model_prob),
            penetration_atr=float(_attr(rep, "penetration_atr", 0.0)),
            wick_ratio=float(_attr(rep, "wick_ratio", 0.0)),
            reclaim_atr=float(_attr(rep, "reclaim_atr", 0.0)),
            h1_trend=int(_attr(rep, "h1_trend", 0)),
            confirmation_delay_bars=float(_attr(rep, "confirmation_delay_bars", 0.0)),
            confirmation_range_atr=float(_attr(rep, "confirmation_range_atr", 0.0)),
            atr_value=float(_attr(rep, "atr_value", 0.0)),
            metadata={"pattern_name": rep.pattern_name,
                      "group_id": g.group_id,
                      "mode": g.mode,
                      "n_patterns": g.n_patterns},
            # v1.1 lineage §9.2
            pattern_name=rep.pattern_name,
            pattern_version=rep.pattern_version,
            pattern_short=short,
            order_comment=order_comment_for_event(rep),
            confluence_score=g.confluence_score,
            confluence_group_id=g.group_id,
            assignment_id=a.assignment_id,
            risk_fraction=a.risk_fraction,
            lifecycle_state=a.state,
        )

    def _record_shadow(self, rep: PatternEvent, g: ResolvedGroup, a: PatternAssignment) -> None:
        """§5.2 shadow bookkeeping hook — persist/notify, never emit a signal."""
        rep.attributes.setdefault("shadow_mode", True)
        rep.attributes.setdefault("confluence_score", g.confluence_score)
        logger.info("MultiPatternEngine %s [shadow] %s event %s recorded (no order)",
                    self.symbol, a.pattern_name, rep.event_id)

    # -- introspection -------------------------------------------------------
    def peek_latest_bar_time(self):
        df = self._fetch(self.assignments[0].timeframe)
        return None if df.empty else df.index[-1]

    def get_last_bar_time(self):
        """Latest known_at among the events of the last scan (or None)."""
        if not self._last_events:
            return None
        return max(pd.Timestamp(e.known_at_ts) for e in self._last_events)

    def get_last_events(self) -> list[PatternEvent]:
        return list(self._last_events)


# ---------------------------------------------------------------------------
# §3.3 helpers — entry-drift guard (do not chase price far from entry)
# ---------------------------------------------------------------------------

def _assignment_of(assignments: list[PatternAssignment], ev: PatternEvent) -> PatternAssignment | None:
    """Find the owning assignment by matching its detector's pattern_name."""
    name = (ev.pattern_name or "").lower()
    for a in assignments:
        a_name = (getattr(a.detector, "name", None) or a.pattern_name or "").lower()
        if a_name == name:
            return a
    for a in assignments:
        if (a.pattern_name or "").lower() == name:
            return a
    return assignments[0] if assignments else None


def _model_requires_plugin(a: PatternAssignment) -> bool:
    """§6.3: True when the assignment's model was trained with ``hmm_*``
    features and lists a required ``hmm_regime`` optional plugin (registry
    §4.3) — the model cannot run without the regime plugin active."""
    from live.engine.feature_emitter import requires_hmm_regime

    has_hmm = any(str(f).startswith("hmm_") for f in a.model_feature_list)
    return has_hmm and requires_hmm_regime(a.model_optional_plugins)


def _gate_allowed(rep: PatternEvent, slot: Any, df: pd.DataFrame | None) -> tuple[bool, str]:
    """Requirements §5 hard gate on *rep* using the slot's regime.

    Returns ``(allowed, reason)``.  The regime state is resolved causally from
    the slot's cached states (computed in ``_run_assignment``) at/just-before
    the event's ``known_at``; if none is available the gate fails closed
    (``no_regime``) so "gate on, plugin silent" never silently lets everything
    through (§5 / §6.3).
    """
    from live.engine.feature_emitter import state_at_confirm_bar
    from live.engine.hard_gate import is_allowed

    states = slot.states() if slot is not None else None
    regime = state_at_confirm_bar(states, df, rep) if states and df is not None else None
    return is_allowed(rep, regime, slot.gate_rules() if slot is not None else None)


def _entry_drift_ok(
    ev: PatternEvent, a: PatternAssignment,
    df: pd.DataFrame | None, symbol: str,
) -> bool:
    """§3.3 — at entry, if current_price moved beyond ``max_entry_drift_atr`` ATR
    from the recorded entry_price, the event is discarded as stale (avoid chasing
    price).  ``current_price`` is the newest *closed* close when the assignment
    does not carry an explicit live quote; falls back to the market close at the
    scan.  Events recorded by a long-gone pattern pass through (no drift data) —
    the engine fail-closes to False only when a genuine price gap proves drift.
    """
    if _nan(ev.entry_price):
        return False
    max_drift = float(a.max_entry_drift_atr or 0.0)
    if max_drift <= 0:
        return True  # guard disabled
    current = a.current_market_price
    if current is None and df is not None and not df.empty:
        current = float(df["close"].iloc[-1])
    if current is None or (isinstance(current, float) and not _finite(current)):
        # no market price to compare against — cannot prove drift, fail open only
        # if the event still maps to an executable upcoming bar; here we keep it.
        return True
    atr_val = _event_atr(ev)
    if atr_val is None or not _finite(atr_val) or atr_val <= 0:
        return True  # cannot scale drift without ATR
    if abs(float(current) - float(ev.entry_price)) > max_drift * atr_val:
        return False
    return True


def _event_atr(ev: PatternEvent) -> float | None:
    attrs = getattr(ev, "attributes", {}) or {}
    for key in ("atr_value", "atr"):
        v = attrs.get(key)
        if v is not None:
            try:
                f = float(v)
                if _finite(f):
                    return f
            except (TypeError, ValueError):
                continue
    return None


def _resolve_min_model_prob(a: Any) -> float | None:
    """Resolve the optional ``min_model_prob`` threshold for an assignment.

    Priority: the assignment's own config (per-symbol/pattern YAML) then the
    module default map (opt-in, empty by default).  ``None`` means "no
    probability gate" — the engine keeps its historical behaviour.
    """
    cfg = getattr(a, "config", None) or {}
    raw: Any = cfg.get("min_model_prob")
    if raw is None:
        raw = _DEFAULT_MIN_MODEL_PROB.get(str(getattr(a, "pattern_name", "")).lower())
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or not (0.0 <= v <= 1.0):
        return None
    return v


def _normalize_rule_score(pattern_name: str, rule_score: Any) -> float:
    """Normalize a pattern's ``rule_score`` to [0, 1] (B2, bug summary).

    Plugins do NOT share one scale: ``liquidity_sweep`` declares
    "weighted rule score 0..100" (observed 32..51) while the double/wedge/H&S
    plugins clamp theirs to [0, 1] (observed 0.51..0.99).  The multi-pattern
    path used to hard-code ``rule_score / 100.0``, which shrank every DB/DT
    score to ~1 % of its value and made ``combined_score`` meaningless for
    every pattern except LSW.

    Detection: use the declared scale when known, otherwise infer it from the
    magnitude (> 1 ⇒ 0..100 scale).  Result is always clamped to [0, 1].
    """
    try:
        v = float(rule_score)
    except (TypeError, ValueError):
        return 0.0
    if v != v:                                  # NaN
        return 0.0
    scale = _RULE_SCORE_SCALE.get(str(pattern_name).lower())
    if scale is None:
        scale = 100.0 if abs(v) > 1.0 else 1.0
    return max(0.0, min(1.0, v / scale))


def _combine_score(rule_score: float, model_prob: float) -> float:
    """Mean of a normalized rule score and the model probability."""
    return round((rule_score + model_prob) / 2.0, 4)


def _price_at(df: pd.DataFrame | None, ts: pd.Timestamp) -> float:
    if df is None or df.empty:
        return float("nan")
    import numpy as _np
    times = df.index.to_numpy(dtype="datetime64[ns]")
    pos = int(_np.searchsorted(times, _np.datetime64(ts), side="left"))
    if pos >= len(df):
        pos = len(df) - 1
    return float(df["open"].iloc[pos]) if not df.empty else float("nan")


def _attr(ev: PatternEvent, key: str, default: float) -> float:
    attrs = getattr(ev, "attributes", {}) or {}
    st = getattr(ev, "structure_levels", {}) or {}
    raw = attrs.get(key, st.get(key, default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _nan(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return f != f or v is None


def _finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    import math
    return math.isfinite(f) and f == f
