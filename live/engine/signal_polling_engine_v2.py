"""
signal_polling_engine.py — SignalPollingEngine V2

Orchestrates the per-symbol signal engine loop:
  1. On every tick, scan registered (validated) symbols for new bars
  2. Route candidates through RiskGuard (kill-switch check, position limits)
  3. Add pending signals to SharedAppState for the GUI to display
  4. For automation levels 2 (semi) / 3 (full), auto-dispatch orders;
     level 1 (manual) never auto-sends — the user confirms in the GUI.

M15 new-bar gating: each symbol's engine only runs when a genuinely new
closed M15 candle has appeared since the last scan (per-symbol last bar
time).  Unchanged frames are skipped with an INFO log — detection is never
re-run on the same frame, so Trigger counts distinct closed candles and the
same event is not re-emitted every 30s poll cycle.

Thread-safe: runs in its own daemon thread, updates shared state via
the thread-safe SharedAppState singleton.

Integration wiring:
  - signal_engine_v2.create_symbol_engine() per symbol
  - risk_guard_v2.RiskGuard.evaluate_new_signal()
  - execution_layer_v2.ExecutionLayer.send_order()
  - logger_v2.log_user_action() for audit trail
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from live.engine.signal_engine_v2 import (
    ModelArtifactNotFoundError,
    SignalCandidate,
    SymbolConfigNotFoundError,
    SymbolNotValidatedError,
    create_symbol_engine,
)
from live.logging.logger_v2 import log_user_action
from live.state.shared_app_state_v2 import (
    AutomationLevel,
    PendingSignal,
    SharedAppState,
)

logger = logging.getLogger("signal_polling_engine")


def _to_epoch(dt: Any) -> float:
    """Convert ``dt`` (a datetime or anything with ``.timestamp()``) to a unix
    epoch; ``0.0`` when it is missing/unconvertible (``0.0`` = not recorded).

    Naive datetimes are interpreted as local time via ``.timestamp()``, which
    matches the codebase convention: MT5 terminal-local times round-trip to the
    same wall clock the GUI shows via ``datetime.fromtimestamp`` in ``fmt_time``.
    """
    try:
        return float(dt.timestamp()) if hasattr(dt, "timestamp") else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Candle source adapter (ExecutionLayer -> signal-engine data interface)
# ---------------------------------------------------------------------------

class MCPCandleSource:
    """Expose the DataFrame-returning ``get_chart_history(symbol)`` that
    ``signal_engine_v2`` requires, backed by ``ExecutionLayer.get_new_candles``
    (MCP tool ``get_chart_history``).

    ExecutionLayer itself does not implement the engine's chart interface, so
    the polling engine hands this adapter to every per-symbol engine.  The
    window is always requested relative to *now* so the newest closed M15
    candles are returned whenever the market provides them.
    """

    _TIME_FMT = "%Y.%m.%d %H:%M:%S"

    def __init__(self, execution_layer: Any, lookback_days: int = 14) -> None:
        self._el = execution_layer
        self._lookback_days = max(1, lookback_days)

    def get_chart_history(self, symbol: str) -> pd.DataFrame:
        """Return the newest closed M15 candles as an OHLCV DataFrame.

        Empty frame when the execution layer is unavailable or returns no
        candles — the engine treats that as "no data this cycle".
        """
        if self._el is None:
            return pd.DataFrame()
        since = (datetime.now() - timedelta(days=self._lookback_days)).strftime(
            "%Y-%m-%dT00:00:00"
        )
        try:
            raw = self._el.get_new_candles(symbol, period="M15", since=since)
        except Exception as exc:
            logger.warning("MCPCandleSource: no candles for %s: %s", symbol, exc)
            return pd.DataFrame()
        return self._to_frame(raw or [], symbol)

    def _to_frame(self, raw: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            time_raw = c.get("time", c.get("datetime", c.get("timestamp")))
            try:
                ts = pd.Timestamp(time_raw)
            except Exception:
                try:
                    ts = pd.to_datetime(time_raw, format=self._TIME_FMT)
                except Exception:
                    continue
            try:
                rows.append({
                    "time": ts,
                    "open": float(c.get("open")),
                    "high": float(c.get("high")),
                    "low": float(c.get("low")),
                    "close": float(c.get("close")),
                    "volume": float(c.get("tick_volume", c.get("volume", 0)) or 0),
                })
            except (TypeError, ValueError):
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("time")
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[["open", "high", "low", "close", "volume"]]


class SignalPollingEngine:
    """Background polling engine for multi-symbol signal generation.

    Owns one signal-engine closure per validated symbol and runs
    a 30-second polling loop that:
      1. Gates each engine on a genuinely new closed M15 candle
         (per-symbol last-bar-time tracking; unchanged frames skip)
      2. Calls the engine to generate signal candidates
      3. Pipes them through RiskGuard
      4. Adds accepted candidates to shared state as PendingSignal
      5. At automation level 3 (full-auto), dispatches immediately

    Thread-safe: runs its own daemon thread.
    """

    def __init__(
        self,
        state: SharedAppState,
        execution_layer: Any,  # ExecutionLayer instance
        risk_guard: Any,       # RiskGuard instance
        polling_interval_s: int = 30,
        candle_source: Any | None = None,
    ) -> None:
        self._state = state
        self._exec_layer = execution_layer
        self._risk_guard = risk_guard
        self._polling_interval_s = polling_interval_s
        # Data source handed to every per-symbol engine.  ExecutionLayer does
        # not implement get_chart_history(), so we wrap it by default.
        self._candle_source = (
            candle_source if candle_source is not None else MCPCandleSource(execution_layer)
        )

        # Per-symbol engine registry: symbol -> callable
        self._engines: dict[str, Callable[[], list[SignalCandidate]]] = {}

        # Per-symbol M15 gating: symbol -> timestamp of the newest closed
        # M15 candle already scanned.  A symbol's engine is only invoked
        # when this differs from the latest closed bar, so Trigger counts
        # distinct closed candles and unchanged frames never re-run
        # detection (no duplicate event emission).  Guarded by self._lock.
        self._last_bar_time: dict[str, Any] = {}

        # Thread management
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        # Wake-up for immediate scans (GUI kick) — also makes stop() and
        # emergency-stop responsive instead of waiting a full interval.
        self._scan_event = threading.Event()
        # Failed engine-registration backoff: symbol -> earliest retry time.
        self._retry_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _config_override(self, symbol: str) -> dict | None:
        """Registry-backed config override so validated+active symbols whose
        market name differs from a per-symbol YAML file (e.g. ``XAUUSDm``)
        still get an engine from their registered ``model_id``."""
        cfg = self._state.get_symbol_config(symbol)
        if cfg is None:
            return None
        override = {"status": cfg.status, "symbol": {"name": cfg.name}}
        if cfg.model_id:
            override["model_id"] = cfg.model_id
        return override

    def _create_engine_for(self, symbol: str) -> Callable[[], list[SignalCandidate]]:
        """Create a per-symbol engine, preferring the symbol's YAML config and
        falling back to the registry config override (status/model_id)."""
        last_exc: Exception | None = None
        try:
            return create_symbol_engine(symbol, mcp_client=self._candle_source)
        except (SymbolNotValidatedError, SymbolConfigNotFoundError,
                ModelArtifactNotFoundError) as exc:
            last_exc = exc
        override = self._config_override(symbol)
        if override is not None:
            try:
                return create_symbol_engine(
                    symbol, mcp_client=self._candle_source,
                    symbol_cfg_override=override,
                )
            except Exception as exc:  # registry/model/artifact problems
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise ValueError(f"No engine configuration available for {symbol}")

    def _retry_due(self, symbol: str) -> bool:
        return self._retry_at.get(symbol, 0.0) <= time.time()

    def _note_registration_failure(self, symbol: str) -> None:
        self._retry_at[symbol] = time.time() + self._polling_interval_s

    def register_symbol_engine(self, symbol: str) -> bool:
        """Create and register a signal engine for *symbol*.

        Args:
            symbol: Instrument name (e.g. "XAUUSD").

        Returns:
            True if the engine was created successfully.
        """
        try:
            engine = self._create_engine_for(symbol)
            with self._lock:
                self._engines[symbol] = engine
                self._last_bar_time.pop(symbol, None)
                self._retry_at.pop(symbol, None)
            logger.info("Registered signal engine for %s", symbol)
            _log_audit("register_symbol_engine", {"symbol": symbol, "status": "ok"})
            return True
        except (
            SymbolNotValidatedError,
            SymbolConfigNotFoundError,
            ModelArtifactNotFoundError,
        ) as e:
            logger.warning("Cannot create engine for %s: %s", symbol, e)
            self._note_registration_failure(symbol)
            _log_audit("register_symbol_engine", {"symbol": symbol, "error": str(e)})
            return False
        except Exception as e:
            logger.error("Unexpected error creating engine for %s: %s", symbol, e)
            self._note_registration_failure(symbol)
            return False

    def unregister_symbol_engine(self, symbol: str) -> bool:
        """Remove a symbol engine from the poll loop."""
        with self._lock:
            if symbol in self._engines:
                del self._engines[symbol]
                self._last_bar_time.pop(symbol, None)
                logger.info("Unregistered signal engine for %s", symbol)
                _log_audit("unregister_symbol_engine", {"symbol": symbol})
                return True
        return False

    def refresh_engines(self) -> None:
        """Sync engines with the current symbol registry.

        Validated+active symbols without an engine get one created (subject to
        registration-failure backoff).  Engines for unregistered,
        non-validated or inactive symbols are removed.  Called by the GUI
        bridge on every poll cycle so symbols are picked up / dropped the
        moment they are validated / activated / deactivated.
        """
        symbols = self._state.get_registered_symbols()
        with self._lock:
            current = set(self._engines.keys())
            desired = set()

            for sym in symbols:
                cfg = self._state.get_symbol_config(sym)
                if cfg and cfg.status == "validated" and cfg.active:
                    desired.add(sym)

            # Remove stale
            for sym in current - desired:
                del self._engines[sym]
                self._last_bar_time.pop(sym, None)
                self._retry_at.pop(sym, None)
                logger.info("Removed stale engine for %s", sym)

            # Add new (respect failed-registration backoff)
            for sym in sorted(desired - current):
                if not self._retry_due(sym):
                    continue
                self._register_engine_nolock(sym)

    def _register_engine_nolock(self, symbol: str) -> None:
        """Register engine with lock assumed held."""
        try:
            engine = self._create_engine_for(symbol)
            self._engines[symbol] = engine
            self._last_bar_time.pop(symbol, None)
            self._retry_at.pop(symbol, None)
            logger.info("Registered signal engine for %s", symbol)
        except (
            SymbolNotValidatedError,
            SymbolConfigNotFoundError,
            ModelArtifactNotFoundError,
        ) as e:
            logger.warning("Cannot create engine for %s: %s", symbol, e)
            self._note_registration_failure(symbol)
        except Exception as e:
            logger.error("Unexpected error creating engine for %s: %s", symbol, e)
            self._note_registration_failure(symbol)

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread (daemon)."""
        with self._lock:
            if self._running:
                logger.warning("SignalPollingEngine already running")
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._poll_loop, daemon=True,
                name="signal-polling-engine",
            )
            self._thread.start()
        logger.info("SignalPollingEngine started (interval=%ds)", self._polling_interval_s)

    def stop(self) -> None:
        """Signal the polling thread to stop (wakes it immediately)."""
        self._running = False
        self._scan_event.set()  # wake the loop so it exits without delay
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("SignalPollingEngine stopped")

    def request_scan(self) -> None:
        """Ask the background thread to run a poll cycle now (non-blocking).

        Used by the GUI bridge on every main-window timer tick so scans are
        driven by the GUI polling loop but always execute on the engine's own
        daemon thread — the GUI thread never blocks on MCP/detection.
        """
        self._scan_event.set()

    def _poll_loop(self) -> None:
        """Main polling loop — runs until *running* is False or the engine is
        halted by an emergency stop.

        One cycle executes immediately on start, then the loop sleeps on
        ``_scan_event`` for up to ``polling_interval_s``; every
        :meth:`request_scan` (e.g. each GUI timer tick) wakes it for another
        cycle, so scans happen as often as the GUI asks while the engine keeps
        a sane fallback cadence when nothing is kicking it.  The per-symbol
        M15 gate (``_run_gated``) still guarantees a symbol's detection only
        runs when a genuinely new closed candle exists.
        """
        while self._running:
            if self._state.emergency_stop:
                # Emergency stop halts the engine thread cleanly; the GUI
                # bridge restarts it once the stop is released.
                logger.info("SignalPollingEngine halted by emergency stop")
                break
            # Registration sync runs on this thread (never on the GUI thread):
            # symbols are picked up / dropped as they are validated/activated,
            # and model artifacts load here, keeping the GUI responsive.
            try:
                self.refresh_engines()
            except Exception as e:
                logger.error("Engine registration sync error: %s", e, exc_info=True)
            try:
                self._poll_once()
            except Exception as e:
                logger.error("Poll cycle error: %s", e, exc_info=True)
            if not self._running:
                break
            self._scan_event.clear()
            self._scan_event.wait(timeout=self._polling_interval_s)
        self._running = False
        logger.info("SignalPollingEngine polling thread exited")

    def _poll_once(self) -> None:
        """One poll cycle: scan engines, route through risk guard, add pending."""
        if self._state.emergency_stop:
            logger.debug("Emergency stop active — skipping poll cycle")
            return

        with self._lock:
            engines = dict(self._engines)  # snapshot

        if not engines:
            return

        for symbol, engine_fn in engines.items():
            try:
                candidates = self._run_gated(symbol, engine_fn)
            except Exception as e:
                logger.error("Engine error for %s: %s", symbol, e)
                # Auto-reconnect on connection errors
                if "connection" in str(e).lower() or "mcp" in str(e).lower():
                    try:
                        self._exec_layer._ensure_session()
                    except Exception:
                        pass
                continue

            if not candidates:
                continue

            for cand in candidates:
                self._process_candidate(cand)

    # ------------------------------------------------------------------
    # M15 new-bar gating (Trigger counts distinct closed candles)
    # ------------------------------------------------------------------

    def _run_gated(self, symbol: str, engine_fn: Callable) -> list[SignalCandidate]:
        """Run *engine_fn* only when a genuinely new closed M15 candle exists.

        Every cycle we peek the newest closed candle time (engine metadata —
        no detection, no statistics) and compare it with the per-symbol time
        we last scanned:

        * unchanged frame  -> skip with an INFO log; counters untouched and
          the same event is never re-detected/re-emitted;
        * new closed bar   -> record the bar, run detection exactly once
          (Trigger +1 for that bar), then reconcile with the bar the scan
          actually processed in case an M15 closed between peek and fetch.

        The gate keeps per-symbol state thread-safe (``self._lock``) and is
        intentionally placed so unchanged frames never enter detection.

        Raises: propagates engine exceptions to the caller (existing poll-loop
        error handling / reconnect) — a failed run is NOT retried on the same
        bar, preserving exactly-one-Trigger-per-bar.
        """
        peek = getattr(engine_fn, "peek_latest_bar_time", None)
        if peek is None:
            # Engine without gating metadata (legacy) — run as before.
            with self._lock:
                self._last_bar_time.pop(symbol, None)
            return engine_fn()

        latest = peek()
        with self._lock:
            known = symbol in self._last_bar_time
            last_seen = self._last_bar_time.get(symbol)

        if latest is None:
            # Cannot see candles this cycle (transient) — never run detection
            # on an unknown frame and never touch the recorded bar time.
            logger.info("No candle data for %s this cycle — skipping", symbol)
            return []

        if known and latest == last_seen:
            logger.info("no new M15 candle for %s, skipping", symbol)
            return []

        # A new closed M15 candle — mark it scanned *before* running so an
        # engine failure cannot cause the same bar to be re-run (and double
        # counted) on the next cycle.
        with self._lock:
            self._last_bar_time[symbol] = latest

        candidates = engine_fn()

        # Race guard: if a newer candle closed between our peek and the
        # engine's own fetch, remember what the scan actually processed so it
        # is never re-run on the following cycle either.
        get_processed = getattr(engine_fn, "get_last_bar_time", None)
        if get_processed is not None:
            processed = get_processed()
            if processed is not None and processed != latest:
                with self._lock:
                    self._last_bar_time[symbol] = processed
        return candidates

    def _process_candidate(self, cand: SignalCandidate) -> None:
        """Route one signal candidate through risk guard and add to pending."""
        # 1. Build position dict for risk guard check
        current_positions = [
            {
                "asset": p.asset,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "position_size": p.position_size,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "open_time": p.open_time,
                "position_id": p.position_id,
            }
            for p in self._state.open_positions
        ]

        # 2. Risk guard check
        rg_result = self._risk_guard.evaluate_new_signal(
            asset=cand.symbol,
            current_open_positions=current_positions,
        )
        if not rg_result.get("allowed", False):
            logger.info(
                "Signal for %s rejected by RiskGuard: %s",
                cand.symbol, rg_result.get("reason", ""),
            )
            return

        # 3. Create PendingSignal
        signal_id = str(uuid.uuid4())
        pending = PendingSignal(
            asset=cand.symbol,
            direction=cand.direction,
            entry_price=cand.entry_price,
            stop_loss=cand.stop_price,
            take_profit=cand.target_price,
            rule_score=cand.rule_score,
            model_probability=cand.model_prob,
            timestamp=time.time(),
            signal_id=signal_id,
            entry_time=_to_epoch(cand.entry_time),
        )

        # 4. Add to shared state
        self._state.set_pending_signals(
            [*self._state.pending_signals, pending]
        )
        logger.info(
            "Pending signal added: %s %s @ %.5f (signal_id=%s)",
            cand.symbol, cand.direction, cand.entry_price, signal_id,
        )
        _log_audit("pending_signal_generated", {
            "signal_id": signal_id,
            "symbol": cand.symbol,
            "direction": cand.direction,
            "entry_price": cand.entry_price,
            "rule_score": cand.rule_score,
            "model_prob": cand.model_prob,
        })

        # 5. Auto-send for automation level >= 2 (semi-auto / full-auto).
        #    Manual (level 1) NEVER auto-sends — the user confirms in the GUI.
        auto_level = self._state.automation_levels.get(cand.symbol, AutomationLevel(level=0)).level
        if auto_level >= 2:
            self._auto_send_order(signal_id, cand, auto_level)

    def _auto_send_order(self, signal_id: str, cand: SignalCandidate, level: int) -> None:
        """Auto-send order for automation level >= 2.

        Level 2 (semi-auto): short delay to allow user rejection
        Level 3 (full-auto): immediate dispatch
        """
        if level == 2:
            # Short delay — user can still reject via GUI
            time.sleep(3)

        # Check if signal was still pending (not rejected by user)
        remaining = [s for s in self._state.pending_signals if s.signal_id == signal_id]
        if not remaining:
            return

        logger.info(
            "Auto-sending order for %s %s (level=%d, signal_id=%s)",
            cand.symbol, cand.direction, level, signal_id,
        )

        try:
            result = self._exec_layer.send_order(
                asset=cand.symbol,
                direction=cand.direction,
                entry_price=cand.entry_price,
                stop_loss=cand.stop_price,
                take_profit=cand.target_price,
                signal_id=signal_id,
                rule_score=cand.rule_score,
                model_probability=cand.model_prob,
                automation_level=level,
                kill_switch_blocking=False,
            )
            if result.get("success", False):
                self._state.remove_pending_signal(signal_id)
                self._risk_guard.record_order_sent(cand.symbol)
                logger.info(
                    "Auto-order sent: %s %s (order_id=%s)",
                    cand.symbol, cand.direction,
                    result.get("event_id", result.get("order_id", "?")),
                )
                _log_audit("auto_order_sent", {
                    "signal_id": signal_id,
                    "symbol": cand.symbol,
                    "direction": cand.direction,
                    "level": level,
                    "success": True,
                })
            else:
                # send_order stores the real rejection reason in
                # order_result["error"] (e.g. "Empty MCP response" or a retcode),
                # not at the top level — surface it so failures are diagnosable.
                _err = (result.get("error")
                        or (result.get("order_result") or {}).get("error")
                        or "unknown")
                logger.warning("Auto-order failed: %s", _err)
                _log_audit("auto_order_failed", {
                    "signal_id": signal_id,
                    "symbol": cand.symbol,
                    "error": _err,
                })
        except Exception as e:
            logger.error("Auto-order exception: %s", e)
            _log_audit("auto_order_exception", {"signal_id": signal_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def registered_symbols(self) -> list[str]:
        with self._lock:
            return list(self._engines.keys())

    @property
    def engine_count(self) -> int:
        with self._lock:
            return len(self._engines)


def _log_audit(action_type: str, details: dict[str, Any]) -> None:
    """Log audit trail — silently ignore on failure."""
    try:
        log_user_action(action_type, json.dumps(details, default=str))
    except Exception:
        pass