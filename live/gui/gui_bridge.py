"""
gui_bridge.py — SystemBridge: GUI ↔ Engine module adapter.

Provides a clean API for GUI tabs to interact with the trading engine
(shared_app_state_v2, logger_v2, risk_guard_v2, execution_layer_v2)
without circular imports or direct module coupling.

Key responsibilities:
  1. MCP connection lifecycle with auto-reconnect
  2. Signal engine polling dispatch
  3. Order execution via execution layer
  4. Performance metrics aggregation across symbols
  5. Emergency stop / kill-switch management
  6. **Model Registry API** — load, query, and assign models to symbols
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from live.engine.execution_layer_v2 import _normalize_direction
from live.logging.logger_v2 import (
    init_db,
    log_user_action,
)
from live.logging.logger_v2 import (
    log as syslog,
)
from live.state.shared_app_state_v2 import (
    ModelInfo,
    SharedAppState,
    SymbolConfig,
    get_model_registry,
)


class SystemBridge:
    """Singleton bridge between GUI and engine modules.

    The GUI creates one SystemBridge instance at startup and passes it
    to every tab. The bridge owns the MCP execution layer instance,
    the timer polling loop, and provides safe thread-pool dispatch.
    """

    _instance: SystemBridge | None = None
    _instance_lock = threading.Lock()

    def __init__(self, config_path: str | None = None) -> None:
        self.state = SharedAppState.get_instance()
        self.config_path = config_path or ""
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._mcp_lock = threading.Lock()
        # Background MCP state-refresh daemon thread. MCP network calls are
        # moved OFF the GUI thread — a slow/unreachable MCP server was freezing
        # the whole UI because refresh_state_from_mcp() ran on the main thread.
        self._mcp_refresh_active = False
        self._mcp_refresh_thread: threading.Thread | None = None

        # -------------------------------
        # Cached market bid prices (read by the GUI, written on the MCP
        # refresh daemon thread).  ``fetch_symbol_prices()`` reads this cache
        # so the GUI NEVER blocks on a slow/unreachable MCP server — a
        # synchronous ``get_symbols()``/``get_symbol_info()`` call on the main
        # thread froze the whole UI for ~2-5s because the MCP client retries
        # a failed session with a 2s sleep.
        # -------------------------------
        self._price_cache: dict[str, float] = {}
        self._price_cache_lock = threading.Lock()

        # -------------------------------
        # Model Registry — load from YAML index
        # -------------------------------
        self._model_registry = get_model_registry()
        self._model_registry.load_from_yaml()

        # -------------------------------
        # Load symbol registry from persisted JSON (if exists)
        # Otherwise fall back to YAML seed configs
        # -------------------------------
        n_persisted = self.state.load_registry()
        if n_persisted == 0:
            # No persisted file — load from YAML seed configs
            self._load_symbol_configs()
            # Immediately persist so the user's first GUI save creates the file
            self.state.save_registry()

        # -------------------------------
        # MCP Execution Layer instance
        # -------------------------------
        # We import here to avoid circular import at module level.
        # The execution_layer_v2 module is created by t4; we provide
        # a mock fallback when it is not yet available.
        self._exec_layer: Any = None
        self._init_exec_layer()

        # -------------------------------
        # DB init
        # -------------------------------
        db_path = str(
            Path(__file__).resolve().parent.parent / "db" / "paper_trading_v2.db"
        )
        init_db(db_path)
        self.state.set_mcp_connection_state("disconnected")

        # -------------------------------
        # Background signal engine (GUI-driven)
        # -------------------------------
        # The SignalPollingEngine owns one signal-engine closure per
        # validated+active symbol and runs the per-symbol scans (where the
        # Trigger/Found/Pass counters increment) on its own daemon thread.
        # It is created here but only *started* from the GUI timer via
        # poll_once(), which also keeps symbol registration in sync and
        # requests an immediate scan every cycle.
        self._risk_guard: Any = None
        self._signal_engine: Any = None
        self._init_signal_engine()

        # -------------------------------
        # v1.1 Multi-pattern assignment store (§10.1)
        # symbol -> {assignment_id: {...}} — pattern, model_id, TF, state.
        # The GUI writes these; the signal engine / MultiPatternEngine
        # consumes them to build per-symbol PatternAssignment lists.
        # -------------------------------
        self._pattern_assignments: dict[str, dict[str, dict[str, Any]]] = {}

        # -------------------------------
        # v1.1 Pattern registry (plugin discovery §2.2) — best effort
        # -------------------------------
        self._pattern_registry: Any = None
        try:
            from live.engine.pattern_registry import get_registry
            self._pattern_registry = get_registry()
        except Exception as exc:  # registry must never block GUI startup
            syslog("WARNING", "System", "", f"pattern registry unavailable: {exc}")

        # -------------------------------
        # Persisted pattern assignments (§10.1) — restore on startup so
        # assignments survive restarts exactly like symbol_registry.json.
        # -------------------------------
        self._load_pattern_assignments()

    def _init_signal_engine(self) -> None:
        """Construct the background signal engine + risk guard (best effort)."""
        try:
            from live.engine.signal_polling_engine_v2 import SignalPollingEngine
            from live.state.risk_guard_v2 import RiskGuard

            self._risk_guard = RiskGuard()
            interval = int(os.environ.get("SIGNAL_POLL_INTERVAL_S", "30") or 30)
            self._signal_engine = SignalPollingEngine(
                self.state,
                self._exec_layer,
                self._risk_guard,
                polling_interval_s=interval,
            )
            syslog("INFO", "System", "",
                   "Signal engine polling controller ready "
                   f"(interval={interval}s; started by GUI timer)")
        except Exception as exc:  # never block bridge startup on engine wiring
            self._signal_engine = None
            syslog("WARNING", "System", "",
                   f"Signal engine polling controller unavailable: {exc}")

    # ------------------------------------------------------------------
    # v1.1 multi-pattern assignment store (§10.1) + breakdown data (§10.2)
    # ------------------------------------------------------------------

    def get_pattern_names(self) -> list[str]:
        """Names of the registered pattern plugins (§2.2)."""
        if self._pattern_registry is not None:
            try:
                return self._pattern_registry.pattern_names
            except Exception:
                pass
        return []

    def get_pattern_metadata(self, pattern_name: str) -> dict[str, Any] | None:
        """Public descriptor (name/version/short_name) for a pattern."""
        if self._pattern_registry is not None:
            try:
                return self._pattern_registry.metadata(pattern_name)
            except Exception:
                pass
        return None

    def get_pattern_assignments(self, symbol: str) -> list[dict[str, Any]]:
        """Return the pattern assignments configured for *symbol*."""
        return list(self._pattern_assignments.get(symbol, {}).values())

    def _assignments_file(self) -> Path:
        """Persisted pattern-assignment store (``db/pattern_assignments.json``)."""
        return Path(__file__).resolve().parent.parent / "db" / "pattern_assignments.json"

    def save_pattern_assignments(self) -> None:
        """Persist the §10.1 pattern-assignment store to JSON.

        Mirrors ``SharedAppState.save_registry()`` (atomic temp+rename write).
        """
        path = self._assignments_file()
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(self._pattern_assignments, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.rename(path)
        except Exception:
            pass  # best-effort: in-memory store remains authoritative

    def _load_pattern_assignments(self) -> None:
        """Restore the §10.1 pattern-assignment store from JSON (if present)."""
        path = self._assignments_file()
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for symbol, by_aid in raw.items():
            if not isinstance(by_aid, dict):
                continue
            self._pattern_assignments[str(symbol)] = {
                str(aid): dict(entry) for aid, entry in by_aid.items()
                if isinstance(entry, dict)
            }

    def set_pattern_assignment(
        self,
        symbol: str,
        assignment_id: str,
        *,
        pattern_name: str,
        timeframe: str = "M15",
        state: str = "live",
        model_id: str = "",
        feature_schema_version: str = "",
    ) -> None:
        """Upsert one pattern assignment for *symbol* (§10.1)."""
        self._pattern_assignments.setdefault(symbol, {})[assignment_id] = {
            "assignment_id": assignment_id,
            "pattern_name": pattern_name,
            "timeframe": timeframe,
            "state": state,
            "model_id": model_id,
            "feature_schema_version": feature_schema_version,
        }
        self.save_pattern_assignments()

    def remove_pattern_assignment(self, symbol: str, assignment_id: str) -> None:
        removed = self._pattern_assignments.get(symbol, {}).pop(assignment_id, None)
        if removed is not None:
            self.save_pattern_assignments()

    def pattern_breakdown(self) -> list[dict[str, Any]]:
        """§10.2 — per (pattern, TF) performance table from the trade log.

        Trades are grouped by the pattern key carried on the trade record
        (``pattern`` field, defaulting to asset when absent) + timeframe, and
        aggregated: trades, win rate, profit factor, expectancy (avg net R),
        rolling equity (cumulative net R).
        """
        try:
            records = self.state.trade_log
        except Exception:
            records = []
        groups: dict[str, dict[str, Any]] = {}
        for t in records:
            asset = getattr(t, "asset", "?") or "?"
            pat = getattr(t, "pattern", "") or ""
            tf = getattr(t, "timeframe", "") or ""
            key = f"{pat or asset}|{tf}"
            g = groups.setdefault(key, {
                "pattern": pat or asset, "timeframe": tf, "n": 0,
                "wins": 0, "net_r": 0.0, "gross_r": 0.0,
            })
            g["n"] += 1
            net_r = float(getattr(t, "net_r", 0.0) or 0.0)
            gross_r = float(getattr(t, "gross_r", 0.0) or 0.0)
            g["net_r"] += net_r
            g["gross_r"] += gross_r
            if str(getattr(t, "result", "")) == "win":
                g["wins"] += 1
        rows = []
        for key, g in groups.items():
            n = g["n"]
            win_rate = g["wins"] / n if n else 0.0
            # profit factor: gross wins / |gross losses| (guard div-by-zero)
            gains = sum(max(float(getattr(t, "gross_r", 0.0) or 0.0), 0.0) for t in records
                        if (getattr(t, "pattern", "") or getattr(t, "asset", "")) and
                        ((getattr(t, "pattern", "") or getattr(t, "asset", "?")) + "|" + (getattr(t, "timeframe", "") or "")) == key)
            losses = sum(abs(min(float(getattr(t, "gross_r", 0.0) or 0.0), 0.0)) for t in records
                         if ((getattr(t, "pattern", "") or getattr(t, "asset", "?")) + "|" + (getattr(t, "timeframe", "") or "")) == key)
            pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
            rows.append({
                "pattern": g["pattern"],
                "timeframe": g["timeframe"],
                "trades": n,
                "win_rate": round(win_rate, 4),
                "profit_factor": round(pf, 3),
                "expectancy_r": round(g["net_r"] / n, 4) if n else 0.0,
                "rolling_equity_r": round(g["net_r"], 4),
            })
        rows.sort(key=lambda r: (r["pattern"], r["timeframe"]))
        return rows

    def confluence_expectancy(self) -> list[dict[str, Any]]:
        """§10.2 confluence panel — expectancy by number of agreeing patterns.

        Built from Event Lake outcomes when available (join by event's
        ``confluence_group_id`` would require the lake); this lightweight
        version reports per-group-size buckets from recent engine groups when
        the engine exposes them, and falls back to an empty report.
        """
        base = [
            {"n_patterns": 1, "count": 0, "avg_net_r": 0.0},
            {"n_patterns": 2, "count": 0, "avg_net_r": 0.0},
            {"n_patterns": 3, "count": 0, "avg_net_r": 0.0},
            {"n_patterns": 4, "count": 0, "avg_net_r": 0.0},
        ]
        try:
            engine = getattr(self, "_signal_engine", None)
            if engine is None:
                return base
            # The polling engine holds per-symbol engines; if any is a
            # MultiPatternEngine expose its last groups via last_events.
            groups: dict[int, list[float]] = {}
            engines = getattr(engine, "_engines", {}) or {}
            for eng_fn in engines.values():
                last = getattr(eng_fn, "get_last_events", None)
                if not callable(last):
                    continue
                for ev in last() or []:
                    gid = getattr(ev, "confluence_group_id", None)
                    if gid is None:
                        groups.setdefault(1, []).append(float(getattr(ev, "rule_score", 0.0) or 0.0))
                    else:
                        n = _group_size_from_last(gid, last())
                        groups.setdefault(n, []).append(float(getattr(ev, "rule_score", 0.0) or 0.0))
            out = []
            for row in base:
                n = row["n_patterns"]
                vals = groups.get(n, [])
                out.append({
                    "n_patterns": n,
                    "count": len(vals),
                    "avg_net_r": round(sum(vals) / len(vals), 4) if vals else 0.0,
                })
            return out
        except Exception:
            return base

    # ------------------------------------------------------------------
    # Background MCP state refresh (keeps MCP network calls OFF the GUI thread)
    # ------------------------------------------------------------------

    def _ensure_mcp_refresh(self) -> None:
        """Start the background MCP-refresh daemon thread (idempotent)."""
        if self._mcp_refresh_thread is not None and self._mcp_refresh_thread.is_alive():
            return
        self._mcp_refresh_active = True
        self._mcp_refresh_thread = threading.Thread(
            target=self._mcp_refresh_loop, daemon=True, name="mcp-refresh")
        self._mcp_refresh_thread.start()

    def _mcp_refresh_loop(self) -> None:
        """Loop on a daemon thread: refresh state from MCP without blocking the GUI."""
        interval = float(os.environ.get("MCP_REFRESH_INTERVAL_S", "2") or 2)
        while self._mcp_refresh_active:
            try:
                self.refresh_state_from_mcp()
            except Exception:
                pass
            time.sleep(interval)

    @classmethod
    def get_instance(cls, config_path: str | None = None) -> SystemBridge:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Symbol config loader (from configs/symbols/*.yaml)
    # ------------------------------------------------------------------

    def _load_symbol_configs(self) -> None:
        """Load validated symbols from configs/symbols/*.yaml into the registry.

        Only runs for symbols NOT already in the registry (user may have
        removed or changed status via GUI). MCP-discovered symbols remain
        in the registry at their existing status (never downgraded by MCP poll).
        """
        configs_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "symbols"
        if not configs_dir.is_dir():
            return
        for yaml_file in sorted(configs_dir.glob("*.yaml")):
            if yaml_file.name == "_template.yaml":
                continue
            try:
                import yaml
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                sym_block = data.get("symbol", {})
                if isinstance(sym_block, dict):
                    name = sym_block.get("name", "")
                    status = sym_block.get("status", "pending")
                else:
                    name = data.get("symbol_name", "")
                    status = data.get("status", "pending")
                if not name:
                    continue
                active = data.get("symbol", {}).get("is_active", True) if isinstance(data.get("symbol"), dict) else data.get("is_active", True)
                # Only register validated symbols from YAML
                if status == "validated":
                    # Critical: only register if symbol is NOT already in registry
                    # (user may have removed or changed status via GUI at runtime)
                    cfg = self.state.get_symbol_config(name)
                    if cfg is None:
                        self.state.register_symbol(
                            name,
                            SymbolConfig(name=name, status="validated", active=bool(active)),
                        )
            except Exception:
                continue

    # ------------------------------------------------------------------
    # MCP Execution Layer
    # ------------------------------------------------------------------

    def _init_exec_layer(self) -> None:
        """Try to import and instantiate execution_layer_v2."""
        try:
            # Lazy import — may not exist if module is not built yet
            from live.engine.execution_layer_v2 import ExecutionLayer

            endpoint = "http://127.0.0.1:22346/mcp"
            token = self.state.mcp_token or os.environ.get("MCP_TOKEN", "")

            self._exec_layer = ExecutionLayer(
                endpoint=endpoint,
                token=token,
                auto_reconnect=True,
            )
        except ImportError:
            import traceback
            _log("_init_exec_layer", {"status": "import_failed", "error": traceback.format_exc()})
            self._exec_layer = None

    def connect_mcp(self, token: str = "") -> bool:
        """Connect to MCP with the given token (or current token).

        Returns True on success, False on failure.
        """
        if token:
            self.state.set_mcp_token(token)
            syslog("INFO", "MCP", "", "MCP token updated via connect_mcp")

        with self._mcp_lock:
            if self._exec_layer is None:
                syslog("INFO", "MCP", "", "Initializing MCP execution layer...")
                self._init_exec_layer()
                if self._exec_layer is None:
                    self.state.set_mcp_connection_state("disconnected")
                    syslog("ERROR", "MCP", "", "MCP execution layer initialization failed")
                    return False
            else:
                # Always sync the latest token from state into exec_layer
                latest_token = self.state.mcp_token or os.environ.get("MCP_TOKEN", "")
                if latest_token and self._exec_layer.token != latest_token:
                    self._exec_layer.set_token(latest_token)

            try:
                self.state.set_mcp_connection_state("reconnecting")
                syslog("INFO", "MCP", "", "Attempting MCP connection...",
                       extra={"endpoint": getattr(self._exec_layer, "endpoint", "unknown")})
                # Attempt health check
                health = self._exec_layer.health_check()
                if health.get("ok"):
                    self.state.set_mcp_connection_state("connected")
                    self.state.update_mt5_status(connected=True)

                    acct = self._exec_layer.get_account_info()
                    acct_type = acct.get("account_type", acct.get("type", "unknown"))
                    self.state.update_mt5_status(
                        connected=True,
                        account_type="demo" if "demo" in str(acct_type).lower() else "real",
                        account_info=acct,
                    )

                    syslog("INFO", "MCP", "", "MCP connected successfully",
                           extra={"endpoint": getattr(self._exec_layer, "endpoint", "unknown")})
                    _log("connect_mcp", {"status": "connected"})
                    return True
                else:
                    self.state.set_mcp_connection_state("disconnected")
                    self.state.update_mt5_status(connected=False)
                    syslog("WARNING", "MCP", "", "MCP health check failed",
                           extra={"health": health})
                    _log("connect_mcp", {"status": "health_check_failed", "health": health})
                    return False
            except Exception as e:
                self.state.set_mcp_connection_state("disconnected")
                self.state.update_mt5_status(connected=False)
                syslog("ERROR", "MCP", "", f"MCP connection error: {e}",
                       extra={"error": str(e)})
                _log("connect_mcp", {"status": "error", "error": str(e)})
                return False

    def disconnect_mcp(self) -> None:
        """Disconnect from MCP."""
        self.state.set_mcp_connection_state("disconnected")
        self.state.update_mt5_status(connected=False)
        syslog("INFO", "MCP", "", "MCP disconnected by user")
        _log("disconnect_mcp", {})

    def reconnect_mcp(self) -> bool:
        """Force reconnection with existing token and auto-reconnect logic."""
        syslog("WARNING", "MCP", "", "MCP reconnection triggered")
        return self.connect_mcp(self.state.mcp_token)

    def fetch_positions(self) -> list[dict[str, Any]]:
        """Fetch open positions from MCP. Reconnects on failure."""
        if self._exec_layer is None:
            return []
        try:
            return self._exec_layer.get_positions()
        except Exception:
            self.reconnect_mcp()
            return []

    def fetch_symbols(self) -> list[str]:
        """Fetch market watch symbols from MCP as a flat string list."""
        if self._exec_layer is None:
            return []
        try:
            raw = self._exec_layer.get_symbols()  # List[Dict] w/ 'symbol' key
            result: list[str] = []
            for s in raw:
                if isinstance(s, str):
                    result.append(s)
                elif isinstance(s, dict):
                    result.append(s.get("symbol", s.get("name", "")))
            return [r for r in result if r]
        except Exception:
            return []

    def _publish_price_cache(self, prices: dict[str, float]) -> None:
        """Store the latest bid-price map (called on the MCP refresh thread)."""
        with self._price_cache_lock:
            self._price_cache = dict(prices)

    def fetch_symbol_prices(self) -> dict[str, float]:
        """Return a snapshot of the cached bid-price map.

        Prices are refreshed on the background MCP daemon thread
        (``refresh_state_from_mcp`` → ``_fetch_prices_blocking``), so this
        method NEVER blocks the GUI thread on a slow/unreachable MCP server
        (a synchronous ``get_symbols()`` here froze the whole UI for ~2-5s).

        Returns ``{}`` when no prices have been cached yet (e.g. MCP not
        connected).  Callers render a placeholder instead of blocking.
        """
        with self._price_cache_lock:
            return dict(self._price_cache)

    def _fetch_prices_blocking(self) -> dict[str, float]:
        """Blockingly fetch bid prices from MCP — background-thread ONLY.

        Do NOT call this from the GUI thread: each MCP round-trip (and the
        client's 2s retry sleep on a failed session) would freeze the UI.
        The MCP refresh daemon thread calls it and publishes the result into
        :attr:`_price_cache`.
        """
        if self._exec_layer is None:
            return {}
        try:
            raw = self._exec_layer.get_symbols()
            price_map: dict[str, float] = {}
            # First pass: check if symbols dicts contain bid/ask
            all_have_prices = True
            for s in raw:
                if isinstance(s, dict):
                    sym_name = s.get("symbol", s.get("name", ""))
                    bid = s.get("bid")
                    if bid is not None:
                        price_map[sym_name] = float(bid)
                    else:
                        all_have_prices = False
                else:
                    all_have_prices = False
            # If any symbol is missing a bid, fall back to get_symbol_info per symbol
            if not all_have_prices:
                # Only query symbols from the current watch list
                symbols = list(price_map.keys()) or [s if isinstance(s, str) else s.get("symbol", s.get("name", "")) for s in raw]
                price_map.clear()
                for sym in symbols:
                    if not sym:
                        continue
                    try:
                        info = self._exec_layer.get_symbol_info(sym)
                        if isinstance(info, dict):
                            bid = info.get("bid", info.get("Bid", info.get("current", info.get("price"))))
                            if bid is not None:
                                price_map[sym] = float(bid)
                    except Exception:
                        continue
            return price_map
        except Exception:
            return {}

    def send_order(
        self, asset: str, direction: str,
        entry_price: float, stop_loss: float, take_profit: float,
        lot_size: float | None = None,
    ) -> dict[str, Any]:
        """Send a market order via MCP execution layer.

        HYBRID risk sizing: if ``lot_size`` is None (the default used by the
        manual GUI path, which has no lot-size input field), the lot is sized
        from the account risk budget by routing through
        ``ExecutionLayer.place_order`` (risk-based).  If an explicit
        ``lot_size`` is supplied by a caller/GUI, it is honored exactly via
        ``ExecutionLayer.send_order(volume=lot_size)``.

        Returns the execution layer's order-result dict.
        """
        if self._exec_layer is None:
            return {"ok": False, "error": "Execution layer not initialized"}

        try:
            order_direction = _normalize_direction(direction)
            with self._mcp_lock:
                if lot_size is None:
                    # Risk-sized manual entry (same compute_risk_lot path as the
                    # automated path).  This is what makes
                    # ExecutionLayer.place_order live instead of dead code.
                    result = self._exec_layer.place_order(
                        symbol=asset,
                        order_type=order_direction,
                        price=entry_price,
                        sl=stop_loss,
                        tp=take_profit,
                    )
                else:
                    # Explicit lot supplied by a caller/GUI -> honor it exactly.
                    result = self._exec_layer.send_order(
                        asset=asset,
                        direction=order_direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=lot_size,
                    )
            _log("send_order", {"asset": asset, "direction": direction,
                                "lot_size": lot_size, "result": str(result)})
            vol_desc = "risk-sized" if lot_size is None else f"{lot_size} lots"
            syslog("INFO", "Execution", asset,
                   f"Order sent via bridge: {direction.upper()} {vol_desc} @ {entry_price}",
                   extra={"direction": direction, "volume": lot_size,
                          "entry": entry_price, "sl": stop_loss, "tp": take_profit})
            return result
        except Exception as e:
            _log("send_order", {"asset": asset, "error": str(e)})
            syslog("ERROR", "Execution", asset,
                   f"Order failed via bridge: {e}",
                   extra={"direction": direction, "error": str(e)})
            return {"ok": False, "error": str(e)}

    def close_position(self, position_id: str, symbol: str = "") -> dict[str, Any]:
        """Close a specific position by ID.

        ``symbol`` is forwarded to the MCP close call as the required safety
        check (the server rejects an empty/mismatched symbol); the GUI resolves
        it from the open-position snapshot before calling this.
        """
        if self._exec_layer is None:
            return {"ok": False, "error": "Execution layer not initialized"}
        try:
            result = self._exec_layer.close_position(position_id, symbol=symbol)
            _log("close_position", {"position_id": position_id, "symbol": symbol, "result": str(result)})
            if result.get("success"):
                syslog("INFO", "Execution", "", f"Position {position_id} closed via bridge")
            else:
                syslog("WARNING", "Execution", "",
                       f"Close position {position_id} returned: {result.get('error', 'unknown')}")
            return result
        except Exception as e:
            syslog("ERROR", "Execution", "", f"Close position {position_id} failed: {e}",
                   extra={"position_id": position_id, "error": str(e)})
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Candle data + risk lot (pending-signal inspector)
    # ------------------------------------------------------------------

    def fetch_candles(self, symbol: str, limit: int = 50) -> Any:
        """Return the last ~``limit`` closed M15 candles for ``symbol``.

        Wraps the engine's ``MCPCandleSource(self._exec_layer).get_chart_history``
        adapter and returns the resulting OHLCV ``pandas.DataFrame``
        (DatetimeIndex, columns ``open/high/low/close/volume``) sliced to the
        last ``limit`` rows.  Returns an empty DataFrame (never raises) when the
        execution layer is unavailable or returns no candles, so the inspector
        chart renders a "No data" placeholder instead of failing.
        """
        import pandas as pd
        if self._exec_layer is None:
            return pd.DataFrame()
        try:
            from live.engine.signal_polling_engine_v2 import MCPCandleSource
            df = MCPCandleSource(self._exec_layer).get_chart_history(symbol)
        except Exception as exc:
            syslog("WARNING", "Execution", symbol, f"fetch_candles failed: {exc}")
            return pd.DataFrame()
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()
        return df.iloc[-max(1, int(limit)):]

    def compute_risk_lot(self, asset: str, entry_price: float,
                         stop_loss: float) -> float | None:
        """Return the risk-based lot for ``asset`` (or ``None`` when not computable).

        Delegates to ``ExecutionLayer.compute_risk_lot`` so the inspector can
        pre-fill its auto-lot spinbox with the same risk sizing (which derives
        from RiskGuard's ``position_size_multiplier``) the automated and
        risk-sized manual paths use.
        """
        if self._exec_layer is None:
            return None
        try:
            lot = self._exec_layer.compute_risk_lot(asset, entry_price, stop_loss)
            return float(lot) if lot is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Polling / Refresh
    # ------------------------------------------------------------------

    def refresh_state_from_mcp(self) -> dict[str, Any]:
        """Fetch latest data from MCP and update shared state.

        Returns the snapshot dict.
        """
        if self._exec_layer is None:
            return self.state.get_snapshot()

        try:
            # Health check
            health = self._exec_layer.health_check()
            if health.get("ok"):
                was_disconnected = self.state.mcp_connection_state != "connected"
                self.state.set_mcp_connection_state("connected")
                self.state.update_mt5_status(connected=True)

                if was_disconnected:
                    syslog("INFO", "MCP", "", "MCP reconnected during polling refresh",
                           extra={"endpoint": getattr(self._exec_layer, "endpoint", "unknown")})

                # Account info
                acct = self._exec_layer.get_account_info()
                acct_type = acct.get("account_type", acct.get("type", "unknown"))
                self.state.update_mt5_status(
                    connected=True,
                    account_type="demo" if "demo" in str(acct_type).lower() else "real",
                    account_info=acct,
                )

                # Positions
                # get_positions() ALREADY publishes OpenPosition objects into
                # state internally (set_open_positions). Do NOT re-store the raw
                # dict list it returns — that clobbered the objects with dicts
                # and made get_snapshot() crash on p.asset (AttributeError).
                self._exec_layer.get_positions()

                # Symbols — only register if not already in registry,
                # preserving existing status (e.g. validated from YAML config)
                symbols = self._exec_layer.get_symbols()
                for s in symbols:
                    name = s if isinstance(s, str) else s.get("name", "")
                    if name and name not in self.state.symbol_registry:
                        self.state.register_symbol(name, SymbolConfig(name=name, status="pending", active=False))

                # Refresh the cached bid-price map so the GUI can read prices
                # instantly without blocking the main thread on MCP.
                if self._exec_layer is not None:
                    prices = self._fetch_prices_blocking()
                    if prices:
                        self._publish_price_cache(prices)
            else:
                self.state.set_mcp_connection_state("disconnected")
                self.state.update_mt5_status(connected=False)

        except Exception:
            self.state.set_mcp_connection_state("disconnected")
            self.state.update_mt5_status(connected=False)

        return self.state.get_snapshot()

    # ------------------------------------------------------------------
    # Signal engine poll (GUI timer hook)
    # ------------------------------------------------------------------

    def poll_once(self) -> None:
        """GUI timer hook — run one signal-engine poll cycle.

        Called by the main window on every refresh tick:

        * keeps the background SignalPollingEngine running (started on
          demand, restarted after an emergency stop is released);
        * requests an immediate scan.

        Symbol registration/refresh happens on the engine's own thread every
        cycle (symbols are picked up / dropped the moment the user validates /
        activates / deactivates them).  All scanning *and* model loading
        therefore execute on the engine daemon thread — this method never
        blocks the GUI thread on MCP calls, detection or artifact loading.
        An active emergency stop wakes the engine so it halts cleanly instead
        of scanning.
        """
        engine = self._signal_engine
        if engine is None:
            return
        # Ensure the background MCP refresh thread is running (idempotent) so
        # state stays current WITHOUT blocking this GUI-timer call.
        self._ensure_mcp_refresh()
        try:
            if self.state.emergency_stop:
                # Wake the loop so it observes the emergency stop and halts.
                engine.request_scan()
                return
            if not engine.is_running:
                engine.start()
                syslog("INFO", "System", "", "Signal engine polling started by GUI timer")
            engine.request_scan()
        except Exception as exc:  # polling must never break the GUI refresh
            syslog("WARNING", "System", "", f"Signal engine poll failed: {exc}")

    # ------------------------------------------------------------------
    # Signal / Risk / Kill-Switch
    # ------------------------------------------------------------------

    def check_kill_switch(self, asset: str, trades: list[dict]) -> dict[str, Any]:
        """Evaluate kill-switch for asset using block bootstrap via risk_guard_v2.

        Returns dict with keys: active, reason, pf_ci_lower.
        """
        try:
            from live.state.risk_guard_v2 import RiskGuard
            guard = RiskGuard()
            result = guard.evaluate(asset, trades)
            return result
        except ImportError:
            return {"active": False, "reason": "RiskGuard not available", "pf_ci_lower": 0.0}

    def log_action(self, action_type: str, details: dict | None = None) -> None:
        """Log a GUI user action to the database."""
        _log(action_type, details or {})

    # ------------------------------------------------------------------
    # Model Registry API
    # ------------------------------------------------------------------
    # These methods provide the Model Registry integration required by
    # GUI_REWORK_REQUIREMENTS.md Sections 2 and 3.

    def get_available_models(self) -> list[ModelInfo]:
        """Return all registered models from the Model Registry.

        Returns:
            Sorted list of ModelInfo dataclasses (empty if registry not loaded).
        """
        return self._model_registry.get_available_models()

    def get_models_for_assignment(
        self,
        pattern_name: str = "",
        feature_schema_version: str = "",
        lifecycle_states: set[str] | None = None,
        symbol: str = "",
    ) -> list[ModelInfo]:
        """Return models eligible for a §10.1 pattern assignment dropdown.

        Filters by ``pattern_name`` + ``feature_schema_version`` +
        ``lifecycle_state ∈ {validated, shadow, live}`` (spec §10.1) so the
        GUI only offers assignable models for the selected pattern.
        """
        return self._model_registry.get_models_for_assignment(
            pattern_name=pattern_name,
            feature_schema_version=feature_schema_version,
            lifecycle_states=lifecycle_states,
            symbol=symbol,
        )

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Look up a single model by its model_id.

        Returns:
            ModelInfo if found, None otherwise.
        """
        return self._model_registry.get_model(model_id)

    def assign_model_to_symbol(self, symbol: str, model_id: str) -> bool:
        """Assign a model to an already-registered symbol.

        The symbol must already exist in the registry.  This changes only
        the ``model_id`` field — it does **not** alter status or active flag.

        Args:
            symbol: Symbol name (e.g. "XAUUSD").
            model_id: Valid model_id from the Model Registry.

        Returns:
            True on success, False if symbol or model_id is invalid.
        """
        if not self._model_registry.get_model(model_id):
            _log("assign_model_to_symbol", {"symbol": symbol, "model_id": model_id, "error": "unknown model"})
            syslog("WARNING", "System", symbol,
                   f"Assign model failed: '{model_id}' not found in registry")
            return False

        cfg = self.state.get_symbol_config(symbol)
        if cfg is None:
            _log("assign_model_to_symbol", {"symbol": symbol, "model_id": model_id, "error": "unknown symbol"})
            syslog("WARNING", "System", symbol,
                   f"Assign model failed: symbol '{symbol}' not in registry")
            return False

        old_model = cfg.model_id
        self.state.register_symbol(
            symbol,
            SymbolConfig(
                name=cfg.name,
                status=cfg.status,
                active=cfg.active,
                model_id=model_id,
                position_size_multiplier=cfg.position_size_multiplier,
            ),
        )
        self.state.save_registry()
        _log("assign_model_to_symbol", {"symbol": symbol, "model_id": model_id})
        syslog("INFO", "System", symbol,
               f"Model assigned: {old_model or 'none'} → {model_id}",
               extra={"old_model": old_model, "new_model": model_id})
        return True

    def add_symbol_with_model(
        self,
        symbol: str,
        model_id: str,
        activate: bool = True,
    ) -> bool:
        """Register a new symbol and assign a model in one call.

        Creates a ``SymbolConfig(status="validated", active=activate, model_id=...)``
        and registers it in the shared state.  The symbol must not already
        be registered.

        Args:
            symbol: Symbol name (e.g. "XAUUSD").
            model_id: Valid model_id from the Model Registry.
            activate: Whether to set ``active=True`` (default True).

        Returns:
            True on success, False if already registered or model unknown.
        """
        if not self._model_registry.get_model(model_id):
            _log("add_symbol_with_model", {"symbol": symbol, "model_id": model_id, "error": "unknown model"})
            syslog("WARNING", "System", symbol,
                   f"Add symbol failed: model '{model_id}' not found in registry")
            return False

        if self.state.get_symbol_config(symbol) is not None:
            _log("add_symbol_with_model", {"symbol": symbol, "model_id": model_id, "error": "already registered"})
            syslog("WARNING", "System", symbol,
                   "Add symbol failed: already registered")
            return False

        config = SymbolConfig(
            name=symbol,
            status="validated",
            active=activate,
            model_id=model_id,
        )
        self.state.register_symbol(symbol, config)
        self.state.save_registry()
        _log("add_symbol_with_model", {"symbol": symbol, "model_id": model_id, "activate": activate})
        syslog("INFO", "System", symbol,
               f"Symbol added with model '{model_id}' (active={activate})",
               extra={"model_id": model_id, "activate": activate})
        return True

    def reload_model_registry(self) -> int:
        """Reload the Model Registry from disk (re-reads YAML).

        Returns:
            Number of models loaded.
        """
        n = self._model_registry.reload()
        _log("reload_model_registry", {"models_loaded": n})
        syslog("INFO", "System", "", f"Model registry reloaded: {n} model(s) loaded",
               extra={"models_loaded": n})
        return n

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown: stop the signal engine, close MCP, stop polling."""
        self._running = False
        self._mcp_refresh_active = False
        if self._signal_engine is not None:
            try:
                self._signal_engine.stop()
                syslog("INFO", "System", "", "Signal engine polling stopped during shutdown")
            except Exception:
                pass
        if self._exec_layer is not None:
            try:
                stop_polling = getattr(self._exec_layer, "stop_polling", None)
                if stop_polling is not None:
                    stop_polling()
            except Exception:
                pass
            try:
                self._exec_layer.close()
                syslog("INFO", "MCP", "", "MCP execution layer closed during shutdown")
            except Exception:
                pass
        self.state.set_mcp_connection_state("disconnected")
        _log("shutdown", {})
        syslog("INFO", "System", "", "SystemBridge shutdown complete")


def _log(action_type: str, details: dict[str, Any]) -> None:
    """Internal helper to log a user action with JSON details."""
    try:
        details_json = json.dumps(details, default=str) if details else None
        log_user_action(action_type, details_json)
    except Exception:
        pass  # Silently fail — logging should never crash the GUI


# Module-level convenience accessor
_global_bridge: SystemBridge | None = None


def _group_size_from_last(gid: str, events: Any) -> int:
    """Count events sharing one confluence_group_id in a last-events list."""
    try:
        n = sum(1 for e in (events or []) if getattr(e, "confluence_group_id", None) == gid)
        return max(n, 2)
    except Exception:
        return 2


def get_bridge(config_path: str | None = None) -> SystemBridge:
    """Get the global SystemBridge singleton."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = SystemBridge.get_instance(config_path)
    return _global_bridge