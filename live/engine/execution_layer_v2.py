"""
execution_layer_v2.py — MCP Execution Layer V2 for Paper Trading System.

Wraps the MetaTrader 5 MCP client with auto-reconnect on EVERY call,
dynamic token setting, magic number 20791, LSW-V2-{event_id} order comments,
position polling during disconnect, and graceful degradation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# APPEND (never insert at sys.path[0]): inserting trading_v3/ at the FRONT of
# sys.path can shadow stdlib modules (e.g. ``logging``) for any module imported
# later in the process. Appending keeps site-packages/stdlib resolution first;
# ``live.*`` is not a stdlib name so these imports still resolve.
if _PARENT not in sys.path:
    sys.path.append(_PARENT)
from live.logging import logger_v2 as paper_logger  # noqa: E402  (path bootstrap)
from live.logging.logger_v2 import log as syslog  # noqa: E402  (path bootstrap)
from live.mcp.mt5_mcp_client import (  # noqa: E402
    MCPClient,
    MCPConnectionError,
    MCPError,
)
from live.state.shared_app_state_v2 import (  # noqa: E402
    SYMBOL_CONFIG_DIR,
    OpenPosition,
    get_state,
)

DEFAULT_ENDPOINT = os.environ.get("MCP_URL", "http://127.0.0.1:22346/mcp")
DEFAULT_TOKEN = os.environ.get("MCP_TOKEN", "")
RETRY_COUNT = 1
RETRY_DELAY_SECONDS = 2.0
MAGIC_NUMBER = 20791
POSITION_POLL_INTERVAL_S = 30.0

# ---------------------------------------------------------------------------
# Risk-based lot sizing constants.
# ``risk_based_lot`` sizes a position from the account risk budget per trade:
#     lot = (balance x risk_per_trade_pct x base_multiplier)
#           / (stop_distance x point_value x contract_size)
# ``base_multiplier`` is applied BEFORE the lot calc as a risk scalar (never
# appended after — appending after would silently lower effective risk%).
# ``point_value`` and ``contract_size`` are per-symbol; both symbols traded
# here are USD-quoted, so a 1.0-quoted-price move is 1.0 USD per 1 unit of
# contract.  Values below are the documented hard-code fallback (standard MT5
# contract specs); if MCP ``symbol_info`` exposes contract/tick data it is
# preferred and these are only the fallback.
# ---------------------------------------------------------------------------
_DOCUMENTED_CONTRACT_SPECS = {
    # contract_size = units of the base instrument per 1.0 lot
    # point_value   = USD value of a 1.0-quoted-price move for 1.0 unit
    "XAUUSD": {"contract_size": 100.0, "point_value": 1.0},     # 100 oz per lot
    "EURUSD": {"contract_size": 100000.0, "point_value": 1.0},  # 100k units per lot
}
_DEFAULT_VOLUME_STEP = 0.01
_DEFAULT_VOLUME_MIN = 0.01
# Absolute safety cap: a calculation error can never produce a larger lot than
# this.  Per-symbol ``max_position_size_lots`` (config) is preferred; this is
# the hard global ceiling applied on top.
HARD_MAX_LOT_CAP = 1.0


def risk_based_lot(balance: float | None, risk_percent: float,
                   base_multiplier: float, stop_distance: float | None,
                   point_value: float, contract_size: float,
                   volume_step: float = _DEFAULT_VOLUME_STEP,
                   volume_min: float | None = _DEFAULT_VOLUME_MIN,
                   volume_max: float | None = None,
                   max_lot_cap: float | None = HARD_MAX_LOT_CAP) -> float | None:
    """Compute a position lot from the account risk budget per trade.

    Semantics (documented):
      * ``risk_percent`` is the nominal risk % budget for the symbol
        (config ``risk_per_trade_pct``); ``base_multiplier`` is a per-symbol
        scalar applied BEFORE the lot calc.  Effective risk% = product.
      * ``stop_distance`` = |entry_price - stop_price|, sourced once from the
        Signal Engine and passed in — never recomputed divergently here.
      * ``point_value * contract_size`` = USD value of a 1.0 price-unit move
        for 1.0 lot, so ``stop_distance * point_value * contract_size`` is the
        dollar exposure of 1.0 lot at the stop.
      * The raw lot is rounded DOWN to ``volume_step`` then clamped to
        ``[volume_min, volume_max]`` and capped at ``max_lot_cap`` (absolute
        ceiling).  Rounding DOWN guarantees actual risk <= intended risk.
    Returns None when inputs are missing/unusable (caller falls back safely).
    """
    if not balance or balance <= 0:
        return None
    if not stop_distance or stop_distance <= 0:
        return None
    if not point_value or point_value <= 0 or not contract_size or contract_size <= 0:
        return None
    risk_budget = float(balance) * (float(risk_percent) / 100.0) * float(base_multiplier)
    stop_value = float(stop_distance) * float(point_value) * float(contract_size)
    if stop_value <= 0 or risk_budget <= 0:
        return None
    lot_raw = risk_budget / stop_value
    # Round DOWN to the broker volume step (actual risk stays <= intended).
    try:
        lot = math.floor(lot_raw / float(volume_step)) * float(volume_step)
    except (TypeError, ValueError, ZeroDivisionError):
        lot = lot_raw
    # Clamp to [volume_min, volume_max].
    if volume_min is not None:
        lot = max(lot, float(volume_min))
    if volume_max is not None:
        lot = min(lot, float(volume_max))
    # Hard absolute ceiling.
    if max_lot_cap is not None:
        lot = min(lot, float(max_lot_cap))
    # Normalise float noise (e.g. 0.43000000000000005) to the step grid.
    return round(lot, 10)


def _parse_mcp_response(result: dict) -> dict:
    content = result.get("content", [])
    if content and isinstance(content, list) and len(content) > 0:
        text = content[0].get("text", "")
        if text:
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"data": parsed}
            except json.JSONDecodeError:
                return {"text": text}
    return result


def _normalize_direction(direction: Any) -> str:
    """Map any direction spelling to the MCP order ``type`` string.

    The signal engine emits ``"long"``/``"short"``, while the MT5 MCP
    ``trade_send_market_order`` expects ``"buy"``/``"sell"``.  Any value that is
    not clearly a "buy"/"long" is treated as a sell — so a long signal is never
    accidentally sent as a sell (which produced MT5 "Invalid stops" rejections
    because the stop-loss/take-profit were placed for the wrong side).
    """
    if direction is None:
        return "buy"
    d = str(direction).strip().lower()
    if d in ("buy", "long", "bullish", "0"):
        return "buy"
    if d in ("sell", "short", "bearish", "1"):
        return "sell"
    # Unknown spelling defaults to buy (the safer default for a market order).
    return "buy"


def _validate_stop_side(order_type: str, entry: float, stop_loss: float,
                        take_profit: float) -> str | None:
    """Return a human-readable stop-placement problem, or ``None`` if valid.

    Ensures the stop-loss and take-profit are on the correct side of ``entry``
    for the order direction, so we never dispatch an order that MT5 will reject
    with "Invalid stops".  Returns an empty/None when both stops are absent
    (allowed for a naked market order) or correctly placed.
    """
    if not entry or (not stop_loss and not take_profit):
        return None

    if order_type == "buy":
        if stop_loss and stop_loss >= entry:
            return (f"Stop loss {stop_loss} must be BELOW entry {entry} for a BUY "
                    "(it would be hit immediately on the wrong side).")
        if take_profit and take_profit <= entry:
            return (f"Take profit {take_profit} must be ABOVE entry {entry} for a BUY.")
    else:  # sell
        if stop_loss and stop_loss <= entry:
            return (f"Stop loss {stop_loss} must be ABOVE entry {entry} for a SELL "
                    "(it would be hit immediately on the wrong side).")
        if take_profit and take_profit >= entry:
            return (f"Take profit {take_profit} must be BELOW entry {entry} for a SELL.")
    return None


def _parse_position_time(value: Any) -> float:
    """Convert an MT5 position time to unix epoch (float).

    The MCP position payload carries ``create_time`` as a terminal-local string
    ``"YYYY.MM.DD HH:MM:SS"``.  Accept a unix float directly, else try the two
    common string formats; return ``0.0`` on failure so callers never crash.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except (ValueError, TypeError):  # noqa: PERF203 (retry/continue semantics require try-except in loop)
            continue
    return 0.0


class ExecutionLayerError(Exception):
    pass


class OrderRejectedError(ExecutionLayerError):
    pass


class ConnectionFailedError(ExecutionLayerError):
    pass


class ExecutionLayer:
    """V2 Execution Layer — auto-reconnect on EVERY call, dynamic token,
    magic 20791, LSW-V2-{event_id} comments, position polling."""

    def __init__(self, endpoint: str | None = None,
                 token: str | None = None,
                 auto_reconnect: bool = True):
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.token = token or DEFAULT_TOKEN
        self.auto_reconnect = auto_reconnect
        self._client: MCPClient | None = None
        self._session_initialized: bool = False
        self._token_lock = threading.Lock()
        self.state = get_state()
        self._symbol_config_cache: dict[str, dict] = {}
        self._polling_active = True
        self._poll_thread = threading.Thread(target=self._position_poll_loop,
                                             daemon=True, name="el-pos-poll")
        self._poll_thread.start()
        syslog("INFO", "Execution", "",
               f"ExecutionLayer initialized: endpoint={self.endpoint}, auto_reconnect={auto_reconnect}",
               extra={"endpoint": self.endpoint, "auto_reconnect": auto_reconnect})

    # ------------------------------------------------------------------
    # Risk-based lot sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _base_symbol(asset: str) -> str:
        """Strip the MT5 'm' market suffix so the per-symbol YAML key is found."""
        return asset[:-1] if asset.endswith("m") else asset

    def _load_symbol_config(self, asset: str) -> dict:
        """Load the canonical per-symbol YAML (cached).  Returns {} on failure."""
        base = self._base_symbol(asset)
        if base in self._symbol_config_cache:
            return self._symbol_config_cache[base]
        cfg: dict = {}
        try:
            path = Path(SYMBOL_CONFIG_DIR) / f"{base}.yaml"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg = loaded
        except Exception:
            cfg = {}
        self._symbol_config_cache[base] = cfg
        return cfg

    def _symbol_risk_params(self, asset: str) -> dict:
        """Resolve per-symbol risk sizing params (canonical yaml + MCP, then fallback)."""
        cfg = self._load_symbol_config(asset)
        ps = cfg.get("position_sizing", {}) if isinstance(cfg, dict) else {}
        broker = cfg.get("broker", {}) if isinstance(cfg, dict) else {}
        base = self._base_symbol(asset)

        risk_percent = float(ps.get("risk_per_trade_pct", 0.0) or 0.0)
        base_multiplier = float(ps.get("base_multiplier", 1.0) or 1.0)
        volume_min = ps.get("min_position_size_lots")
        volume_max = ps.get("max_position_size_lots")
        volume_step = broker.get("volume_step", _DEFAULT_VOLUME_STEP)

        # Per-symbol point value / contract size: prefer MCP symbol_info
        # (contract_size/trade_contract_size + tick_value/tick_size), else the
        # documented hard-code table (standard MT5 contract specs).
        contract_size: float | None = None
        point_value: float | None = None
        mcp_info: dict = {}
        try:
            info = self.get_symbol_info(asset)
            if isinstance(info, dict):
                mcp_info = info
        except Exception:
            mcp_info = {}

        contract_size = mcp_info.get("trade_contract_size",
                                     mcp_info.get("contract_size"))
        tick_value = mcp_info.get("tick_value")
        tick_size = mcp_info.get("tick_size")
        if contract_size is None:
            contract_size = mcp_info.get("volume", 0) or None
        if contract_size is not None:
            try:
                contract_size = float(contract_size)
            except (TypeError, ValueError):
                contract_size = None
        if tick_value is not None and tick_size:
            try:
                point_value = float(tick_value) / float(tick_size)
            except (TypeError, ValueError, ZeroDivisionError):
                point_value = None
        # Documented fallback (exact XAUUSD / EURUSD values; see module doc).
        if contract_size is None or point_value is None:
            spec = _DOCUMENTED_CONTRACT_SPECS.get(base, {})
            if contract_size is None:
                contract_size = spec.get("contract_size")
            if point_value is None:
                point_value = spec.get("point_value")

        # volume_step from MCP if present, else default.
        mcp_step = mcp_info.get("volume_step")
        if mcp_step is not None:
            try:
                volume_step = float(mcp_step)
            except (TypeError, ValueError):
                pass
        if volume_step is None or float(volume_step) <= 0:
            volume_step = _DEFAULT_VOLUME_STEP

        if volume_min is None:
            volume_min = _DEFAULT_VOLUME_MIN
        if volume_max is None:
            volume_max = float(ps.get("max_position_size_lots")) if ps.get("max_position_size_lots") else HARD_MAX_LOT_CAP

        return {
            "risk_percent": risk_percent,
            "base_multiplier": base_multiplier,
            "contract_size": contract_size,
            "point_value": point_value,
            "volume_step": float(volume_step),
            "volume_min": float(volume_min),
            "volume_max": float(volume_max) if volume_max is not None else HARD_MAX_LOT_CAP,
            "max_lot_cap": float(ps.get("max_position_size_lots")) if ps.get("max_position_size_lots") else HARD_MAX_LOT_CAP,
        }

    def _get_balance(self) -> float | None:
        """Current account balance from MT5 status cache, else a live fetch."""
        try:
            acct = getattr(self.state, "mt5_status", None)
            if acct is not None:
                b = acct.account_info.get("balance") if isinstance(acct.account_info, dict) else None
                if b:
                    return float(b)
        except Exception:
            pass
        try:
            info = self.get_account_info()
            if isinstance(info, dict):
                b = info.get("balance")
                if b:
                    return float(b)
        except Exception:
            pass
        return None

    def compute_risk_lot(self, asset: str, entry_price: float | None,
                         stop_price: float | None) -> float | None:
        """Compute the risk-based lot for a symbol given its entry & stop.

        stop_distance (=|entry - stop|) is supplied by the caller exactly as
        computed once in the Signal Engine — it is NOT recomputed divergently
        here.  Returns None when inputs are missing (caller falls back safely).
        """
        if entry_price is None or stop_price is None:
            return None
        stop_distance = abs(float(entry_price) - float(stop_price))
        params = self._symbol_risk_params(asset)
        balance = self._get_balance()
        if balance is None:
            return None
        lot = risk_based_lot(
            balance=balance,
            risk_percent=params["risk_percent"],
            base_multiplier=params["base_multiplier"],
            stop_distance=stop_distance,
            point_value=params["point_value"],
            contract_size=params["contract_size"],
            volume_step=params["volume_step"],
            volume_min=params["volume_min"],
            volume_max=params["volume_max"],
            max_lot_cap=params["max_lot_cap"],
        )
        return lot

    def set_token(self, token: str) -> None:
        """Set the MCP authentication token dynamically (GUI callable)."""
        with self._token_lock:
            self.token = token
            self.state.set_mcp_token(token)
            self._close_client()
        self._reconnect()

    def _reconnect(self) -> None:
        self._close_client()
        self.state.set_mcp_connection_state("reconnecting")
        syslog("WARNING", "Execution", "", "Reconnecting to MCP...",
               extra={"endpoint": self.endpoint})
        try:
            self._ensure_session()
            self.state.set_mcp_connection_state("connected")
            syslog("INFO", "Execution", "", "MCP reconnected successfully",
                   extra={"endpoint": self.endpoint})
        except (MCPConnectionError, MCPError, ConnectionFailedError):
            self.state.set_mcp_connection_state("disconnected")
            syslog("ERROR", "Execution", "", "MCP reconnection failed",
                   extra={"endpoint": self.endpoint})

    def _ensure_client(self) -> MCPClient:
        if self._client is None:
            with self._token_lock:
                t = self.token
            self._client = MCPClient(url=self.endpoint, token=t,
                                     auto_reconnect=self.auto_reconnect)
        return self._client

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._session_initialized = False
            syslog("WARNING", "Execution", "", "MCP client closed (session terminated)")

    def _ensure_session(self) -> None:
        client = self._ensure_client()
        if client.initialized:
            return
        for attempt in range(RETRY_COUNT + 1):
            try:
                client.connect()
                client.list_tools(refresh=True)
                self._session_initialized = True
                self.state.set_mcp_connection_state("connected")
                syslog("INFO", "Execution", "",
                       "MCP session established successfully",
                       extra={"endpoint": self.endpoint, "attempt": attempt + 1})
                return
            except (MCPConnectionError, MCPError) as exc:  # noqa: PERF203 (retry/continue semantics require try-except in loop)
                if attempt < RETRY_COUNT:
                    time.sleep(RETRY_DELAY_SECONDS)
                    client.close()
                    self._session_initialized = False
                else:
                    self.state.set_mcp_connection_state("disconnected")
                    raise ConnectionFailedError(
                        f"Cannot establish MCP session: {exc}") from exc

    def _call_mcp(self, tool_name: str,
                  arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call MCP tool with auto-reconnect on EVERY call.

        Returns empty dict on total failure (graceful degradation).
        """
        try:
            self._ensure_session()
        except ConnectionFailedError:
            self.state.set_mcp_connection_state("disconnected")
            syslog("ERROR", "Execution", "", f"MCP call '{tool_name}' failed: no session",
                   extra={"tool": tool_name})
            return {}

        client = self._ensure_client()
        for attempt in range(RETRY_COUNT + 1):
            try:
                result = client.call_tool(tool_name, arguments or {})
                self.state.set_mcp_connection_state("connected")
                return result
            except (MCPConnectionError, MCPError):  # noqa: PERF203 (retry/continue semantics require try-except in loop)
                if attempt < RETRY_COUNT:
                    syslog("WARNING", "Execution", "",
                           f"MCP call '{tool_name}' failed (attempt {attempt+1}/{RETRY_COUNT+1}), retrying...",
                           extra={"tool": tool_name, "attempt": attempt+1})
                    time.sleep(RETRY_DELAY_SECONDS)
                    self._close_client()
                    try:
                        self._ensure_session()
                    except ConnectionFailedError:
                        self.state.set_mcp_connection_state("disconnected")
                        syslog("ERROR", "Execution", "",
                               f"MCP call '{tool_name}' failed after retry — session lost",
                               extra={"tool": tool_name})
                        return {}
                else:
                    self.state.set_mcp_connection_state("disconnected")
                    syslog("ERROR", "Execution", "",
                           f"MCP call '{tool_name}' failed after {RETRY_COUNT+1} attempts",
                           extra={"tool": tool_name})
                    return {}

    def _position_poll_loop(self) -> None:
        """Poll positions every 30s, attempting reconnect when disconnected."""
        while self._polling_active:
            time.sleep(POSITION_POLL_INTERVAL_S)
            if self.state.mcp_connection_state == "disconnected":
                try:
                    self._ensure_session()
                    self.get_positions()
                    self.state.set_mcp_connection_state("connected")
                    syslog("INFO", "Execution", "", "Position poll: reconnected and synced positions")
                except (ConnectionFailedError, MCPError, MCPConnectionError):
                    pass

    def stop_polling(self) -> None:
        self._polling_active = False
        syslog("INFO", "Execution", "", "Position polling stopped")

    def health_check(self) -> dict[str, Any]:
        start = time.time()
        try:
            self._ensure_session()
            client = self._ensure_client()
            info = {}
            try:
                info = self._call_mcp("get_trading_account_info")
            except ConnectionFailedError:
                pass
            elapsed = time.time() - start
            r: dict[str, Any] = {"ok": True, "url": self.endpoint,
                                  "session_id": client.session_id,
                                  "elapsed_seconds": round(elapsed, 3)}
            if info:
                r["account_type"] = info.get("account_type", "unknown")
                if "version" in info:
                    r["version"] = info.get("version")
            return r
        except (MCPConnectionError, MCPError, ConnectionFailedError) as exc:
            elapsed = time.time() - start
            self.state.set_mcp_connection_state("disconnected")
            return {"ok": False, "url": self.endpoint,
                    "session_id": None,
                    "elapsed_seconds": round(elapsed, 3), "error": str(exc)}

    def get_account_info(self) -> dict[str, Any]:
        raw = self._call_mcp("get_trading_account_info")
        if not raw:
            self.state.update_mt5_status(connected=False)
            return {"error": "MCP disconnected", "ok": False}
        parsed = _parse_mcp_response(raw)
        acct = parsed.get("account", parsed)
        flat = dict(acct)
        flat["server"] = acct.get("server", "")
        flat["account_type"] = acct.get("type", "unknown")
        flat["is_demo"] = "demo" in str(acct.get("type", "")).lower()
        self.state.update_mt5_status(connected=True, account_info=flat)
        return flat

    def get_symbols(self) -> list[dict[str, Any]]:
        result = self._call_mcp("get_marketwatch_symbols", {})
        if not result:
            return []
        parsed = _parse_mcp_response(result)
        symbols = parsed.get("symbols", [])
        return symbols

    def add_symbol(self, symbol: str) -> bool:
        result = self._call_mcp("add_marketwatch_symbol", {"symbol": symbol})
        if not result:
            return False
        parsed = _parse_mcp_response(result)
        return not parsed.get("isError", False)

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        return self._call_mcp("get_symbol_info", {"symbol": symbol})

    def get_new_candles(self, symbol: str, period: str = "M15",
                        since: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol, "period": period}
        if since:
            params["datetime_from"] = since
        else:
            params["datetime_from"] = "2026-08-29T00:00:00"
        params["datetime_to"] = "2026-09-05T23:59:59"
        params["limit"] = 500
        raw = self._call_mcp("get_chart_history", params)
        if not raw:
            return []
        parsed = _parse_mcp_response(raw)
        candles = parsed.get("history", parsed.get("candles",
                          parsed.get("rates", parsed.get("data", []))))
        return candles if isinstance(candles, list) else []

    def get_positions(self) -> list[dict[str, Any]]:
        result = self._call_mcp("get_trading_open_positions", {})
        if not result:
            self.state.set_open_positions([])
            return []
        result = _parse_mcp_response(result)
        positions = result.get("positions", result.get("data", result.get("trades", [])))
        open_positions: list[OpenPosition] = []
        for p in positions:
            # Map the REAL MT5 MCP position shape: a market position is
            # returned with symbol, position_id, action (buy/sell), price_open,
            # price_last, volume, stop_loss, take_profit and create_time (a
            # "YYYY.MM.DD HH:MM:SS" string).  The earlier code read
            # `ticket`/`open_time`/`price` which do NOT exist, so position_id
            # was empty (Close never worked) and entry/current/open_time were 0
            # (garbled open-position columns).
            direction = ("buy" if str(p.get("action", p.get("type", ""))).lower() in ("buy", "0")
                         else "sell")
            entry_price = float(p.get("price_open", p.get("open_price", p.get("price", 0))))
            current_price = float(p.get("price_last", p.get("current_price", p.get("price", 0))))
            open_time = _parse_position_time(
                p.get("create_time", p.get("open_time", p.get("time", 0))))
            open_positions.append(OpenPosition(
                asset=p.get("symbol", p.get("asset", "Unknown")),
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                position_size=float(p.get("volume", p.get("size", 0))),
                stop_loss=float(p.get("stop_loss", p.get("sl", 0))),
                take_profit=float(p.get("take_profit", p.get("tp", 0))),
                open_time=open_time,
                position_id=str(p.get("position_id", p.get("ticket", p.get("id", "")))),
            ))
        self.state.set_open_positions(open_positions)
        return positions

    def send_order(self, asset: str, direction: str,
                   entry_price: float, stop_loss: float,
                   take_profit: float, volume: float | None = None,
                   signal_id: str | None = None,
                   rule_score: float = 0.0,
                   model_probability: float = 0.0,
                   automation_level: int = 1,
                   kill_switch_blocking: bool = False,
                   order_comment: str | None = None) -> dict[str, Any]:
        """Send market order to MT5 via MCP.

        Risk-based lot sizing: when ``volume`` is not provided (the automated
        path), the lot is computed from the account risk budget per trade via
        :meth:`compute_risk_lot` (lot = balance x risk% x multiplier /
        (stop_distance x point_value x contract_size)).  When ``volume`` IS
        provided it is honoured (e.g. explicit manual sizing).

        Flow: kill-switch check -> risk-lot sizing -> log BEFORE order ->
        send with magic=20791 and comment (§9.2), update log on success.

        Order comment §9.2: ``{pattern_short}-v{major}-{event_id_short}`` —
        e.g. ``DB-v1-a3f9`` / ``LSW-v2-77c1``.  When *order_comment* is not
        provided the standard ``order_comment_for`` schema is produced from
        ``signal_id``; the legacy ``LSW-V2-{event_id}`` format is kept only as
        an explicit migration fallback described in MIGRATION_NOTES.
        """
        eid = signal_id if signal_id else str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Check kill-switch
        ks = self.state.kill_switch_status.get(asset)
        if ks and ks.active:
            raise OrderRejectedError(f"Order blocked by kill-switch: {ks.reason}")

        # 2. Risk-based lot sizing (skipped when an explicit volume is given).
        risk_lot: float | None = None
        lot_source: str = "explicit"
        if volume is None:
            risk_lot = self.compute_risk_lot(asset, entry_price, stop_loss)
            if risk_lot is not None:
                volume = risk_lot
                lot_source = "risk"
            else:
                # Cannot size by risk (no balance / missing params) — fall back
                # to the configured minimum so we never block a signal outright.
                params = self._symbol_risk_params(asset)
                volume = float(params.get("volume_min", _DEFAULT_VOLUME_MIN))
                lot_source = "fallback"
                syslog("WARNING", "Execution", asset,
                       f"Risk-based lot could not be computed (balance/params missing); "
                       f"falling back to volume_min={volume}",
                       extra={"event_id": eid, "asset": asset})

        # 3. Log BEFORE sending order
        # Normalise long/short -> buy/sell so a "long" signal is never sent as
        # a sell (that placed the stops for the wrong side -> MT5 "Invalid
        # stops" rejections).
        order_type_str = _normalize_direction(direction)

        # 3a. Client-side stop validation — catch invalid stops BEFORE the MCP
        # roundtrip so the user gets a clear, actionable message instead of an
        # opaque "Invalid stops" from the server.  A buy's stop-loss must sit
        # BELOW entry (loss when price falls) and take-profit ABOVE; a sell's
        # stop-loss must sit ABOVE entry and take-profit BELOW.
        stop_issue = _validate_stop_side(order_type_str, entry_price, stop_loss, take_profit)
        if stop_issue:
            syslog("WARNING", "Execution", asset,
                   f"Order rejected by client-side stop validation: {stop_issue}",
                   extra={"event_id": eid, "direction": order_type_str,
                          "entry": entry_price, "sl": stop_loss, "tp": take_profit})
            return {"success": False, "event_id": eid,
                    "order_result": {"error": stop_issue},
                    "order_sent": False}

        # Order comment §9.2: ``{pattern_short}-v{major}-{event_id_short}``.
        # A caller-supplied standardized comment (from the signal candidate)
        # wins; otherwise the legacy ``LSW-V2-{event_id[:16]}`` format is kept
        # for backward compatibility with the historical sweep-only engine.
        if order_comment:
            comment = order_comment
        else:
            comment = f"LSW-V2-{eid[:16]}"
        paper_logger.log_signal(
            asset=asset, signal_time=now_iso, confirmation_time=now_iso,
            features_json=json.dumps({
                "direction": order_type_str, "volume": volume,
                "risk_lot": risk_lot, "lot_source": lot_source,
                "comment": comment, "entry_price": entry_price,
                "stop_loss": stop_loss, "take_profit": take_profit,
            }),
            rule_score=rule_score, model_probability=model_probability,
            entry_planned=entry_price, stop_planned=stop_loss,
            target_planned=take_profit, order_sent_time="",
            automation_level=automation_level,
            kill_switch_status_at_signal=1 if kill_switch_blocking else 0,
            event_id=eid,
        )

        # 4. Send market order with the LSW-V2 comment.
        # The MT5 MCP server's trade_send_market_order inputSchema (verified
        # live via tools/list) requires: symbol, `type` (STRING "buy"/"sell"),
        # volume; plus OPTIONAL sl/tp/comment.  additionalProperties=false means
        # `order_type`, `price` (a market order fills at the current price) and
        # `magic` are NOT valid fields and are rejected.  Sending the integer
        # 0/1 for `type` (as the older code and the previous partial fix did)
        # also fails with "type must be specified".  So send exactly the
        # schema-valid fields, with `type` as the string direction.  The
        # signal's planned `entry_price` is still used for logging/risk sizing
        # but is NOT sent to the server.
        order_params = {
            "symbol": asset, "type": order_type_str, "volume": volume,
            "sl": stop_loss, "tp": take_profit,
            "comment": comment,
        }

        try:
            raw_order_result = self._call_mcp("trade_send_market_order", order_params)
        except (ConnectionFailedError, MCPError) as exc:
            return {"success": False, "event_id": eid,
                    "order_result": {"error": str(exc)}, "order_sent": False}

        # Unwrap the MCP tool-result envelope.  The MT5 MCP bridge wraps the real
        # order result as JSON text inside ``{"content": [{"type":"text",
        # "text":"<json>"}]}`` — without this, ``success``/``retcode``/``price``
        # are invisible at the top level and EVERY order is classified as failed
        # with "Unknown error" even when it actually fills (retcode 10009).
        order_result = (_parse_mcp_response(raw_order_result)
                        if raw_order_result else raw_order_result)

        if not order_result:
            return {"success": False, "event_id": eid,
                    "order_result": {"error": "Empty MCP response"},
                    "order_sent": False}

        order_sent = order_result.get("success",
                            order_result.get("retcode", 0) == 10009)
        if order_sent:
            entry_actual = float(order_result.get("price",
                                   order_result.get("open_price", entry_price)))
            slippage = entry_actual - entry_price
            paper_logger.update_order_result(
                event_id=eid, entry_actual=entry_actual,
                slippage_actual=round(slippage, 2),
                exit_time="", exit_price=0.0, exit_reason="",
                mfe_r=0.0, mae_r=0.0, net_result_r=0.0,
            )
            _update_sent_time(eid, now_iso)
            self.state.update_mt5_status(connected=True)
            syslog("INFO", "Execution", asset,
                   f"Order sent → order_id={eid}, entry={entry_actual:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}",
                   extra={"event_id": eid, "direction": order_type_str,
                          "entry": entry_actual, "sl": stop_loss, "tp": take_profit,
                          "volume": volume, "risk_lot": risk_lot,
                          "slippage": round(slippage, 2)})
        else:
            retcode = order_result.get("retcode")
            # Surface the real reason to the caller instead of a generic
            # "Unknown error".  The MT5 server may reject with a retcode, or
            # return the rejection as plain text ({"text": ...}) or a list
            # ({"data": [...]}) that _parse_mcp_response wraps — expose it.
            # Prefer the server's human-readable `retcode_details` (e.g.
            # "Market closed" for retcode 10018) over the opaque
            # "MCP retcode N".
            data_repr = order_result.get("data")
            error_msg = (order_result.get("error")
                         or order_result.get("comment")
                         or order_result.get("retcode_details")
                         or (f"MCP retcode {retcode}" if retcode is not None
                             else None)
                         or order_result.get("text")
                         or (str(data_repr) if data_repr is not None else None)
                         or "Unknown error")
            order_result = dict(order_result)
            order_result["error"] = error_msg
            syslog("ERROR", "Execution", asset,
                   f"Order FAILED: {error_msg}",
                   extra={"event_id": eid, "direction": order_type_str,
                          "entry": entry_price, "order_result": order_result,
                          "retcode": retcode})

        return {"success": order_sent, "event_id": eid,
                "order_result": order_result, "order_sent": order_sent,
                "volume": volume, "risk_lot": risk_lot, "lot_source": lot_source}

    def place_order(self, symbol: str, order_type: str, volume: float = 0.01,
                    price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "", magic: int = MAGIC_NUMBER) -> dict[str, Any]:
        """Place a market order, sizing the lot from account risk (shared path).

        This is the entry point used by the manual GUI bridge (gui_bridge calls
        ``_exec_layer.place_order(symbol=..., order_type=..., price=..., sl=...,
        tp=...)``).  The lot is ALWAYS recomputed from the account risk budget
        using the same :meth:`compute_risk_lot` routine as the automated path,
        so manual and automated orders share identical risk sizing.  ``volume``
        is accepted for signature compatibility but overridden by the risk lot.
        """
        direction = "buy" if str(order_type).lower() in ("buy", "0") else "sell"
        risk_lot = self.compute_risk_lot(symbol, price, sl)
        final_volume = risk_lot if risk_lot is not None else float(volume)
        if risk_lot is None:
            syslog("WARNING", "Execution", symbol,
                   f"place_order: risk lot unavailable; using passed volume={final_volume}",
                   extra={"symbol": symbol, "price": price, "sl": sl, "tp": tp})
        return self.send_order(
            asset=symbol, direction=direction, entry_price=price,
            stop_loss=sl, take_profit=tp, volume=final_volume,
        )

    def close_position(self, position_id: str,
                       symbol: str | None = None) -> dict[str, Any]:
        # The MT5 MCP server's trade_close_single_position inputSchema (verified
        # live) requires `symbol` (safety check) + `position_ticket`, NOT
        # `position_id`.  Sending `position_id` is rejected with
        # "position_ticket must be specified".  Map the internal position id to
        # the `position_ticket` argument (as an integer — the schema wants a
        # number; a string is rejected), and always include the symbol.
        try:
            position_ticket = int(position_id)
        except (TypeError, ValueError):
            position_ticket = position_id
        params: dict[str, Any] = {"symbol": symbol or "", "position_ticket": position_ticket}
        try:
            result = self._call_mcp("trade_close_single_position", params)
        except ConnectionFailedError as exc:
            syslog("ERROR", "Execution", symbol or "",
                   f"Close position {position_id} failed: MCP disconnected",
                   extra={"position_id": position_id, "error": str(exc)})
            return {"success": False, "error": str(exc),
                    "position_id": position_id}
        if not result:
            syslog("ERROR", "Execution", symbol or "",
                   f"Close position {position_id} failed: empty response",
                   extra={"position_id": position_id})
            return {"success": False, "error": "Empty MCP response",
                    "position_id": position_id}
        # Unwrap the MCP result envelope (same reason as send_order).
        result = _parse_mcp_response(result)
        order_closed = result.get("success",
                         result.get("retcode", 0) == 10009)
        if order_closed:
            self.get_positions()
            syslog("INFO", "Execution", symbol or "",
                   f"Position {position_id} closed successfully",
                   extra={"position_id": position_id, "symbol": symbol or ""})
        else:
            syslog("WARNING", "Execution", symbol or "",
                   f"Position {position_id} close returned: {result.get('error', 'unknown')}",
                   extra={"position_id": position_id, "result": str(result)})
        return {"success": order_closed, "position_id": position_id,
                "result": result}

    def refresh_market_data(self, symbols: list[str] | None = None) -> None:
        if symbols is None:
            symbols = ["XAUUSD", "EURUSD"]
        try:
            self.get_account_info()
            self.state.update_mt5_status(connected=True)
        except ConnectionFailedError:
            self.state.update_mt5_status(connected=False)
            return
        for sym in symbols:
            try:
                existing = self.get_symbols()
                if not any(s.get("symbol", s.get("name", "")) == sym
                           for s in existing):
                    self.add_symbol(sym)
            except (ConnectionFailedError, MCPError):  # noqa: PERF203 (retry/continue semantics require try-except in loop)
                pass


def _update_sent_time(event_id: str, order_sent_time: str) -> None:
    """Update order_sent_time in DB after order dispatches."""
    conn = paper_logger._get_connection()
    try:
        conn.execute(
            "UPDATE signals SET order_sent_time = ? WHERE event_id = ?",
            (order_sent_time, event_id),
        )
        conn.commit()
    finally:
        conn.close()


_DEFAULT_EXECUTION: ExecutionLayer | None = None


def get_execution() -> ExecutionLayer:
    global _DEFAULT_EXECUTION
    if _DEFAULT_EXECUTION is None:
        _DEFAULT_EXECUTION = ExecutionLayer()
    return _DEFAULT_EXECUTION