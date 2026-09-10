# Paper Trading V2 — Knowledge Base: Error Patterns & Fixes

> **Purpose**: Central reference for every team member (GUI, Signal Engine, Risk Guard, Execution Layer) to diagnose and fix known errors in the MCP/MT5/Paper Trading ecosystem.
>
> **Scope**: MCP connection, MCP tools, MT5 terminal, Signal Engine, Risk Guard, GUI (PySide6), and auto-reconnect patterns.

---

## Table of Contents

1. [MCP Connection Errors](#1-mcp-connection-errors)
2. [MCP Tool Errors](#2-mcp-tool-errors)
3. [MT5 Terminal Errors](#3-mt5-terminal-errors)
4. [Signal Engine Errors](#4-signal-engine-errors)
5. [Risk Guard Errors](#5-risk-guard-errors)
6. [GUI (PySide6) Errors](#6-gui-pyside6-errors)
7. [Auto-Reconnect Pattern](#7-auto-reconnect-pattern)
8. [Database Errors](#8-database-errors)
9. [Configuration Errors](#9-configuration-errors)
10. [Appendix: Diagnostic Commands](#10-appendix-diagnostic-commands)

---

## 1. MCP Connection Errors

### MCP_TOKEN is not set.

**Root cause**: The MCP_TOKEN environment variable is empty or unset when MCPClient.connect() is called.

**Affected modules**: execution_layer.py, execution_layer_v2.py, mt5_mcp_client.py, gui_tab_account.py

**Fix**:
1. Set MCP_TOKEN in your shell: export MCP_TOKEN="your-token-here"
2. Or input the token via Tab4 -> MCP Token Input in the GUI (V2 only)
3. Then click Reconnect

**Prevention**: Always set MCP_TOKEN before starting run.py. The V2 GUI stores the token in shared_app_state so it persists across reconnects.

### MCP HTTP 401 — authentication failed

**Root cause**: The MCP server rejected the token — wrong, expired, or malformed.

**Affected modules**: mt5_mcp_client.py _post() method, execution_layer_v2.py

**Fix**:
1. Verify your token with the MCP bridge provider
2. Update via Tab4 -> MCP Token Input -> Reconnect
3. Check MCP_TOKEN env var for leading/trailing whitespace

**Diagnostic**: python3 mt5_mcp_client.py health — if it returns ok: false with "401", the token is wrong.

### Connection refused / Cannot connect to MCP server

**Root cause**: The MCP bridge (MetaTrader 5 MCP server) is not running at the expected endpoint.

**Affected modules**: mt5_mcp_client.py, execution_layer.py, execution_layer_v2.py

**Fix**:
1. Ensure MetaTrader 5 is running
2. Ensure the MCP bridge is started (check MT5 Navigator -> MCP panel)
3. Verify the endpoint URL: default is http://127.0.0.1:22346/mcp
4. Check if another process is using port 22346: lsof -i :22346

**Prevention**: The V2 polling engine re-checks MCP health every 30 seconds and auto-reconnects.

### MCP returned SSE data that could not be parsed

**Root cause**: The MCP server returned malformed Server-Sent Events (SSE) data — possibly a proxy/nginx interfering, or the server is in a bad state.

**Affected modules**: mt5_mcp_client.py _parse_sse()

**Fix**:
1. Restart the MCP bridge
2. Check if a reverse proxy is buffering or modifying the SSE stream
3. Ensure MCP-Protocol-Version header matches the server's expectation (2025-06-18)

### MCP initialize returned an empty response

**Root cause**: The MCP server responded with no data during the initialize handshake.

**Fix**:
1. Restart the MCP bridge
2. Check network connectivity
3. Verify the server URL includes /mcp path

---

## 2. MCP Tool Errors

### Tool not found

**Root cause**: The tool name passed to call_tool() does not match any tool registered by the MCP server.

**Affected modules**: Any module calling MCPClient.call_tool() or ExecutionLayer._call_mcp()

**Fix**:
1. Run python3 mt5_mcp_client.py tools to list all available tools
2. Verify the exact tool name (case-sensitive)
3. Common tool names: get_chart_history, get_trading_account_info, get_trading_open_positions, trade_send_market_order, get_marketwatch_symbols, add_marketwatch_symbol, get_symbol_info, get_workspace_info

**Prevention**: The V2 _call_mcp() method logs tool names on failure. Check logs for the exact name used.

### Invalid params / Invalid parameters

**Root cause**: The tool arguments do not match the expected schema.

**Affected modules**: execution_layer.py, execution_layer_v2.py

**Fix**:
1. Check the tool schema: python3 mt5_mcp_client.py call <tool_name> '{}'
2. Common mismatches: symbol vs sym, datetime_from vs from_date
3. Wrong data type (string vs integer for volume)

**Example**: get_chart_history expects {symbol, period, datetime_from, datetime_to, limit}

### Session not initialized

**Root cause**: The MCP session expired, was closed by the server, or was never initialized.

**Affected modules**: mt5_mcp_client.py, all modules that use it

**Fix**: The auto-reconnect mechanism in _call_mcp() resets the session and re-initializes automatically. If the issue persists, restart the MCP bridge.

**Detection**: client.initialized becomes False; the next call triggers connect().

---

## 3. MT5 Terminal Errors

### Symbol not in MarketWatch

**Root cause**: The symbol (e.g. XAUUSDm) is not added to MetaTrader 5's MarketWatch.

**Affected modules**: execution_layer.py add_symbol(), execution_layer_v2.py

**Fix**:
1. Call add_symbol(symbol) before trading
2. In V2, the polling engine calls refresh_market_data() which auto-adds configured symbols
3. Manual fix: Right-click MarketWatch in MT5 -> Symbols -> find and enable the symbol

**Prevention**: refresh_market_data() calls add_symbol() for every configured symbol on each poll cycle.

### Invalid volume / Volume out of range

**Root cause**: The lot size does not match MT5's symbol constraints (min lot, max lot, step size).

**Affected modules**: execution_layer.py send_order()

**Fix**:
1. Check symbol's volume constraints via get_symbol_info()
2. Common values: XAUUSD min=0.01, step=0.01, max varies by broker
3. Round volume: volume = round(volume / step) * step

**Prevention**: The V2 execution layer fetches symbol_info and validates volume before sending orders.

### No connection / Terminal disconnected

**Root cause**: MetaTrader 5 terminal has lost connection to the broker.

**Affected modules**: All MCP-dependent modules

**Fix**:
1. Check MT5 terminal status (bottom-right corner)
2. Reconnect MT5 to the broker
3. The MCP bridge auto-reconnects when MT5 is back online

**Prevention**: The V2 polling engine checks health_check() every 30s and updates shared state.

### Trade context busy

**Root cause**: MT5 is processing another trade operation.

**Fix**: Retry after a short delay (1-2 seconds). The V2 _call_mcp() already implements retry logic.

### Invalid stops / Invalid S/L or T/P

**Root cause**: Stop-loss or take-profit levels violate broker constraints (distance from market, freeze level).

**Fix**:
1. Check get_symbol_info() for stops_level and freeze_level
2. Ensure S/L distance >= stops_level * point
3. Increase buffer_atr if S/L is too tight

---

## 4. Signal Engine Errors

### model.pkl not found

**Root cause**: The frozen V2 model artifact (model.pkl) is missing or the path is wrong.

**Affected modules**: signal_engine.py _load_model_and_calibrator()

**Fix**:
1. Check expected path: xauusd-liquidity-sweep/pipeline_v2/artifacts/models/model.pkl
2. Verify: ls -la xauusd-liquidity-sweep/pipeline_v2/artifacts/models/
3. Use absolute paths or set env var
4. For new symbols without model, set signal_engine.enabled: false in symbol YAML

**Prevention**: SignalEngineBridge.load() catches FileNotFoundError and logs a warning instead of crashing.

### model.pkl has no .estimator

**Root cause**: The loaded pickle is not a CalibratedClassifierCV — wrong file or corrupted.

**Fix**: Verify the model file was generated by the V2 training pipeline. Re-run training if needed.

### Feature mismatch

**Root cause**: Signal engine expects 30 features but candle data produces a different set.

**Affected modules**: signal_engine.py _compute_model_prob()

**Fix**:
1. Check feature_pipeline.py and SIGNAL_FEATURE_NAMES in src/features/registry.py
2. The engine handles missing columns by falling back to available ones

**Diagnostic**: Print X.shape and compare to model.estimator.n_features_in_

### No candles returned for XAUUSD

**Root cause**: MCP get_chart_history returned empty data for the symbol.

**Fix**:
1. Check MCP connection
2. Verify symbol is in MarketWatch
3. Ensure symbol has recent trading data

### Missing required columns in MCP data

**Root cause**: MCP returned candle data without {open, high, low, close} columns.

**Fix**:
1. Check MCP candle response format
2. _MCPClientAdapter normalises column names to lowercase
3. Missing volume defaults to 0

### Signal engine lazy-load failure

**Root cause**: signal_engine module could not be imported (missing pandas, numpy, sklearn, joblib).

**Fix**: pip install pandas numpy scikit-learn joblib pyyaml. Python 3.9+ required.

---

## 5. Risk Guard Errors

### Only X/Y trades — not enough for kill-switch evaluation

**Root cause**: Kill-switch requires KILL_SWITCH_MIN_TRADES (default 20) completed trades before evaluating PF CI.

**Fix**: Expected during warm-up. Risk guard auto-activates once enough trades accumulate.

**GUI display**: Shows "N/A — insufficient trades (X/20)" with neutral color.

### PF CI lower bound < 1.0 — kill-switch activated

**Root cause**: Block bootstrap 95% CI lower bound on Profit Factor fell below 1.0.

**Affected modules**: risk_guard.py _refresh_asset()

**Fix**:
1. Review trade log for poor performance
2. Do NOT change frozen configs
3. Manual override: risk_guard.manual_override_kill_switch(asset, active=False)
4. GUI Tab3 has "Confirm Reopen" button (requires text input)

**Recovery**: Kill-switch does NOT auto-reset. Human must review and manually reopen.

### Block bootstrap returned None CIs

**Root cause**: Fewer than 100 valid bootstrap resamples (zero or uniform net_r_values).

**Fix**: Check that trades have non-zero net_result_r values. Bootstrap needs both profitable and losing trades.

### Max open positions / interval checks

**Root cause**: Signal rejected due to position limits or minimum interval.

**Fix**: Protective limits — wait for existing positions to close or interval to elapse.

---

## 6. GUI (PySide6) Errors

### PySide6 not installed

**Root cause**: PySide6 is not installed.

**Fix**: pip install PySide6

**Alternative**: Fall back to Dear PyGui: pip install dearpygui and set gui_framework: "dearpygui" in v2_frozen.yaml

### GUI freezes during background operations

**Root cause**: Long-running operations on the main GUI thread block the event loop.

**Fix**:
1. Use QThread for background polling
2. Use QTimer for periodic refresh
3. Use signals/slots for cross-thread updates
4. Never call time.sleep() in the main thread

**V2 Pattern**: gui_components.py defines PollWorker QObject with data_ready signal.

### QTimer / QThread already referenced

**Root cause**: Timer or thread started twice without stopping previous instance.

**Fix**: Stop/join previous thread before starting a new one. Use persistent thread with stop() flag.

### GUI not updating after state change

**Root cause**: State updated from worker thread but GUI not notified.

**Fix**: Use pyqtSignal to notify main thread. Call QWidget.update() after state changes.

### Cannot create children for a parent in a different thread

**Root cause**: QWidget created in worker thread with main-thread parent.

**Fix**: Create GUI widgets only in the main thread. Use signals to pass data.

---

## 7. Auto-Reconnect Pattern

**This pattern must be used in EVERY module that calls MCP.**

The auto-reconnect pattern consists of two levels:

### Level 1: Low-level MCPClient (mt5_mcp_client.py)

MCPClient._post() checks for 400 errors (session expired) and auto-reconnects:
- Sets session_id = None, initialized = False, tools_cache = None
- The next call_tool() triggers a fresh connect()

MCPClient.call_tool() checks error messages containing "session" and "initialized" to trigger reconnect.

### Level 2: ExecutionLayer (execution_layer.py / execution_layer_v2.py)

ExecutionLayer._call_mcp() wraps every MCP call with retry + reconnect:
- Up to RETRY_COUNT + 1 attempts (default 2 total)
- On failure: close client, reset session flag, re-init session
- Only raise ConnectionFailedError after all retries exhausted

**Rule**: Every method that calls an MCP tool MUST go through _call_mcp(). Never call client.call_tool() directly outside ExecutionLayer.

### MCP Token Configuration

The MCP token can be configured via:
1. Environment variable: export MCP_TOKEN="your-token"
2. GUI input: Tab4 -> MCP Token Input field -> Reconnect button
3. Constructor: ExecutionLayer(token="your-token")

The token is stored in SharedAppState so it survives GUI reconnects.

---

## 8. Database Errors

### database is locked (SQLite)

**Root cause**: Concurrent writes from multiple threads/processes.

**Fix**: Use WAL journal mode (PRAGMA journal_mode=WAL). The logger already sets this.

### no such table: signals

**Root cause**: Database not initialized. Call logger.init_db() before any write.

**Fix**: Ensure run.py calls init_db() at startup before the signal engine starts.

---

## 9. Configuration Errors

### Config file not found

**Root cause**: The YAML config path is wrong or the file does not exist.

**Fix**: Verify the --config argument path. Default: paper_trading_v2/configs/v2_frozen.yaml

### Symbol YAML validation failed

**Root cause**: A per-symbol YAML file in configs/symbols/ has missing or invalid fields.

**Fix**: Run the symbol validator: python3 -c "import yaml; yaml.safe_load(open('configs/symbols/XAUUSD.yaml'))"

### Missing required field in symbol config

**Root cause**: A symbol YAML is missing required fields (symbol.name, broker.mt5_symbol, position_sizing).

**Fix**: Compare against _template.yaml. Required fields are marked [REQUIRED].

---

## 10. Appendix: Diagnostic Commands

### MCP Health Check
```
python3 mt5_mcp_client.py health
```

### List MCP Tools
```
python3 mt5_mcp_client.py tools
```

### Call an MCP Tool
```
python3 mt5_mcp_client.py call get_trading_account_info '{}'
python3 mt5_mcp_client.py call get_marketwatch_symbols '{}'
python3 mt5_mcp_client.py call get_chart_history '{"symbol":"XAUUSD","period":"M15","limit":10}'
```

### Validate Symbol Configs
```
python3 -c "import yaml; yaml.safe_load(open('configs/symbols/XAUUSD.yaml')); print('OK')"
python3 -c "import yaml; yaml.safe_load(open('configs/symbols/EURUSD.yaml')); print('OK')"
```

### Check MCP Port
```
lsof -i :22346
```

### Check Model Artifacts
```
ls -la xauusd-liquidity-sweep/pipeline_v2/artifacts/models/
```

### Run Paper Trading System
```
python3 run.py --config paper_trading_v2/configs/v2_frozen.yaml
```
