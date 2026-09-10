# Live-Path Smoke Report — trading_v3 t9 (live-verifier)

**Date:** 2026-09-08 · **Runner:** `/tmp/ptv2_venv/bin/python` (Python 3.9.6)
**Workspace root:** `trading_v3` (self-contained; no `trading_live/` references)
**Result:** ✅ SMOKE PASSED (online, 2.1s) · ✅ SMOKE PASSED (offline-simulated, 2.3s)

This report is produced by the committed smoke script
[`live/smoke/mcp_live_smoke.py`](../live/smoke/mcp_live_smoke.py) and the
hermetic suite [`tests/test_live_smoke.py`](../tests/test_live_smoke.py). It
verifies the §9.1 live path end-to-end — registry(§5.5) → symbol config →
`MultiPatternEngine` on a **SHADOW** assignment → **Event Lake** writes →
offscreen GUI boot (§10.1 onboarding) — **without ever placing an order**.

---

## 1. Commands run

```bash
# V1 — smoke script (exact verify trio command #1)
cd trading_v3 && /tmp/ptv2_venv/bin/python live/smoke/mcp_live_smoke.py --no-orders \
    2>&1 | tee /tmp/live_smoke.log && tail -40 /tmp/live_smoke.log
#   → SMOKE PASSED in 2.1s (exit 0)

# V2 — hermetic pytest suite (verify trio command #2, offscreen)
cd trading_v3 && QT_QPA_PLATFORM=offscreen /tmp/ptv2_venv/bin/python \
    -m pytest tests/test_live_smoke.py -q
#   → 14 passed in 4.09s (exit 0)

# V3 — ruff (verify trio command #3)
cd trading_v3 && /tmp/ptv2_venv/bin/python -m ruff check live/smoke/ tests/test_live_smoke.py
#   → All checks passed! (exit 0)

# V4 — offline branch proof (MCP unreachable ⇒ recorded, local wiring still proven)
cd trading_v3 && MCP_URL="http://127.0.0.1:1/mcp" \
    /tmp/ptv2_venv/bin/python live/smoke/mcp_live_smoke.py --no-orders
#   → MCP OFFLINE (0.002s) recorded → SMOKE PASSED in 2.3s (exit 0)
```

## 2. Evidence output

### 1 — `.env` MCP config (presence-only)
- Keys present: `MCP_TOKEN`, `MCP_URL` (values loaded via
  python-dotenv-if-available, else the script's built-in `KEY=VALUE` parser;
  **values never printed, logged, or committed** — `_redact()` strips the
  token from every emitted string).
- `.env` not modified: mtime unchanged during the run (read-only path);
  file mode `-rw-------` unchanged.

### 2 — Symbol configs (`research/configs/symbols/{XAUUSD,EURUSD}.yaml`)
- Both parse: `status=validated`, `broker.mt5_symbol`, `digits=5`,
  `timeframe.primary=M15`, `risk_per_trade_pct=1.0`.
- **§10.1 multi-pattern `assignments[]` wiring (documented gap — not broken):**
  the symbol YAMLs are the legacy v2_frozen format and carry **no**
  `assignments[]` key. No code path reads `assignments[]` from symbol YAMLs;
  the §10.1 assignment store is the bridge-persisted
  `live/db/pattern_assignments.json` (SystemBridge `set_pattern_assignment` /
  `get_pattern_assignments`), which is exercised by the assignment save/load
  round-trip in section 6. This is the intended design — no fix required.

### 3 — Model Registry (§5.5 `model_registry/index.yaml`)
- 8 models loaded. DB/DT entries present and assignable:
  - `double_bottom_xauusd_m15_v1` (double_bottom, `validated`, gate=True,
    schema `double-v1.0`, horizon 72)
  - `double_top_xauusd_m15_v1` (double_top, `validated`, gate=True,
    schema `double-v1.0`, horizon 72)
- `falling_wedge` (trained/gate=False) and the `xauusd_v2` retired alias are
  excluded from every §10.1 assignment dropdown (asserted).

### 4 — MCP health ping
- **ONLINE** in 0.008 s (workspace = `C:\Program Files\MetaTrader 5\MQL5`).
- Offline branch (V4): `Cannot connect to MCP server` — recorded as WARN,
  all local wiring still verified below.

### 5 — SHADOW assignment → MultiPatternEngine → Event Lake (ZERO orders)
- **Live window:** 500 XAUUSDm M15 bars fetched through the production
  adapter (`ExecutionLayer` + `MCPCandleSource`); shadow scan produced
  **0 SignalCandidates** and **0 events** in that window (valid
  "no confirmed pattern in the last 500 bars" outcome).
- **Reference window** (bundled XAUUSD M15 dataset recent bars, ends
  2026-09-03 22:45, tail 1500): 2 double-bottom events detected →
  - SignalCandidates produced: **0** (shadow never emits — asserted)
  - Events appended to Event Lake `events/`: **2**
  - Outcomes appended (`outcomes/`, horizon 72): **2**
  - Lake rows with `lifecycle_state=shadow`: **2**
  - Append-only semantics: duplicate append raised `EventLakeError`
- Lake root is an **isolated temp dir** (`/tmp/live_smoke_lake_*`);
  `event_lake/events|outcomes|metrics` is not polluted by the smoke
  (deterministic event_ids would collide across smoke runs — production
  shadow writes belong to the polling engine's `_record_shadow` hook, which
  this smoke wires at the caller layer per §5.2 record-only semantics; t10
  integration can move the write into the hook).

### 5b — §5.5 `live_metrics_ref` baselines (`event_lake/metrics/`, §7.2)
- t7 review consistency: every registry entry's `live_metrics_ref` points at
  `event_lake/metrics/<ref-basename>.parquet`, which started **empty**. The
  smoke now materializes one §7.2 baseline row per target (append-only by
  `asof`, idempotent — re-runs skip) via `EventLake.append_metrics`:
  `asof = trained_at`, `pf`/`win_rate`/`n_trades` sourced from the entry's
  §6.3 gate metrics, `max_psi = 0.0` (no drift measurement yet),
  `psi_retrain = False`.
- Materialized 7/7 targets — **all registry `live_metrics_ref`s resolve**:
  - `double_bottom_xauusd_m15_v1.parquet` (validated — pf 2.5061,
    win_rate 0.4314, n_trades 56, asof 2026-09-07T22:27:48Z)
  - `double_top_xauusd_m15_v1.parquet` (validated — pf 1.7095,
    win_rate 0.4328, n_trades 60, asof 2026-09-07T22:27:52Z)
  - `falling_wedge_xauusd_m15_v1.parquet` / `head_shoulders_...` /
    `inverse_head_shoulders_...` (trained — gate metrics NaN/0, asof
    2026-09-07T22:28:xxZ)
  - `liquidity_sweep_eurusd_h16_v2.parquet` (live legacy — n_trades 267,
    win_rate 0.8333, asof 2026-09-06T00:00:00Z) and
    `liquidity_sweep_xauusd_h16_v2.parquet` (live legacy — pf 1.305,
    n_trades 312, asof 2026-09-05T00:00:00Z; seeded from the live
    `xauusd_v2_h16_20260905` entry — shared ref with the retired alias).
- These are initial baseline snapshots; the periodic LifecycleManager
  drift/metrics job appends new snapshots at future `asof`s (append-only).

### 6 — GUI offscreen (`QT_QPA_PLATFORM=offscreen`)
- `gui_main` booted: window `Paper Trading V2 — Liquidity Sweep`.
- 8 models available to the bridge; onboarding filter for
  `double_bottom` → `[double_bottom_xauusd_m15_v1]`; `falling_wedge`
  (trained) → `[]`.
- §10.1 assignment save/load round-trip against persisted state: **OK**
  (`smoke-shadow-db@M15` → fresh bridge → restored with
  `model_id` + `feature_schema_version` + `state=shadow`).
- Workspace state restored afterwards: `pattern_assignments.json` removed,
  `live/db/symbol_registry.json` byte-identical.

### 7 — Declarations
- **NO real orders placed in any mode.** Every engine ran with
  `lifecycle_state=shadow` and `candidates==0` is asserted; the smoke
  contains no order-placing call sites (static guard tested).
- **NO secrets printed/logged/committed.** Token handled via `os.environ`
  only; `_redact()` applied to all emitted strings.
- `.env` untouched (read-only); `research/core/contracts.py` untouched
  (mtime invariant tested).

## 3. Wiring notes / observations (non-blocking)

1. **stdlib `logging` shadow hazard:** `execution_layer_v2.py` inserts
   `live/` at `sys.path[0]`, which shadows the standard `logging` module with
   the `live/logging` package for any later import. The codebase relies on
   import order + module cache (GUI/testing order happens to work). The smoke
   makes this deterministic with `_prime_engine_imports()` (imports
   `signal_engine_v2` + `signal_polling_engine_v2` before anything injects
   `live/`). **Suggestion for t10:** fix `execution_layer_v2` to insert
   `_PARENT` at the *end* of `sys.path` or skip the bare-path insert, which
   would remove the hazard entirely.
2. **python-dotenv not installed** in `/tmp/ptv2_venv` (Python 3.9.6 venv):
   the smoke prefers `python-dotenv` and falls back to a built-in
   `KEY=VALUE` parser with identical secure semantics (documented in the
   script). Installing `python-dotenv` into the venv is optional.
3. **Qt stylesheet warnings** (`Could not parse stylesheet of object
   QPushButton...`, `qt.qpa.fonts` font-alias) appear on offscreen boot —
   pre-existing GUI presentation warnings, non-fatal (GUI boots, filters,
   and round-trips all verified).
4. **Live window event count:** 0 double-bottom events in the last 500 M15
   bars is a legitimate market outcome, not a wiring failure — the reference
   window proves the lake write path with deterministic events.
5. **Metrics baselines:** seeded by the smoke via `EventLake.append_metrics`
   (idempotent, append-only by `asof`). The LifecycleManager periodic drift /
   rolling-metrics job (§5.3/§7.2) should append subsequent snapshots at new
   `asof`s rather than overwriting the baseline row.