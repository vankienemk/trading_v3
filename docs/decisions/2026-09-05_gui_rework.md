# GUI Rework Decision — 2026-09-05

**Date:** 2026-09-05
**Decision:** Complete GUI rework — remove 7-step wizard, add Model Registry, add System Log tab
**Status:** ✅ Implemented and verified (gui_rework_team, all 7 tasks PASS)

---

## Context

The original Paper Trading V2 GUI featured a 7-step wizard (Data Audit → Train → Gate → Activate) that was overly complex for the intended use case. Users needed to:
- Train models from within the GUI
- Navigate through intermediate statuses (`auditing`, `building_events`, `building_dataset`, `training`, `walk_forward`, `gating`)
- Understand complex pipeline logic just to activate a symbol

The rework was driven by the need for **simplicity, observability, and separation of concerns** — model training belongs in the research pipeline, not in the live trading GUI.

---

## Decisions

### 1. Remove 7-Step Wizard (BREAKING)

**Before:** Symbol Onboarding had 7 steps (Data Audit → Build Events → Build Dataset → Train Model → Walk-Forward → Gate → Activate) with intermediate statuses.

**After:** Simplified to a single **Add New Symbol** dialog: select symbol + select model → status=`validated`, active=true.

**Removed:**
- Entire wizard code (`_start_wizard`, `_on_wizard_next`, step handlers, progress gauge)
- All intermediate statuses: `auditing`, `building_events`, `building_dataset`, `training`, `walk_forward`, `gating`
- All pipeline training/dataset/events calls from the GUI
- Hard-gate CI/train/walk-forward logic

**Retained:**
- Only symbols with `status = "validated"` may run Signal Engine
- Emergency Stop (red button, globally active)
- Confirmation dialogs for all important actions
- Dark theme

### 2. Add Model Registry (NEW)

A centralized registry to manage pre-trained models, decoupling model training (research pipeline) from model usage (live trading).

**Structure per model:**

```yaml
model_id: "xauusd_v2_h16_20260905"
symbol_origin: "XAUUSD"
model_path: "artifacts/models/XAUUSD/model.pkl"
calibrator_path: "artifacts/models/XAUUSD/calibrator.pkl"
feature_schema: "artifacts/feature_schemas/XAUUSD/features.json"
horizon: 16
target: "outcome_2r_h16"
metrics:
  pf_oos: 1.305
  ci_lower: 1.092
  ci_upper: 1.561
  n_trades: 312
created_at: "2026-09-05"
notes: "Pooled OOS meta-analysis V2"
```

**API:**

```python
get_available_models() → List[ModelInfo]
get_model(model_id: str) → ModelInfo | None
assign_model_to_symbol(symbol: str, model_id: str) → bool
add_symbol_with_model(symbol: str, model_id: str, activate: bool = True) → bool
```

**Storage:** `configs/models/` or `artifacts/models/index.yaml`

### 3. Add System Log Tab (NEW)

A real-time, filterable system log for observability — users can see exactly what the system is doing after activating a symbol.

**Log format:**
```
[TIMESTAMP] [LEVEL] [SOURCE] [SYMBOL] Message
```

**Example:**
```
2026-09-05 21:45:00.123 [INFO ] [SignalEngine] [XAUUSD] New M15 candle closed @ 2650.45 → running detection
2026-09-05 21:45:00.187 [INFO ] [SignalEngine] [XAUUSD] Sweep detected (bullish, penetration=0.12 ATR) → score=72.4, prob=0.68
```

**Features:**
- Real-time auto-scroll (with Pause button)
- Level filters (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- Symbol filter (multi-select or "All")
- Source filter (SignalEngine, RiskGuard, Execution, MCP, System, UserAction)
- Time range filter (Last 5 min / 15 min / 1 hour / Today / Custom)
- Full-text search
- Click-to-expand detail view (JSON `extra`)
- Export log (CSV/TXT)
- 10,000 line / 24-hour ring buffer limit
- Thread-safe

**Backend logger** (`logger_v2.py`):
- Writes to SQLite (`system_logs` table)
- Pushes to in-memory ring buffer for GUI
- Optional rotating file logger

**Required logging points:**
- `signal_engine_v2.py` (especially after each new M15 candle)
- `risk_guard_v2.py`
- `execution_layer_v2.py`
- MCP / connection logic in `gui_bridge.py`

### 4. Update SymbolConfig

```python
@dataclass
class SymbolConfig:
    name: str
    status: str                     # "validated" | "candidate" | "rejected"
    active: bool = False
    model_id: Optional[str] = None  # reference to Model Registry
    position_size_multiplier: float = 1.0
    automation_level: int = 1       # 1 = Manual, 2 = Semi-auto
```

**Rule:** Only when `status == "validated"` AND `model_id` is valid may the Signal Engine run for that symbol. `active` is a runtime toggle only.

### 5. Update Signal Engine

- On initialization for a symbol → reads `model_id` from `SymbolConfig`
- Loads the correct `model.pkl`, `calibrator.pkl`, `feature_schema`
- If `model_id` missing or files not found → logs ERROR and skips that symbol

---

## Files Modified

| File | Action |
|------|--------|
| `gui_tab_onboarding.py` | **Rewritten** — table + simple dialog, no wizard |
| `gui_main.py` | Added System Log tab, kept Emergency Stop |
| `gui_bridge.py` / `SystemBridge` | Added Model Registry methods + model assignment |
| `shared_app_state_v2.py` | Updated `SymbolConfig` (added `model_id`) |
| `logger_v2.py` | Extended into centralized system logger |
| `signal_engine_v2.py` | Load model by `model_id` |
| `configs/models/` or `artifacts/models/index.yaml` | Created for model management |
| Removed | Entire 7-step wizard code + intermediate statuses |

---

## Acceptance Criteria

### Symbol Onboarding
1. ✅ Add new symbol by selecting symbol + selecting model → status=`validated`, active=true
2. ✅ Table displays assigned model clearly
3. ✅ Change model without delete/re-add
4. ✅ No train model buttons or flows in GUI
5. ✅ Clean layout, no text cutoff

### System Log
1. ✅ After activating a symbol, new M15 candle close → at least 1 INFO line describing Signal Engine action
2. ✅ Filter by ERROR + CRITICAL for a specific symbol
3. ✅ Log preserved when switching tabs
4. ✅ Auto-scroll works, with Pause
5. ✅ Order placement / connection errors visible with full info (order_id, price, reason)
6. ✅ Stable performance (no GUI lag)

### Safety
1. ✅ Only `validated` + valid `model_id` → Signal Engine runs
2. ✅ Emergency Stop still works
3. ✅ All important changes have confirmation + log

---

## Implementation Order

1. ✅ **Model Registry + SymbolConfig update** (foundation)
2. ✅ **Rewrite Symbol Onboarding tab** (remove wizard)
3. ✅ **Centralized Logger + System Log tab**
4. ✅ **Update Signal Engine** to load model by `model_id`
5. ✅ **Cleanup old code + update docs**

---

## Team Execution

| Task | Owner | Status |
|------|-------|--------|
| t1 — Model Registry + SymbolConfig | Model_Registry_Dev | ✅ COMPLETED |
| t2 — Workspace cleanup | Workspace_Cleanup | ✅ COMPLETED |
| t3 — Symbol Onboarding rewrite | SymbolOnboarding_Rewrite | ✅ COMPLETED (546 lines, no wizard) |
| t4 — Centralized System Logger + System Log tab | SystemLogger | ✅ COMPLETED |
| t5 — System logging integration | SystemLog_Integration | ✅ COMPLETED |
| t6 — Review all changes | GUI_Reviewer | ✅ COMPLETED (VERDICT PASS) |
| t7 — Post-review fixes | Post_Review_Fixes | ✅ COMPLETED |

**gui_rework_team dissolved** 2026-09-05.

---

## Source Documents

- `GUI_REWORK_REQUIREMENTS.md` — Full rework requirements document
- `paper_trading_v2/gui_tab_onboarding.py` — Current onboarding implementation
- `paper_trading_v2/gui_tab_system_log.py` — Current system log implementation
- `paper_trading_v2/logger_v2.py` — Centralized logger
- `paper_trading_v2/shared_app_state_v2.py` — Updated SymbolConfig

> **Note on paths:** The source document references above use the original project layout (pre-restructure).  
> After the 2026-09-06 workspace restructure to `trading_live/`, these files now live at:  
> - `trading_live/live/gui/gui_tab_onboarding.py`  
> - `trading_live/live/gui/gui_tab_system_log.py`  
> - `trading_live/live/logging/logger_v2.py`  
> - `trading_live/live/state/shared_app_state_v2.py`  
> 
> The decision document itself was written before the restructure and reflects the original paths,  
> which were correct at the time of writing.