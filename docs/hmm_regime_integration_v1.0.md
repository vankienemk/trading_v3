# HMM Regime Plugin — t3 Integration Note (v1.0)

**Task:** t3 (integrator) — emiter + plug/unplug + Model Registry §4.3 + hard gate + Event Lake publish.
**Contract:** `docs/HMM_REGIME_REQUIREMENTS_v1.0.md` §4/§5/§6 (acceptance B6–B10).
**Status:** delivered — default **OFF**, legacy models byte-identical.

---

## 1. What was built

### 1.1 New runtime modules (`live/engine/`)

The t3 integration runtime lives in `live/engine` (the t3 verify gate mypy-
strict's `live/engine/feature_emitter.py` + `live/state/shared_app_state_v2.py`;
the emitter's runtime location per the t3 verify contract).  The t2 plugin
package `research/regime/` (BaseRegimePlugin + CausalGaussianHMM + persist) is
left untouched.

| Module | Contents | Purpose |
|---|---|---|
| `live/engine/feature_emitter.py` | `HMMFeatureEmitter` (+ `state_at_known_at`, `state_at_confirm_bar`, `regime_flags`, `requires_hmm_regime`, `build_feature_vector`, `publish_regime_states`) | Causal injection of `hmm_state`, `hmm_prob_<state>`, `hmm_confidence` into `PatternEvent.attributes` / feature frames (guide §5 step 4, requirements §4.2); guide §4.2B feature-vector builder; §5 step 2 Event Lake publish. |
| `live/engine/hard_gate.py` | `is_allowed`, `apply_gate`, `regime_filter_rules` (+ reason constants) | Per-pattern regime hard gate (guide §8, requirements §5) — **one shared function** for live ≡ backtest. |
| `live/engine/regime_wiring.py` | `RegimeSlot`, `resolve_regime_config`, `resolve_regime_wiring`, `load_plugin_config` | Resolves YAML master switches + `optional_plugins` into a per-assignment slot (default OFF). |

### 1.3 MultiPatternEngine integration (`live/engine/signal_engine_v2.py`)

- `PatternAssignment` gains optional regime fields (`regime_plugin`,
  `regime_config`, `regime_filter`, `model_feature_list`,
  `model_optional_plugins`, `emitter_enabled`, `gate_enabled`,
  `regime_states`) — all default → **no behaviour change for legacy
  assignments**.
- `MultiPatternEngine` accepts a shared `regime_plugin` +
  `regime_config_source`; per-assignment slots resolved in
  `_assign_regime_slots`.
- `_run_assignment`: after detect + causality validation, computes the causal
  states once (cached on the slot) and injects `hmm_*` attributes **only**
  when the feature emitter is enabled.
- `check_new_bar`: before `_group_to_candidate` — (a) §6.3 fail-closed: a
  model whose `feature_list` has `hmm_*` + `required: true` plugin with no
  active wiring → `discard_reason="hmm_unavailable"`, no signal; (b) hard
  gate: blocked events keep `discard_reason="regime_blocked"` (Event Lake
  §7.1) and produce no candidate.

### 1.4 Backtest runner (`research/multi_backtest/runner.py`)

- `BacktestAssignment` mirrors the live regime fields (default OFF).
- `detect_all` resolves slots + attaches emitter features (same causal rules).
- `run_symbol_backtest` replay loop applies the **same** `is_allowed` before
  `simulate_trade`; blocked representative → rejected trade with
  `cap_reason="regime_blocked"` + `discard_reason` on the event.
- `make_tier2_scorer` reads `hmm_*` from event attributes for models whose
  `feature_names` include them (new-HMM-model support); legacy models with no
  `hmm_*` are untouched.
- `build_assignments` accepts regime wiring kwargs.

### 1.5 Model Registry §4.3 (`live/state/shared_app_state_v2.py` + `model_registry/index.yaml`)

- `ModelInfo` gains `feature_list` + `optional_plugins` (default empty).
- `load_from_yaml` parses both; when `feature_list` absent → back-compat
  auto-resolve from the artifact's `features.json` (supports both the
  `{features: [str, ...]}` and `[{"name": ...}, ...]` shapes).  All 8 existing
  entries auto-resolve; nothing in lifecycle/metrics is changed.
- `index.yaml` header documents the §4.3 fields + an example HMM-model entry
  (schema illustration — real HMM entries point at real artifacts when t5/OOS
  trains them).

### 1.6 Event Lake publish (`live/engine/feature_emitter.py`)

`publish_regime_states` / `read_regime_states` write the `RegimeState` series
to `<lake_root>/regime/<SYMBOL>_<TF>.parquet` (guide §5 step 2) via the
plugin's `research.regime.persist` round-trip (replaceable materialisation —
deterministic recompute, not append-only).

### 1.7 Configs — canonical path (t7 consistency note)

**Canonical sample plugin config: `trading_v3/configs/plugins/hmm_regime.yaml`**
(root-relative `configs/plugins/hmm_regime.yaml`).

- Master `plugin.enabled: false` (default OFF) + `hmm.enabled` + per-pattern
  `regime_filter.rules` (DB/LSW/DT/H&S/…) with `regime_filter.enabled: false`.
- `configs/symbols/_hmm_template.yaml` — sample symbol-level wiring:
  `features.optional_plugins: [hmm_regime]` + `features.optional.hmm.enabled`
  (guide §4.2A/§6) + the registry-entry example.

**Path decision + reason (vs requirements §2/§9 which drew the file at
`research/configs/plugins/hmm_regime.yaml`):**

1. **Actual code reads an explicit path — there is NO default-path auto-load.**
   `research/regime/config.py::load_hmm_regime_config(path)` (t2) and
   `live/engine/regime_wiring.py::load_plugin_config(path)` (t3) both take the
   path as a required argument; the engine/runner receive an already-fitted
   plugin + a resolved config dict from the caller (wiring / OOS driver).
   Nothing in the runtime hard-codes `research/configs/plugins/`.
2. **One physical file exists**: only `trading_v3/configs/plugins/hmm_regime.yaml`
   is on disk (created by t2 per its declared in-scope `trading_v3/configs/`);
   `research/configs/plugins/` does not exist and no duplicate was added.
3. **Tests agree**: t2 `tests/test_hmm_regime_plugin.py` (CONFIG_YAML) and t3
   `tests/test_hmm_regime_integration.py` both load
   `trading_v3/configs/plugins/hmm_regime.yaml`.
4. The guide itself (§6, line `configs/plugins/hmm_regime.yaml`, and the
   requirements §5 example) already uses the top-level `configs/` form; the
   `research/configs/plugins/` string appears only in the requirements §2/§9
   layout illustration (documentation-only) and one stale docstring in t2's
   `research/regime/config.py` (comment-only, no code dependency).

**Conclusion:** keep `trading_v3/configs/plugins/hmm_regime.yaml` as the single
canonical sample; do NOT duplicate into `research/configs/plugins/`. Any
runtime/OOS loading (t5) must pass this path explicitly to
`load_plugin_config` / `load_hmm_regime_config`.

---

## 2. Activation contract (default OFF)

```
plugin.enabled (hmm_regime.yaml)          ← master switch
   AND
pattern config lists hmm_regime            ← features.optional_plugins
───────────────────────────────────────────────────────
   → RegimeSlot built (plugin attached)
      • emitter runs when hmm.enabled (resolved emitter_enabled)
      • hard gate runs when regime_filter.enabled AND a rule exists
No listing / master off / no plugin        → slot None → legacy behaviour
```

A legacy assignment (no plugin, no rules) resolves to `regime=None` and is
byte-identical: the golden LSW suite (15/15) and the full existing test suite
stay green.

---

## 3. No-lookahead guarantees preserved

- State lookup is at the **last closed bar ≤ event `known_at`** (preferring
  the detector's `confirm_bar` attribute — DB/DT store it): never a future
  bar (requirements §3.4 / guide §2).
- The plugin itself is the t2 `CausalGaussianHMM` (forward-filter only); the
  integration only consumes already-computed states.
- Feature schema (`uses_future_data=False`, `available_at='confirm'`) is t2's;
  integration tests assert the injected state stamps are ≤ `known_at`.

---

## 4. Verification (t3)

| Check | Result |
|---|---|
| `pytest tests/test_hmm_regime_integration.py tests/test_model_registry.py tests/test_gui_onboarding_registry.py -q` | 47 passed |
| `pytest tests/test_liquidity_sweep_golden.py -q` | 15 passed (bit-identical, 163s) |
| `ruff check live/engine live/state model_registry research/multi_backtest tests/test_hmm_regime_integration.py` | clean |
| `mypy --strict live/engine/feature_emitter.py live/state/shared_app_state_v2.py` | clean |
| Full `tests/` suite (minus heavy golden) | see t3 report |

Integration tests cover: emitter on/off, causal stamping, engine injection,
byte-identical legacy (scorer + engine), registry §4.3 round-trip + auto-
resolve, hard gate reasons + engine/runner application + shared-function
identity, fail-closed §6.3, Event Lake regime round-trip, config templates.

---

## 5. Out of scope / hand-off notes

- `walkforward_trainer.build_feature_frame` was **not** modified (out of t3
  scope): new-HMM-model scoring sources `hmm_*` from attributes inside
  `make_tier2_scorer` (runner) instead.  Training a new HMM model (t5/OOS,
  optional §8.5) may extend the trainer within its own task.
- Frozen files untouched: `research/core/contracts.py`, `causal_checks.py`,
  `config_hash.py`, `research/patterns/liquidity_sweep/src/`.
- Hard-gate rule thresholds are **samples** chosen a priori (requirements
  §8.2: tuning only on the train prefix) — the OOS script (t5) is responsible
  for the final LSW/DB/DT verdicts.