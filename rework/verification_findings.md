# Verification Findings — Verified Against Actual Source

**Task:** team `trend-hmm-rework`, task **t1** (requirements r1)
**Author:** `code_verifier` (requirements-analyst)
**Date:** 2026-09-11
**Request under verification:** `rework/trading_v3_rework_request_trend_hmm.md`
**Method:** every claim read directly from source. No source file was modified.
**Interpreter note:** system `python3` (3.9.6) has **no pytest installed** — test
suites below are verified **by reading test source, not by execution**. Any count
presented as "documented" is a claim from a repo doc, not a re-run.

---

## 0. Verdict summary

| Request claim | Verdict | Evidence |
|---|---|---|
| §1.2 `pattern_length` computed + emitted as feature, **no gate uses it** | ✅ **VERIFIED** | `double_bottom/detector.py:100-103` declares it; computed at `:362-363` (`left_len`/`right_len`, but see F-01) ; no gate |
| §1.3 `hard_gate.py` has **no** threshold for `model_prob`/`rule_score`/`combined_score` | ✅ **VERIFIED** | `hard_gate.py` is 155 lines; only `allowed_states` (`:66`) + `min_confidence` (`:68-70`) |
| §1.3 premise that **nothing** gates low probability in live | ⚠️ **PARTIAL / MISLEADING** | a `min_model_prob` gate **already exists** at `signal_engine_v2.py:1145-1152`, but it is **OFF by default** (`_DEFAULT_MIN_MODEL_PROB` empty at `:75`) |
| §1.4 HMM classifies **volatility, not trend direction** | ✅ **VERIFIED** | `DEFAULT_STATE_NAMES = ["trending","sideways","high_vol"]` at `gaussian_hmm.py:62`; `input_features` are `log_return_1, atr_14_norm, volume_zscore_20` at `:55-59` — no directional feature |
| §1.4 / §2.4 regime output is `volatility_ratio` | ❌ **WRONG** | `volatility_ratio` does **not exist** in any regime/engine `.py`. Only 3 hits repo-wide, all in **markdown** (`rework/trading_v3_rework_request_trend_hmm.md:20,46,127`; `rework/trading_v3_pattern_rework_spec.md:266,268,270`). See F-02 |
| §1.4 HMM infra is live and used for dynamic `stop_buffer_atr` | ❌ **WRONG — it is INERT** | `configs/plugins/hmm_regime.yaml:20` `enabled: false`, `:36` `regime_filter.enabled: false`; live symbol config has **no** regime section at all (F-03) |
| §2.1 line locations in base class | ✅ verified, exact lines given below (F-04) |
| §2.2 `max_pattern_length_bars` is cheap + independent | ✅ **VERIFIED** (with a naming trap — F-01) |
| §2.3 `PROB_THRESHOLDS` "copied from bug_summary B1, needs re-confirm" | ✅ **VERIFIED — and it does NOT match**: B1 shipped as an **opt-in** gate, not a hardcoded per-pattern map |
| §3 `stop_mode="neckline"` + 4 ATR cap are measured-harmful | ✅ **VERIFIED** | `docs/pattern_rework_resolution.md:25-26` |
| §4.3 "168 no_lookahead tests" | ⚠️ **DOCUMENTED, NOT RE-RUN** — from `docs/handoff.md:13,87`, `docs/qa_checklist.md:158`. Cannot execute (no pytest) |
| §4.3 "golden LSW 15/15" | ⚠️ **DOCUMENTED** — `docs/handoff.md:13`. Test file present at `tests/test_liquidity_sweep_golden.py` |

**Overall:** the two *core* findings (missing max-length gate; no score gate in
`hard_gate.py`) are **real**. But the request's **regime premise is materially
wrong** and would send the implementation in the wrong direction. Read F-02 and
F-03 before writing any 2.4 code.

---

## 1. F-classified findings (things that would cause a wrong implementation)

### F-01 — `pattern_length` is NOT stored in DB/DT event attributes (BLOCKER for a naive 2.2)

The request (§1.2, §2.2) says: *"`pattern_length` đã được tính và emit như một
feature... Cần team verify vị trí feature `pattern_length` được tính, để chèn gate
ngay cạnh."*

Reality — two different things share the name:

1. **Declared feature schema**: `PatternFeature("pattern_length", "int", AVAILABLE_AT_DETECT, False, ...)` at `research/patterns/double_bottom/detector.py:100-103`.
2. **What is actually computed** in `_detect_double`: `left_len = i2 - i1` (`:362`) and `right_len = i3 - i2` (`:363`), with `symmetry` derived at `:364`. The **span between the two extremes** used by §2.2 (`extreme2_bar - extreme1_bar`) is **never assigned to a variable and never written to `attributes`**.

The `attributes` dict at `:391-407` contains `extreme1_bar`, `neckline_bar`,
`extreme2_bar`, `depth_atr`, `low_offset_atr`, `confirm_range_atr`, `left_len`,
`right_len`, `target_capped`, `realized_rr`, `discard_reason` — **there is no
`pattern_length` key.**

Consequences if §2.2 is taken literally:
- `PatternEvent.attributes["pattern_length"]` is **absent** for DB and DT.
  Any consumer (live engine, Event Lake export, `rework_measure.py`, DebugDrawer
  `export_debug_events.py:18`) that reads it gets `None`/missing.
- The declared `feature_schema` entry is therefore **already a lie** — it
  advertises an `AVAILABLE_AT_DETECT` feature the detector never populates. Adding
  a gate beside "where it is computed" is impossible; the value must first be
  **computed and stored**.
- Contrast with a pattern that *does* store it: `research/patterns/head_shoulders/detector.py:337` writes `"pattern_length": int(i5 - i1)` into `attributes`. So the convention exists elsewhere — DB/DT simply never adopted it.

**Required fix before implementing 2.2:** compute
`pattern_length = i3 - i1` inside `_detect_double` and add it to the
`attributes` dict at `:391` alongside the other detect-time features, then gate on
it. Do not assume the key exists.

---

### F-02 — `volatility_ratio` DOES NOT EXIST (WRONG claim; §2.4 step 1 is based on a phantom API)

Request §1.4: *"chỉ output `volatility_ratio` (dùng cho `stop_buffer_atr` dynamic)"*;
§2.4 step 1: *"output hiện có (`volatility_ratio`)"*.

- Repo-wide grep for `volatility_ratio` across all `*.py`: **zero matches.**
- Only 6 matches exist repo-wide and **all 6 are inside markdown documents**:
  `rework/trading_v3_rework_request_trend_hmm.md:20`, `:46`, `:127` and
  `rework/trading_v3_pattern_rework_spec.md:266`, `:268`, `:270`.
- The spec's `stop_buffer_atr(volatility_ratio)` (spec `:266-270`) is a
  **prose code sketch from an unbuilt proposal**. It was explicitly **NOT
  implemented**: `docs/pattern_rework_resolution.md:29` records
  §2.3.5 "dynamic stop buffer" as **⏸️ KHÔNG LÀM** (not done), with the reason
  that it depends on §2.3.1, which was measured harmful.

**What the emitter REALLY exposes** — the complete `hmm_*` surface, from
`live/engine/feature_emitter.py` and `research/regime/base.py:120-128`:

| Attribute written | Source | Line |
|---|---|---|
| `hmm_state` (int) | `attrs["hmm_state"] = int(reg.state)` | `feature_emitter.py:172` |
| `hmm_prob_<name>` (float, per configured state) | `attrs[f"hmm_prob_{state_name}"] = float(prob)` | `feature_emitter.py:173-174` |
| `hmm_confidence` (float) | `attrs["hmm_confidence"] = float(reg.confidence)` | `feature_emitter.py:175` |
| `hmm_known_at` (str, lineage) | `attrs["hmm_known_at"] = str(reg.timestamp)` | `feature_emitter.py:178` |
| `hmm_config_hash` (str, lineage) | `attrs["hmm_config_hash"] = str(reg.config_hash)` | `feature_emitter.py:179` |

`BaseRegimePlugin.get_feature_names` (`base.py:116-128`) derives names dynamically
from `state_names`: `["hmm_state"] + [f"hmm_prob_{n}" for n in state_names] + ["hmm_confidence"]`.
So the emitted set is **exactly** `hmm_state`, `hmm_prob_trending`,
`hmm_prob_sideways`, `hmm_prob_high_vol`, `hmm_confidence` under the default config.

There is **no** `volatility_ratio`, no ATR-fast/ATR-slow ratio, and no
`stop_buffer_atr` consumer anywhere. §2.4 must be re-planned against the real API.

---

### F-03 — The HMM regime is currently **INERT** in the live path (contradicts "đã có hạ tầng ... dùng cho")

Request §1.4 and §0 claim the system "**đã có hạ tầng HMM**" wired for volatility.
The infrastructure exists but is **switched off, and not even reachable from the
live entry point**:

1. **Plugin master switch is false.**
   `configs/plugins/hmm_regime.yaml:20` → `enabled: false`
   `configs/plugins/hmm_regime.yaml:36` → `regime_filter.enabled: false`
   `file_emitter` activation requires both: `hmm.enabled: true` at `:23` is gated
   behind `plugin.enabled` via `regime_wiring.py:164`
   (`"emitter_enabled": plugin_enabled and bool(hmm.get("enabled", False))`).
   → resolves to **False**.

2. **The live symbol config has no regime section whatsoever.**
   The live config dir is `research/configs/symbols/` — defined by
   `SYMBOL_CONFIG_DIR` in `live/state/shared_app_state_v2.py:39`
   (`.../ "research" / "configs" / "symbols"`).
   The live `XAUUSD.yaml` (92 lines, read in full) contains **no**
   `regime`, `hmm`, `optional_plugins`, or `model_id` key.
   Grep for `regime|hmm|optional_plugins|plugins` in
   `research/configs/symbols/XAUUSD.yaml`: **zero matches**.
   The regime-aware template `configs/symbols/_hmm_template.yaml` lives in a
   **different, unused directory** (`configs/symbols/`, note: `configs/` not
   `research/configs/`).

3. **`XAUUSD.yaml:74` `model_path: ""` and no `model_id`.**
   `create_symbol_engine` reads `model_id` at `signal_engine_v2.py:615` and
   **raises `SymbolNotValidatedError`** at `:658` when neither `model_id` nor a
   model override is present. The live engine's fallback is
   `_config_override` (`signal_polling_engine_v2.py:198-208`), which only
   supplies `status`/`model_id`/`name` from the registry — **never a regime
   plugin**.

4. **The live polling path never constructs a `MultiPatternEngine`.**
   `signal_polling_engine_v2.py` `_create_engine_for` (`:210-230`) calls
   `create_symbol_engine` (`signal_engine_v2.py:552`), which returns the **legacy
   single-pattern LSW closure** `check_new_bar` (`:676-679`). The entire HMM
   regime path lives only in `MultiPatternEngine` (`signal_engine_v2.py:~800+`,
   slots resolved at `:838-915`, gated at `:1021-1043`).
   Grep for `regime|hmm|plugin|MultiPatternEngine` in
   `signal_polling_engine_v2.py`: **zero matches**.

5. **`regime_plugin` is only ever supplied by tests and the research runner.**
   Every non-test call site of `MultiPatternEngine(...` is
   `live/smoke/mcp_live_smoke.py:333`; all others are in `tests/` or
   `research/multi_backtest/runner.py`.

**Conclusion:** §2.4's framing ("extend the existing, proven pipeline") is
**partially false**. The pipeline is not proven-in-live; it has never run in the
live path at all. Enabling 2.4 means also (a) enabling the plugin master switch,
(b) adding a `regime`/`optional_plugins` block to the *correct* symbol config dir,
(c) migrating the live path onto `MultiPatternEngine`, or (d) wiring the regime
into the legacy closure. This is a **much larger** scope than §5.5's "2–3d".

---

### F-04 — §2.3 `PROB_THRESHOLDS` "copy from B1" is stale: B1 shipped as an opt-in, not a hardcoded map

Request §6 bullet 3 asks to confirm whether the §2.3 `PROB_THRESHOLDS` map
(`{"double_bottom": 0.55, ...}`) is still accurate "from bug_summary B1".

Reality: **B1 was implemented as an opt-in gate with an EMPTY default map.**

- `signal_engine_v2.py:72-75`:
  `_DEFAULT_MIN_MODEL_PROB: dict[str, float] = {}` — comment at `:72-74`:
  *"Empty by default = no probability gate (historical behaviour)."*
- `_resolve_min_model_prob` (`:1302-1321`) reads `a.config["min_model_prob"]`,
  falls back to the module map, returns `None` → **gate does not run**.
- The gate body is `:1145-1152`; it writes `discard_reason = "low_probability"` (`:1147`).
- `docs/bug_summary_resolution.md:17` confirms: *"gate `min_model_prob` ... mặc định **tắt** (không đổi hành vi cũ)"* (default off, no behaviour change).
- `docs/bug_summary_resolution.md:178` explicitly leaves it as an **open user decision**: *"B1 — có muốn bật gate không? ... Hiện tại đang tắt."*

Also note the request's §2.3 code sketch checks
`model_prob < threshold or rule_score < RULE_SCORE_FLOOR` — but **no `rule_score`
floor exists anywhere** today. B1 is model-prob-only. The request's proposed
`RULE_SCORE_FLOOR = 0.6` would be **new** behaviour, not a re-confirmation of B1.

**Threshold-scale trap:** `rule_score` is **not on one scale.**
`_RULE_SCORE_SCALE` (`signal_engine_v2.py:70`) declares
`{"liquidity_sweep": 100.0}`; `_normalize_rule_score` (`:1324-1346`) infers
`100.0` when `abs(v) > 1.0` else `1.0`. DB/DT clamp to `[0,1]`
(`double_bottom/detector.py:376-382`), LSW emits `0..100`.
A `RULE_SCORE_FLOOR = 0.6` applied to **raw** `rule_score` would **destroy every
liquidity_sweep event** (their raw scores are ~32–51, all `> 0.6`) — or,
inverted, would be a no-op. The floor MUST be applied to the **normalized** score,
or made per-pattern. The request's pseudo-code does not say which.

---

### F-05 — `detect()` config-merge semantics interact with any new gate default

`DoublePatternDetectorBase.__init__` (`:114-120`) and `detect` (`:178-182`):
```python
cfg = self.config
if config: cfg = {**self.config, **config, "version": self.version}
```
`self.config` is `get_default_config()` merged with the constructor arg.
A new key added to `get_default_config()` (e.g. `max_pattern_length_bars`,
`min_rule_score`, `trend_context_enabled`) changes **every** config dict, which
changes `compute_config_hash(cfg)` (`:222`, and `research/core/config_hash.py:50`).
That is expected and required (§5.7 re-run of the config_hash gate) — but it means
the **golden LSW** and **DB/DT dataset regenerations are mandatory**, and the
existing pinned `train_summary.json` / `features.json` artifacts will mismatch.
Plan for artifact regeneration, not just a code edit.

Additionally, `get_default_config()` MUST keep the `"version"` key or
`compute_config_hash` raises (`research/core/config_hash.py:50` docstring +
`tests/test_core_contracts.py:260-263`). Bump `FEATURE_SCHEMA_VERSION`
(`double_bottom/detector.py:47`, currently `"double-v1.0"`) only on a genuine
**feature** semantic change — §2.2/§2.3(add-a-gate) do not add features; §2.4
adding `trend_regime_state` as an attribute used by the model **would**.
Note `research/patterns/double_bottom/scripts/train_walkforward.py:80,86` hardcodes
`"feature_schema_version": "double-v1.0"` — it must be updated in lockstep.

---

## 2. Exact insertion points (request §6, bullet 1)

File: `research/patterns/double_bottom/detector.py`
Class: `DoublePatternDetectorBase` (`:66`)
Method: `_detect_double` (`:187`)

| Anchor | Exact line(s) | Notes |
|---|---|---|
| **Swing triple loop** | `for k in range(len(sw) - 2):` at `:225`; triple unpacked at `:226` (`s1, s2, s3 = sw[k], sw[k+1], sw[k+2]`) | This is the loop §2.1/§2.2 gate on |
| Kind-sequence filter | `:227-228` | first `continue` |
| Ordering guard | `:232-233` (`if not (i1 < i2 < i3): continue`) | guarantees `i3 - i1 > 0` |
| **Min-separation gate** | `:234-235` (`if i2 - i1 < min_sep or i3 - i2 < min_sep: continue`) | **2.2 belongs immediately after this** — it is the same class of bar-count gate |
| ATR solved at detect bar | `:236-241` (`atr_k = float(atr[detect_bar])`) | required for any ATR-relative gate |
| Equal-level gate | `:247-248` (bull) / `:254-255` (bear) | |
| **Depth gate** (`min_depth_atr`) | `:257-258` — `if depth < min_depth_atr * atr_k: continue` | the "depth gate" the request §1.1 refers to; **2.1 trend-context** naturally follows `:258` |
| Confirmation scan | `:263-274` | |
| `attributes` dict (add `pattern_length` here) | `:391-407`; literal `"discard_reason": None` at `:406` | |
| `rule_score` formula | `:376-382` | a 2.3 `min_rule_score` gate goes **after** this (the score must exist first) |
| NMS + dedupe | `:431-433` | a gate placed *before* `:431` reduces NMS input |
| **`get_default_config` dict** | `:125-176`; `"min_depth_atr": 3.0` at `:138`; `"nms_overlap": True` at `:169`; closes at `:176` | new config keys belong beside `:137-138` |

**Both DB and DT consume this class — CONFIRMED.**
`research/patterns/double_top/detector.py:24` imports
`DoublePatternDetectorBase`; `:27` `class DoubleTopDetector(DoublePatternDetectorBase)`;
`:34` overrides only `kind_seq = ("H","L","H")`, `:33` `direction = DIRECTION_BEARISH`.
Its module docstring `:6-7` and the base docstring `:69-73` both state double_top
reuses "100 % infra của P1". **No** method is overridden — `_detect_double`,
`get_default_config`, `_nms_structure_overlap`, `_dedupe` are all inherited.
Therefore a gate added to the base applies to **both**, and a gate that must be
direction-aware (2.1) must branch on `bullish = self.direction == DIRECTION_BULLISH`
(computed at `:193`) — a single `continue` cannot serve both directions.

---

## 3. Real API of the live regime layer (request §6, bullet 2)

### `live/engine/feature_emitter.py` (336 lines)

| Symbol | Signature | Line |
|---|---|---|
| `HMM_FEATURE_PREFIX` | `= "hmm_"` | `:42` |
| `state_at_known_at` | `(states: Sequence[RegimeState], df: pd.DataFrame, known_at: pd.Timestamp) -> RegimeState \| None` | `:65-69` |
| `state_at_confirm_bar` | `(states: Sequence[RegimeState], df: pd.DataFrame, ev: PatternEvent) -> RegimeState \| None` | `:91-95` |
| `HMMFeatureEmitter.__init__` | `(plugin: BaseRegimePlugin, config: dict[str,Any] \| None = None)` | `:143-147` |
| `HMMFeatureEmitter.attach` | `(events, states_by_bar, df) -> int` (count attached) | `:152-157` |
| `HMMFeatureEmitter.feature_frame` | `(df, events, states_by_bar=None) -> pd.DataFrame` | `:183-188` |
| `HMMFeatureEmitter.feature_names` | `() -> list[str]` | `:217-218` |
| `regime_flags` | `(config) -> tuple[bool,bool]` → `(emitter_enabled, gate_enabled)` | `:221-228` |
| `requires_hmm_regime` | `(optional_plugins) -> bool` (any `name=="hmm_regime"` with `required: true`) | `:231-237` |
| `lists_hmm_regime` | `(optional_plugins) -> bool` (any entry named `hmm_regime`) | `:240-249` |
| `build_feature_vector` | `(event, model_meta: dict) -> np.ndarray` | `:257-260` |
| `publish_regime_states` | `(lake_root: str\|Path, symbol: str, timeframe: str, states) -> Path` | `:290-295` |
| `read_regime_states` | `(lake_root, symbol, timeframe) -> list[RegimeState]` (`[]` when absent) | `:310-314` |

**What `state_at_known_at` returns:** the `RegimeState` of the **last closed bar
`≤ known_at`**, via `np.searchsorted(times, stamp, side="right") - 1` (`:85`);
`None` when there is no such bar, when `states` is empty, or when the index is not
a `DatetimeIndex` (`:78-82`). Comparison is tz-normalized to UTC-naive via
`_naive_ns`/`_naive_ts` (`:50-62`).

**What `state_at_confirm_bar` returns:** prefers `attributes["confirm_bar"]`
(`:116-126`) — but **only** if the state's timestamp at that bar is `<= ev.known_at_ts`;
otherwise falls back to `state_at_known_at(states, df, ev.known_at_ts)` (`:127`).
The docstring `:103-112` is explicit that the bound is the **derived**
`ev.known_at_ts` property, *never* the raw `ev.known_at` field (which no detector
sets — confirmed: no writer exists; `contracts.py:85` default `None`,
`known_at_ts` derives `max(detect_time, confirm_time)` at `:110-117`).
**This is the exact causal anchor §2.4 step 3 must use** — it already prevents
an attacker-planted future `confirm_bar` from attaching a future regime state.

**How a `RegimeState` reaches the engine (the real chain):**
`configs/plugins/hmm_regime.yaml` → `load_plugin_config` (`regime_wiring.py:128`)
→ `resolve_regime_config` (`:136`) → `resolve_regime_wiring` (`:178`) →
`RegimeSlot` (`:46`), stored per assignment in
`MultiPatternEngine._regime_slots` (`signal_engine_v2.py:853`, populated `:838-915`).
At detect time `_run_assignment` (`:1055`) computes states once and caches them on
the slot (`:1089-1098`: `slot.plugin.predict(df, slot.hmm_config)` →
`slot.set_states(...)`), then `slot.emitter.attach(events, states, df)`.
The gate reads them back via `_regime_slot(aid)` at `:1021` / `:1089`.
**There is no `build_feature_vector` call in the live engine** — it is a
trainer-side contract (`live/model_...`/trainer); live inference consumes
`hmm_*` through the model's own `feature_list`.
**`publish_regime_states` / `read_regime_states` are NOT called by
`signal_engine_v2.py` or `signal_polling_engine_v2.py`** — they are script-side
(the `research/multi_backtest/scripts/oos_hmm_regime_filter.py` harness).

### `live/engine/regime_wiring.py` (218 lines)

| Symbol | Signature | Line |
|---|---|---|
| `HMM_PLUGIN_NAME` | `= "hmm_regime"` | `:43` |
| `RegimeSlot.__init__` | `(plugin, emitter_enabled, gate_enabled, rules, hmm_config, assignment_id, pattern_name, model_feature_list=None, model_optional_plugins=None)` | `:49-60` |
| `RegimeSlot.emitter` | property → `HMMFeatureEmitter` | `:76-78` |
| `RegimeSlot.emitter_active` | `() -> bool` = `emitter_enabled and _list_plugin()` | `:80-81` |
| `RegimeSlot.gate_active` | `() -> bool` = `gate_enabled and _list_plugin() and bool(rules)` | `:86-87` |
| `RegimeSlot.set_states` / `states` | `(states) -> None` / `() -> list[RegimeState] \| None` | `:95-99` |
| `RegimeSlot.model_requires_plugin` | `() -> bool` | `:104-109` |
| `load_plugin_config` | `(path) -> dict` (whole mapping) | `:128-133` |
| `resolve_regime_config` | `(plugin_yaml, hmm, regime_filter, pattern_name) -> dict` | `:136-141` |
| `resolve_regime_wiring` | `(plugin, resolved_config, assignment_id, pattern_name, model_feature_list=None, model_optional_plugins=None) -> RegimeSlot \| None` | `:178-185` |

`resolve_regime_wiring` returns **`None`** unless `resolved_config["enabled"]` is
truthy (`:193-194`) — the default-OFF guarantee. `_list_plugin` (`:111-125`)
returns `lists_hmm_regime(model_optional_plugins)` when metadata is present, else
`bool(emitter_enabled or gate_enabled)`.

---

## 4. `RegimeState` fields and the frozen plugin contract (request §6, bullet 3)

`RegimeState` — `@dataclass` at `research/regime/base.py:32-50`:

| Field | Type | Line |
|---|---|---|
| `timestamp` | `pd.Timestamp` — bar **close** time == `known_at` | `:43` |
| `state` | `int` — argmax filtered posterior | `:44` |
| `state_name` | `str` — `"trending" \| "sideways" \| "high_vol"` | `:45` |
| `state_prob` | `dict[str,float]` — `{"trending": 0.82, ...}` | `:46` |
| `confidence` | `float` — `max(state_prob.values())` | `:47` |
| `lag_bars` | `int` | `:48` |
| `model_version` | `str` | `:49` |
| `config_hash` | `str` — `sha1(canonical_json(hmm_config))[:12]` | `:50` |

`BaseRegimePlugin` ABC (`:53`) abstract methods:
`fit(df, config) -> BaseRegimePlugin` (`:74-79`),
`predict(df, config) -> list[RegimeState]` (`:86-91`),
`get_feature_schema(config) -> list[PatternFeature]` (`:99-103`),
`get_default_config() -> dict` (`:108-114`).
Concrete: `get_feature_names(config) -> list[str]` (`:116-128`), derived from
`state_names`. Class attrs: `name="hmm_regime"` (`:65`), `version="1.0.0"` (`:68`),
`short_key="HMM"` (`:70`), `feature_schema_version` (`:72`) =
`HMM_FEATURE_SCHEMA_VERSION = "hmm-v1.0"` (`:29`).

**Can a plugin emit a trend-direction state without breaking the frozen contract?
— YES, and it is the intended extension point.** Reasons:

1. `state_name` is typed `str` and its docstring (`:45`) already anticipates
   alternatives via `"..."`. Nothing constrains it to the three volatility names.
2. `get_feature_names` (`:116-128`) derives feature names **dynamically** from
   the configured `state_names` — the docstring at `:122-124` explicitly states
   *"a 2-state config yields a different column set than the default 3-state one."*
   So a config with `state_names: ["uptrend","downtrend","range"]` produces
   `hmm_state`, `hmm_prob_uptrend`, `hmm_prob_downtrend`, `hmm_prob_range`,
   `hmm_confidence` **with no code change**.
3. `CausalGaussianHMM._resolve_config` (`gaussian_hmm.py:286-327`) validates only
   `len(state_names) == n_states` and uniqueness (`:293-296`) — arbitrary names pass.
4. `get_feature_schema` (`:615-649`) likewise builds from `state_names`.

**But there are three real constraints §2.4 step 2 must respect:**

- **Input features are a closed vocabulary.** `CANONICAL_INPUT_FEATURES = ("log_return_1","atr_14_norm","volume_zscore_20")` (`gaussian_hmm.py:55-59`); `compute_causal_input_features` (`:102`) and `_validate_input_columns` (`:346`) **raise `ValueError`** on any name that is neither a `df` column nor canonical (`:130-134`, `:350-354`). A "slope" feature must be **pre-supplied as a `df` column** (that path is allowed at `:117-118`) or added as a new canonical feature.
- **`fit` must never see the OOS region** (docstring `:80-84`) — `min_fit_bars=2000` (`:282`).
- **The 3 volatility states and 3 trend states are different model outputs.** Nothing in the current schema supports *both* from one plugin instance/`RegimeState` — `state_name` is a single string. Emitting trend direction requires either a separate plugin instance/config (and therefore a separate `hmm_config` + `config_hash` + persisted artifact) or an additive field on `RegimeState`, which **is** a contract change to `base.py`. §2.4 step 2 ("mở rộng HMM") is not free.

**Recommended minimal-risk shape:** a **second `CausalGaussianHMM` instance** with
`state_names: ["uptrend","downtrend","range"]` and a slope/return input column,
wired as its own slot — leaving the frozen `base.py` contract, the existing
volatility config, and all `config_hash` values untouched. Note the existing
`configs/plugins/hmm_regime.yaml:25` pins `state_names: ["trending","sideways","high_vol"]`.

---

## 5. `discard_reason` convention (request §6, bullet 4)

**Convention:** a plain lowercase snake_case `str`, stored in
`PatternEvent.attributes["discard_reason"]` (declared at
`research/core/contracts.py:100`; Event Lake column at
`research/core/event_lake.py:84,215,274`). Detectors initialise it to `None`
in the attributes dict. **No central enum/constant registry exists** — each
writer uses a bare string literal (except the one exported constant below).

### Complete enumeration of `discard_reason` values currently in use

| Value | Written by | file:line |
|---|---|---|
| `"regime_blocked"` | hard gate | `live/engine/hard_gate.py:141` (via constant `DISCARD_REASON_REGIME_BLOCKED`, defined `:32`) |
| `"regime_blocked"` | multi-pattern engine | `live/engine/signal_engine_v2.py:1042` |
| `"regime_blocked"` | backtest runner (live ≡ backtest parity) | `research/multi_backtest/runner.py:449` |
| `"correlated"` | correlation manager | `live/engine/correlation_manager.py:440` (doc `:386`) |
| `"stale"` | entry-drift guard | `live/engine/signal_engine_v2.py:1011` |
| `"hmm_unavailable"` | emitter failure (fail-closed) | `live/engine/signal_engine_v2.py:1024`, `:1109` |
| `"hmm_unavailable"` | backtest runner | `research/multi_backtest/runner.py:418`, `:504` |
| `"low_probability"` | B1 model-prob gate | `live/engine/signal_engine_v2.py:1147` |
| `None` (unset placeholder) | detectors | `double_bottom/detector.py:406`, `head_shoulders/detector.py:338`, `rising_wedge/detector.py:392` |
| `"threshold"`, `"dedup"` | **test fixtures only** | `tests/test_event_lake.py:183`, `:200` |

Also returned as **internal reason codes** by `hard_gate.is_allowed` (not written
to `attributes`): `"no_rule"`, `"allowed"`, `"state_not_allowed"`,
`"low_confidence"`, `"no_regime"` — `hard_gate.py:35-39`.

**Verdict on the three proposed reasons:**
- `"low_rule_score"` → ✅ matches convention (lowercase snake_case).
- `"pattern_too_long"` → ✅ matches convention.
- `"low_trend_context"` → ✅ matches convention.
- ⚠️ **Naming inconsistency to resolve:** `"low_probability"` (B1, existing) vs
  the request's proposed `"low_model_prob"` (§2.3 code sketch at request `:118`).
  Using `low_model_prob` would create **two names for the same concept**. Prefer
  reusing `"low_probability"`.

**Fail-closed convention §2.3 relies on — confirmed:**
`hard_gate.py:63-64` returns `(False, "no_regime")` when a rule exists but
`regime is None` (docstring `:52-54`: *"fail-closed so 'gate on but plugin off'
never silently lets everything through"*). This is the pattern to copy.

---

## 6. Model artifacts and trainer location (request §6, bullet 6)

| Artifact | Path |
|---|---|
| **double_bottom** model dir | `research/patterns/double_bottom/artifacts/models/double_bottom_xauusd_m15_v1/` |
| ↳ files | `model.pkl`, `calibrator.pkl`, `features.json`, `feature_schema.json`, `train_summary.json` |
| **double_top** model dir | `research/patterns/double_top/artifacts/models/double_top_xauusd_m15_v1/` |
| ↳ files | `model.pkl`, `calibrator.pkl`, `features.json`, `feature_schema.json`, `train_summary.json` |
| Other patterns | `falling_wedge/`, `liquidity_sweep/`, `head_shoulders/` each under their own `artifacts/models/` |
| **`train_walkforward.py`** | `research/patterns/double_bottom/scripts/train_walkforward.py` (only one in the repo) |
| Registry index (authoritative `model_path`) | `model_registry/index.yaml` — DB `:68-96` (`model_path` `:81`), DT `:111` |

The registry `index.yaml:30-44` documents the `optional_plugins` metadata field
(§4.3), and `:37-44` shows a **commented-out** `double_bottom_xauusd_m15_v1_hmm`
entry whose `feature_schema_version` is **`"double-v1.1"`** (`:42`) with
`feature_list` already listing `hmm_state, hmm_prob_trending, hmm_prob_sideways,
hmm_prob_high_vol, hmm_confidence` (`:43`) and
`optional_plugins: [{name: hmm_regime, version: "1.0.0", required: true}]` (`:44-45`).
**So the §2.4 step-5 `double-v1.0` → `double-v1.1` bump was already designed and
pinned in the registry** — reuse that exact version string. But the entry is
**commented out and the model was never created**, which matches F-03: no live
model requires or lists the regime plugin.

Trainer schema hardcodes: `train_walkforward.py:80,86` →
`"feature_schema_version": "double-v1.0"`; `:92` → `"wedge-v1.0"`.
A `double-v1.1` bump must edit these too.

---

## 7. §3 "do not re-apply" — CONFIRMED consistent with the repo

`docs/pattern_rework_resolution.md` (read in full, 60+ lines) is unambiguous:

- `:25` — **§2.3.1 structure stop (neckline)** → *"⚠️ CÀI NHƯNG TẮT MẶC ĐỊNH — **Đo được: làm KẾT QUẢ XẤU ĐI** (winrate 48%→23%)"*
- `:26` — **§2.3.2 capped target + min_rr** → *"⚠️ CÀI NHƯNG TẮT MẶC ĐỊNH — Cặp 4 ATR cap + min_rr=1.0 **xoá 96% event**"*
- `:35` — *"**Sai.** SL gần hơn 5,2× (4,56→0,87 ATR) nhưng winrate **48%→23%**, expectancy **−0,01R→−0,47R**"*
- `:36` — *"**Sai hoàn toàn.** Cặp này xoá **96% event** (DB 354→15)"*
- `:29` — §2.3.5 dynamic stop buffer → **⏸️ KHÔNG LÀM** (this is what would have consumed `volatility_ratio` — see F-02)
- `:31-32` — one-sentence summary: the two SL/TP proposals were *"đã được đo và chứng minh là phản tác dụng"* (measured and proven counter-productive)

This is independently corroborated in-code by the detector's own defaults and
comments: `stop_mode: "legacy"` (`double_bottom/detector.py:163`, with the
measured 0.48→0.23 note at `:156-162`), `structure_target_atr: 0.0` (`:154`),
`target_cap_atr: 0.0` (`:155`), and the 96%/354→15 note at `:141-153` and `:323-337`.
Measurement source: `docs/rework_measure.py` → `docs/rework_measure_baseline.json`,
`docs/rework_measure_after.json`.

**§3 of the request is fully consistent with the repository. No action needed;
do not re-enable these.**

---

## 8. Measurement harness reality (request §4.1)

`docs/rework_measure.py` exists (240 lines) and **does** measure on the full
XAUUSD M15 history:
- `SYMBOL = "XAUUSD"`, `TIMEFRAME = "M15"` (`:44-45`)
- reads via `load_symbol_frame` + `live.engine.pattern_registry.get_registry` (`:37-42`)
- outputs `n_bars` (`:88`) — the 204k-bar claim is testable through this script
- `_span(ev)` (`:62-83`) derives a structure window generically: `extreme1_bar`/`extreme2_bar` for DB/DT, `low1_bar`/`low3_bar` for wedges, `left_shoulder_bar` for H&S. **It does not read `pattern_length`** — consistent with F-01.
- measured keys: `overlap_rate`, `opposite_rate`, `target_atr`, `risk_atr`, `rr`, `winrate`, `expectancy`, `events_per_year` (`:10-18`, `:88-89`)

**Gap for §4.2:** the script has **no OOS-split / sample-size-counting logic** and
**no** helper for "events remaining AFTER gates 2.1+2.2". The ≥100-after-gates
requirement is **new work** for t6, not a re-run of an existing measurement.

---

## 9. Answers to the four §6 confirmation bullets (direct)

1. **Where to insert 2.1 / 2.2 in the DB/DT base class** → §2 above. Loop `:225`,
   sep gate `:234-235` (2.2 goes here), depth gate `:257-258` (2.1 goes here),
   config dict `:125-176`, attributes dict `:391-407`. Both DB and DT inherit
   (`double_top/detector.py:24,27,34`).
2. **API of `feature_emitter.py` / `regime_wiring.py`** → §3 above. Inputs are
   `log_return_1`/`atr_14_norm`/`volume_zscore_20`; **output is NOT
   `volatility_ratio`** — it is `hmm_state`/`hmm_prob_<state>`/`hmm_confidence`
   (+ `hmm_known_at`/`hmm_config_hash` lineage). No refactor is needed to *read*
   states; a refactor is needed to *train* a directional model.
3. **`PROB_THRESHOLDS` still accurate?** → **NO.** B1 shipped as an **opt-in gate
   with an empty default map** (`signal_engine_v2.py:75`), not the hardcoded
   per-pattern map in §2.3. There is no `rule_score` floor at all. Also
   `rule_score` has **two scales** (LSW 0..100, DB/DT 0..1) — see F-04.
4. **`discard_reason` format** → §5 above; all three proposed strings match the
   convention. Prefer `low_probability` (existing) over the newly proposed
   `low_model_prob` to avoid two names for one concept.

---

## 10. Recommended corrections to the request before implementation

1. **Rewrite §1.4/§2.4.** Delete every `volatility_ratio` reference (F-02). State
   the real output surface (§3 table). State plainly that the regime plugin is
   **currently inert in live** and not reachable from `signal_polling_engine_v2`
   (F-03) — so 2.4 is not "extend a proven pipeline", it is "build and wire a
   second model, then migrate the live path".
2. **Do not implement 2.4 as the gate for 2.1 in this sprint.** §4.5/§5.3 (Option
   B, regression slope + R²) is the only path with realistic scope. Note the
   request's Option-C justification ("HMM đã kiểm chứng") is **false** per F-03.
3. **Implement 2.2 with an explicit `pattern_length = i3 - i1`** written into
   `attributes` (F-01). Do not rely on the declared-but-unpopulated
   `feature_schema` entry.
4. **Decide the `rule_score` scale explicitly** before writing the 2.3 floor
   (F-04) — normalized or per-pattern, never raw.
5. **Reuse `discard_reason = "low_probability"`** rather than inventing
   `"low_model_prob"`.
6. **Re-scope the measurement task (t6)** to add OOS-split + after-gates event
   counting, which `rework_measure.py` does not do (§8).
7. **Plan artifact regeneration**, not just code edits: any new
   `get_default_config()` key changes `config_hash` for every DB/DT event (F-05).

---

## 11. Explicitly NOT verified (honesty section)

- **`pytest -q -m no_lookahead` = 168** — **not executed**; no pytest in the
  system interpreter. Value taken from `docs/handoff.md:13,87` and
  `docs/qa_checklist.md:158`. Test files exist: `tests/no_lookahead_base.py`,
  `tests/test_adversarial_lookahead.py`.
- **golden LSW 15/15** — **not executed**; documented at `docs/handoff.md:13`.
  Test file exists: `tests/test_liquidity_sweep_golden.py` (asserts bit-identity
  vs the legacy reference at `:296-330`, and `>= 100` events at `:336`). The
  "15" is a documented count of golden assertions/tests, not a number I read in
  the file itself.
- **204,133 bars** — taken from `docs/pattern_rework_resolution.md:6`
  (`2018-06 → 2026-09, 204.133 nến`). The script prints `n_bars`
  (`rework_measure.py:88`) but I did not run it.
- No source file was modified, per task scope.
