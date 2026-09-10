# Canonical Schemas, Naming Conventions & Data Contracts

> **Owner:** Agent 0 (Project Lead / Integrator). This document is the single
> source of truth for schema and naming conventions across the whole project.
> Per coordination rule 30.1 any schema change **must**:
> 1. update this file,
> 2. update `src/schema.py` constants,
> 3. notify the Project Lead,
> 4. update tests,
> 5. update all downstream modules,
> 6. be logged in `CHANGELOG.md`.
>
> Schema version: `1.1` (v1.0 locked at Sprint 1 / t1; v1.1 = Phase 4 level
> registry extensions, locked at t8 — see `CHANGELOG.md`).

---

## 1. Global naming conventions

| Item | Convention | Example |
|---|---|---|
| Column names | `snake_case`, ASCII lowercase | `penetration_atr`, `event_high`, `level_price` |
| DataFrames | singular nouns, no prefix | `levels`, `events`, `features` |
| Timestamp columns | `timestamp` (index of candle frames), `known_at`, `origin_time`, `event_time`, `confirmation_time` | — |
| Timestamp format | `datetime64[ns, UTC]`, tz-aware, never naive | `2022-06-02 09:15:00+00:00` |
| Price columns | `float64` | — |
| Level ID | `{type}_{direction}_{origin_pos}` where type ∈ `rolling\|swing\|equal\|prev_day`; H1 swing rows prefixed `h1_` (`h1_swing_low_1234`) | `swing_low_145` |
| Event ID | `{direction}_{index}` (direction ∈ `long\|short`) | `long_2341` |
| Direction (levels) | `low` = support/swept-low, `high` = resistance/swept-high | — |
| Direction (events) | `long` (bullish sweep / buy setup), `short` (bearish sweep / sell setup) | — |
| Booleans | Python `bool` / numpy `bool_` (never `int` 0/1) | `is_confirmed` |
| Unknown/missing | `NaN` (float columns), `None` (object), or omitted row — never empty string | — |
| Volume | tick volume from MT5 `TICKVOL` promoted to `volume`; raw extra columns kept as `tick_volume`, `real_volume` | — |

**Causality rule applied to every column:** any column whose value at time `t`
depends only on data strictly before or at `t` (inclusive of the bar itself
*after close*) is causal. Any column that would change when future rows are
removed is forbidden. Every level row carries `known_at`; every feature column
carries `available_at` semantics; no centered windows, no `shift(negative)`.

---

## 2. Config schema (`configs/*.yaml`)

Merged by `src/config.py::load_config` (deep merge, `baseline.yaml` +
overrides). Section keys are fixed; unknown top-level sections are allowed
(forward-compat) but future merges must keep documented keys.

| Section | Required keys | Types |
|---|---|---|
| `project` | `symbol`, `timeframe`, `timezone`, `random_seed` | str, str, str, int |
| `data` | `input_path`, `output_path`, `timestamp_column`, `volume_column`, `source_format`, `volume_kind` | str |
| `indicators` | `atr_period`, `volume_zscore_window` | int, int |
| `sessions` (optional) | `timezone`; `asia`/`london`/`new_york` hour ranges | str, list[int] (half-open `[start, end)`) |
| `liquidity` | `rolling_lookback`, `enable_rolling`; optional `swing`, `equal_levels` sub-maps | int, bool, dict |
| `sweep` | `min_penetration_atr`, `max_penetration_atr`, `min_wick_ratio`, `min_reclaim_atr`, `cooldown_bars`, `group_rule` | float, float, float, float, int, str ∈ `first\|deepest_penetration\|strongest_reclaim` |
| `confirmation` | `enabled`, `max_wait_bars`, `require_break_sweep_extreme`, `min_body_ratio`, `min_range_atr` | bool, int, bool, float, float |
| `entry` | `mode`, `slippage_price`, `spread_price` | str ∈ `next_open_after_confirmation\|next_open_after_sweep`, float, float |
| `stop` | `mode`, `buffer_atr` | str ∈ `sweep_extreme`, float |
| `labeling` | `horizons` (sorted ints), `reward_r_values`, `same_bar_policy`, `time_barrier_result` | list[int], list[float], str ∈ `ambiguous\|conservative\|optimistic`, str ∈ `mark_to_market\|zero` |
| `costs` | `half_spread_price`, `slippage_price`, `commission_r` | float, float, float |
| `splitting` | `train_fraction`, `validation_fraction`, `test_fraction`, `embargo_bars`, `purge` | float, float, float, int, bool |
| `model` | `target`, `probability_calibration`, `models` | str, str ∈ `isotonic\|sigmoid\|none`, list[str] |
| `scoring` | `score_buckets` | list[list[int]] |

Validation is enforced by `src/config_schema.py::validate_config(cfg)`, invoked
from `load_config(validate=True)` (default). Invalid type, missing required
key, or out-of-range value raises `ConfigError` with the offending path.

---

## 3. Processed OHLCV frame (Parquet, `data/processed/*.parquet`)

Index: `timestamp` (`datetime64[ns, UTC]`, sorted ascending, unique).
Columns (in order): `open`, `high`, `low`, `close`, `volume` — all `float64`.
Optional preserved columns: `tick_volume`, `real_volume`, `spread` (float64).

Invariants enforced by `src/data/validator.py::validate_ohlcv`:

- no missing OHLC values;
- `high >= max(open, close)`, `low <= min(open, close)`, `high >= low`;
- monotonic, unique timestamps;
- the frame may contain `atr`, `true_range`, and liquidity columns added
  downstream — those are *not* part of the processed schema and must be added
  by the respective stage modules.

---

## 4. Liquidity levels table (long format, in-memory / `data/interim/levels.parquet`)

| Column | dtype | Notes |
|---|---|---|
| `level_id` | str | `{type}_{direction}_{origin_pos}` unique |
| `level_type` | str | `rolling` \| `swing` \| `equal` \| `prev_day` |
| `direction` | str | `low` = support, `high` = resistance |
| `price` | float64 | canonical level price |
| `price_min` | float64 | cluster min (equal levels only, else = price) |
| `price_max` | float64 | cluster max (equal levels only, else = price) |
| `origin_pos` | int | bar position (row index) where level last formed |
| `origin_time` | datetime64[ns, UTC] | bar time of `origin_pos` |
| `known_at` | datetime64[ns, UTC] | earliest timestamp at which the level is usable (causal): swing = `t + right_bars`, equal = completing touch, `prev_day` = first bar of the next trading day, H1 swing = first M15 bar at/after the confirming H1 candle's close |
| `status` | str | `active` \| `expired` (locked enum; `expired` whenever `sweep_state != active`) |
| `touch_count` | int | cumulative touches since origin (Phase 4 — supersedes v1.0's "swing/rolling: 1"): equal levels seed the exact cluster positions, swing/H1 swings include the pivot touch, `prev_day` counts from `known_at` |
| `first_touch_time` / `last_touch_time` | datetime64[ns, UTC] | first/last touch (all level types; `prev_day` = NaT until first touch) |

Sorting: ascending `known_at`. **Causality contract:** a level may be used by
the sweep detector only for bars with `timestamp >= known_at`. Rolling levels
are the special case with `known_at == origin_time` and are stored as columns
(`liq_low`/`liq_high`) on the candle frame rather than rows — the detector
consumes either form through a single registry interface.

---

## 4.1. Level registry extensions (Phase 4 / schema lock v1.1)

`src/liquidity/level_registry.py::build_liquidity_levels(df, config)` emits
exactly `LEVEL_COLUMNS` followed by these documented extension columns
(constants in `src/schema.py`: `LEVEL_EXTENSION_COLUMNS` and
`LEVEL_REGISTRY_COLUMNS`). `touch_positions` is an internal column consumed by
the registry (exact cluster membership for equal levels) and may be dropped by
downstream serialisation.

| Column | dtype | Notes |
|---|---|---|
| `known_pos` | int | bar position of `known_at` — the causal usability gate (usable only for `bar_pos >= known_pos`) |
| `age_bars` | int | bars since `origin_pos`, as of the final bar of the input frame |
| `bars_since_last_touch` | float64 | NaN when the level has never been touched (beyond formation) |
| `sweep_state` | str | `active` \| `swept` \| `invalidated` (Phase 4 lifecycle; `swept` is sticky, `invalidated` = age `> max_age_bars` without a sweep; values in `src.schema.SWEEP_STATES`) |
| `first_swept_at` | datetime64[ns, UTC] | timestamp of first penetration (NaT if never) |
| `invalidated_at` | datetime64[ns, UTC] | timestamp of age-expiry (NaT if never / not reached in-sample) |
| `is_h1` | bool | True for H1 swing levels (`level_type` stays `swing`) |
| `max_age_bars` | int | expiry policy applied to this level (swing/H1/equal from config; `prev_day` default 200) |
| `touch_tolerance_atr` | float64 | per-bar touch-tolerance multiplier (equal levels: clustering `tolerance_atr`; all others 0.0 = exact price) |
| `equal_dispersion_atr` | float64 | `(price_max - price_min)/ATR[known_pos]` (equal levels only, else NaN) |
| `formation_atr_tolerance` | float64 | ATR-scaled cluster tolerance at formation, `touch_tolerance_atr × ATR[known_pos]` (equal levels only, guide §9.3) |
| `touch_positions` | object (list[int]) | internal: exact touch bar positions (equal levels; else None) |

The detector tables (`swing_levels.detect_swing_levels`,
`swing_levels.detect_h1_swing_levels`, `equal_levels.detect_equal_levels`,
`level_registry.detect_previous_day_levels`) carry `LEVEL_COLUMNS` +
`known_pos`/`is_h1`/`max_age_bars` — the registry is the consuming contract.

---

## 4.2. Per-bar causal level state (`level_state_at_bar`)

`src/liquidity/level_registry.py::level_state_at_bar(registry, df, atr, bar_pos)`
returns one row per level *known at* `bar_pos` — levels with
`known_pos > bar_pos` are excluded (they do not exist for decisions at that
bar). Every value uses only bars `<= bar_pos`. Columns
(`src.schema.LEVEL_STATE_COLUMNS`):

`level_id, level_type, direction, price, known_at, touch_count,
last_touch_time, age_bars, bars_since_last_touch, sweep_state, status,
first_swept_at, invalidated_at`

This is the no-look-ahead interface the sweep detector (t9) and feature
pipeline (t10) must use for event-time level state.

---

## 5. Event table (in-memory / `data/interim/events.parquet`)

One row per deduplicated sweep event. `src/events/sweep_detector.py::build_sweep_events`
emits exactly this 12-column set (reconciled in rule 30.1; `direction` uses the
sweep-side vocabulary, guide sections 10.1/10.2):

| Column | dtype | Notes |
|---|---|---|
| `event_id` | str | `SWP-XXXXXX`, deterministic ascending ordinal, unique |
| `event_time` | datetime64[ns, UTC] | close time of the sweep candle (= bar timestamp) |
| `direction` | str | `bullish` (swept low → long setup) \| `bearish` (swept high → short setup) |
| `level_id` | str | source level (traceability) |
| `level_price` | float64 | swept level price |
| `event_open` / `event_high` / `event_low` / `event_close` | float64 | OHLC of the sweep candle |
| `penetration_atr` | float64 | `(level_price - sweep_extreme) / atr` — measured at candle close |
| `wick_ratio` | float64 | directional wick / candle range |
| `reclaim_atr` | float64 | ATR-normalized close reclaim of the level (positive = reclaimed) |

**Direction reconciliation (rule 30.1, F2):** the detector's `direction` column
is the *sweep side* (`bullish`/`bearish`). Downstream trade logic (entry/stop/
target, labels, features, scoring) MUST derive the *trade direction* via
`src.schema.normalize_event_direction` (bullish→long, bearish→short) or use
`SWEEP_TO_TRADE_DIRECTION`. `src.schema.TRADE_DIRECTIONS = ["long", "short"]`;
`EVENT_DIRECTIONS = ["bullish", "bearish"]` reflect the detector vocabulary.

**Causal-absolute dedup (QA F1):** `sweep.group_rule: "first"` (baseline.yaml,
enforced by `src/config_schema.py`) makes the run's first bar the
representative event — zero intra-run look-ahead. `deepest_penetration` /
`strongest_reclaim` (guide §12) peek ahead inside the run and are permitted
only for controlled A/B experiments, never in the labelling/feature/scoring
path.

Downstream stages attach their own columns on top of this table (features
add `level_type`, `bar_index`, `dedup_group`; labeling adds entry/stop/target
and outcomes; scoring adds rule-score columns) — see sections 6–9. No row may
contain NaN in the core columns above. Events at `bar_index` within the ATR
warm-up (`index < atr_period`) or within the first `rolling_lookback` bars are
rejected by the detector.

---

## 6. Confirmation columns (attached to event table / features)

Confirmation is *optional* (config `confirmation.enabled`); when enabled and
found, these columns are set; otherwise `is_confirmed = False` and
`confirmation_time = NaN`. Setup vs confirmation separation (principle 3.2):

| Column | dtype | Notes |
|---|---|---|
| `is_confirmed` | bool | whether a confirmation candle exists within `max_wait_bars` |
| `confirmation_time` | datetime64[ns, UTC] | close time of the confirmation candle |
| `confirmation_delay_bars` | int | bars between sweep close and confirmation close |
| `confirmation_close` | float64 | close of the confirmation candle |
| `confirmation_range_atr` | float64 | range of confirmation candle / ATR |
| `confirmation_body_ratio` | float64 | |body| / range of confirmation candle |
| `confirmation_volume` | float64 | volume of confirmation candle |

Entry mode (`entry.mode`) then chooses entry at open of the candle right after
`sweep close` (A) or after `confirmation close` (B), enabling the A/B
comparison from Phase 5.

---

## 7. Feature matrix (`data/processed/features.parquet`)

One row per event, keyed by `event_id` (str) + `event_time`. Every feature
column must appear in `configs/features.yaml` with `uses_future_data: false`.
The registry is the contract; `src/features/feature_pipeline.py::build_event_features`
must output exactly the registered names (extras are allowed only if also
registered). `available_at` ∈ {`event_time`, `confirmation_time`} is respected:
confirmation features are NaN for events without confirmation.

Dtypes follow the registry (`float64`, `int64`, `bool`, `category`). Category
columns are stored as `str` with `expected_range` values.

The registry has two locked views that must never drift: the editable
`configs/features.yaml` (source of truth) and the machine-readable
`artifacts/feature_schemas/features.json` (generated by
`src/features/registry.py::write_feature_schema_json`). Every registered
feature carries `uses_future_data: false`; the workspace bans
outcome/exit/MFE/MAE/bars-to-target/bars-to-stop/future-return columns as
model features. Liquidity-level features are resolved causally at the event bar
via `src/liquidity/level_registry.py::level_state_at_bar`
(never the registry's final as-of-last-bar lifecycle), and session hours come
from the optional `sessions` config section.

---

## 8. Labeled dataset (`data/processed/labeled_events.parquet`)

Columns added by `src/labeling/*` on top of the event + feature schema:

| Column | dtype | Notes |
|---|---|---|
| `target` columns | str | `outcome_{R}r_h{H}`; values `tp` \| `sl` \| `time` \| `ambiguous` |
| `mfe_{H}` / `mae_{H}` | float64 | max favorable/adverse excursion within H bars (in price) |
| `mfe_{H}_atr` / `mae_{H}_atr` | float64 | excursion normalized by ATR at event time |
| `entry_time` / `entry_price` | datetime64 / float64 | per entry mode |
| `stop_price` | float64 | sweep extreme ± `buffer_atr` |
| `target_{R}r` | float64 | risk-multiple target prices |
| `costs_{R}r_h{H}` | float64 | net outcome after costs; gross variant kept as `outcome_{R}r_h{H}` |
| `same_bar_ambiguous` | bool | row flagged by same-bar policy |

Costs: spread/slippage/commission applied per guide section 17; gross columns
are always retained so costs are auditable.

---

## 9. Rule score columns (added by `src/scoring/rule_score.py`)

| Column | dtype | Notes |
|---|---|---|
| `rule_score` | int | 0–100 total |
| `score_liquidity` | int | 0–20 |
| `score_sweep_quality` | int | 0–25 |
| `score_reclaim` | int | 0–15 |
| `score_confirmation` | int | 0–20 |
| `score_breakdown` | str (JSON) | per-component detail dict |
| `score_bucket` | str | e.g. `"50-59"` from `scoring.score_buckets` |

---

## 10. Manual review CSV (`reports/manual_review_v{1,2}.csv`)

> **Path reconciliation (t16, rule 30.1):** round-1 output of Agent 5 (t7)
> actually lives in `reports/manual_review.csv` (plus `manual_review_summary.md`
> / `.json`). Canonical location is `reports/manual_review_v{1,2}.csv`; the
> shipped v1 file (`reports/manual_review.csv`) is kept as-is and never
> rewritten. Round 2 (Phase 5b / task t12) MUST write
> `reports/manual_review_v2.csv` without touching v1.

| Column | dtype | Notes |
|---|---|---|
| `review_id` | str | unique |
| `event_id` | str | references event table |
| `reviewer` | str | `agent_5` \| `human` |
| `verdict` | str | `correct` \| `incorrect` \| `ambiguous` |
| `notes` | str | free text |
| `review_version` | int | `1` (Phase 3) or `2` (Phase 5b) — files never overwrite |
| `reviewed_at` | timestamp | UTC |

---

## 11. Artifacts & output paths

| Artifact | Path (config-relative) |
|---|---|
| Processed OHLCV | `data/processed/xauusd_m15.parquet` |
| Data quality report | `reports/data_quality/xauusd_m15_quality.json` (canonical; legacy t2 copy at `reports/data_quality.json` kept for reference) |
| Interim levels | `data/interim/levels.parquet` |
| Interim events | `data/processed/events.parquet` (Pipeline 2 output, schema §5+§6) |
| Feature matrix | `data/processed/features.parquet` |
| Labeled dataset | `data/processed/labeled_events.parquet`; integration dataset `artifacts/datasets/liquidity_sweep_events.parquet` (Pipeline 3) |
| Model artifacts | `artifacts/models/*.pkl` (model.pkl = pipeline dict, calibrator.pkl) |
| Event charts | `reports/event_charts/` |
| Manual reviews | `reports/manual_review_v*.csv` (v1 shipped as `reports/manual_review.csv`) |
| Metrics | `reports/metrics/` |

Paths are resolved via `config.resolve_path` against the project root and are
overridable in YAML.

---

## 12. Versioning rules

- `SCHEMAS.md` header `Schema version` bumps on every breaking change.
- `CHANGELOG.md` records every schema change with date, agent, and what was
  affected (rule 30.1).
- Test set: `data/processed/test_events.parquet` is written only by Agent 6 /
  Agent 7 at the final evaluation step (rule 30.5); agents 0–5 never read it.