# Module Interface Contracts

> **Contract owner:** Agent 0 (integrator). Every module must conform to these
> signatures and the schemas in `docs/SCHEMAS.md`. Per rule 30.2 every module
> must have docstrings, type hints, unit tests, input/output schema, error
> handling, and must not mutate its input DataFrame unless explicitly
> documented. Per rule 30.1 interface changes require this file + `SCHEMAS.md`
> + `src/schema.py` + tests + changelog updates and notification of Agent 0.

All functions below are the *canonical* contracts from guide section 27,
expanded with the locked column schemas. Implementations may add keyword-only
options but may not change positional order or output column names.

---

## 1. Data stage (Agent 1 → consumers: all)

```python
def load_ohlcv(path: str) -> pd.DataFrame
def normalize_ohlcv(df: pd.DataFrame, volume_kind: str = "tick", timezone: str = "UTC") -> pd.DataFrame
def validate_ohlcv(df: pd.DataFrame) -> None            # raises ValueError on violation
def run_data_pipeline(cfg: dict[str, Any]) -> pd.DataFrame  # full Pipeline 1
```

- Input frame: MT5 export columns (`DATE TIME OPEN HIGH LOW CLOSE TICKVOL VOL SPREAD`,
  tab-separated, angle-bracketed headers).
- `normalize_ohlcv` returns schema §3: index `timestamp` (UTC, sorted, unique),
  columns `open high low close volume` (float64).
- `validate_ohlcv` raises `ValueError` (not assert) with a message naming the
  failing invariant (missing columns, duplicate/unsorted timestamps, null
  OHLC, high/low violations).
- Errors: `FileNotFoundError` / `ValueError`. No mutation of input.

## 2. Indicators stage (Agent 2/4 → detector, features)

```python
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame        # adds true_range, atr
def atr_percentile_causal(atr: pd.Series, window: int = 200) -> pd.Series
def volatility_regime_causal(atr_pct: pd.Series, low: float = 0.33, high: float = 0.67) -> pd.Series
def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series    # excludes current bar's own volume
def ema(series: pd.Series, period: int) -> pd.Series
def ema_slope_sign(series: pd.Series, period: int, lag: int = 1) -> pd.Series
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame
def closed_higher_timeframe_merge(df: pd.DataFrame, rule: str, prefix: str, columns: tuple = (...)) -> pd.DataFrame
def previous_day_high_low(df: pd.DataFrame) -> pd.DataFrame            # adds prev_day_high, prev_day_low
```

- All indicators are causal: only data ≤ the current bar (after close); never
  centered windows, never `shift(negative)`.
- Input `df` is never mutated; a copy is returned.

## 3. Liquidity levels (Agent 2 → detector, features)

```python
def add_rolling_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame  # adds liq_low, liq_high
def rolling_level_at(df: pd.DataFrame, lookback: int = 20) -> tuple[pd.Series, pd.Series]
def detect_swing_levels(df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3, max_age_bars: int = 200) -> pd.DataFrame
def detect_equal_levels(df: pd.DataFrame, tolerance_atr: float = 0.10, min_touches: int = 2, max_age_bars: int = 200, atr: pd.Series | None = None) -> pd.DataFrame
def build_liquidity_levels(df: pd.DataFrame, config: dict) -> pd.DataFrame  # unified registry (schema §4 + §4.1)
# Phase 4 additions (schema lock v1.1, t8):
def detect_h1_swing_levels(df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3, max_age_bars: int = 200, rule: str = "1h") -> pd.DataFrame
def detect_previous_day_levels(df: pd.DataFrame, max_age_bars: int = 200) -> pd.DataFrame  # level_type="prev_day"
def level_state_at_bar(registry: pd.DataFrame, df: pd.DataFrame, atr: pd.Series, bar_pos: int) -> pd.DataFrame  # causal per-bar state, schema §4.2
```

- **Causality contract:** swing levels get `known_at = origin + right_bars`;
  equal levels `known_at = last_touch` (the completing cluster touch);
  `prev_day` levels `known_at` = first bar of the next trading day; H1 swings
  `known_at` = first M15 bar at/after the confirming (*closed*) H1 candle's
  close — a still-forming H1 candle never contributes. Rolling levels are
  columns already shifted by 1. The detector may only consume a level for
  bars with `timestamp >= known_at`.
- Level table columns per schema §4 (exact names, ordering flexible);
  `build_liquidity_levels` emits the schema §4.1 extension columns and
  `level_state_at_bar` returns schema §4.2 columns. Single implementation
  point: `src.schema` constants (`LEVEL_COLUMNS`, `LEVEL_REGISTRY_COLUMNS`,
  `LEVEL_STATE_COLUMNS`, `SWEEP_STATES`).
- Errors: `ValueError` for non-monotonic index / missing columns.

## 4. Events (Agent 3 → labeling, features, scoring)

```python
def detect_sweep_events(candles: pd.DataFrame, levels: pd.DataFrame, config: dict) -> pd.DataFrame
def deduplicate_events(events: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame
def attach_confirmations(candles: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame
```

- Output event table per schema §5; every row has a unique `event_id`
  (`SWP-XXXXXX`); `direction` uses the sweep-side vocabulary
  (`bullish`/`bearish`, guide §10); penetration measured at candle close
  only. Downstream stages MUST derive the trade direction
  (`long`/`short`) via `src.schema.normalize_event_direction`.
- `attach_confirmations` adds schema §6 columns; setup and confirmation are
  kept as separate columns so strategy A/B can be compared (principle 3.2).
- Events with `bar_index < atr_period` or `bar_index < rolling_lookback` are
  dropped (warm-up).

## 5. Features (Agent 4 → modeling)

```python
def build_event_features(candles: pd.DataFrame, levels: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame
```

- Output: one row per event: `event_id`, `event_time`, plus every feature
  registered in `configs/features.yaml` (exact names, `uses_future_data: false`
  mandatory). Registry mismatch raises `ValueError` (a feature in registry is
  missing from output, or an unregistered column appears).
- Categorical features keep the registry's `expected_range` values.

## 6. Labels (Agent 5 → modeling, scoring)

```python
def build_event_labels(candles: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame
```

- Adds target columns `outcome_{R}r_h{H}` ∈ {tp, sl, time, ambiguous}, MFE/MAE
  columns per schema §8, entry/stop/target prices, and cost-adjusted variants.
- No access beyond the dataset end; same-bar ambiguity flagged
  (`same_bar_ambiguous`), not silently resolved.

## 7. Scoring (Agent 6 → modeling; Agent 7 audits)

```python
def calculate_rule_scores(events: pd.DataFrame, config: dict) -> pd.DataFrame  # adds rule_score + breakdown (§9)
def calculate_expected_value(scores: pd.DataFrame, config: dict) -> pd.DataFrame
```

- `rule_score` ∈ [0, 100]; breakdown columns sum-checkable to the total.

## 8. Modeling (Agent 6 → final artifacts)

```python
def train_models(dataset: pd.DataFrame, config: dict) -> dict  # {"model", "metrics", "report"}
```

- Internal APIs (`split.py`, `evaluate.py`, `inference.py`) keep the same
  contract: time-based split only (never random), embargo/purge applied,
  calibration fit on validation only.

---

## Cross-cutting requirements

1. **No-look-ahead:** any function whose output at row `t` depends on rows
   after `t` is a schema violation (rule 30.4). Baseline no-lookahead tests
   live in `tests/test_no_lookahead.py` (marker `no_lookahead`); the
   authoritative suite is owned by Agent 7 (t5).
2. **Immutability:** input DataFrames are never modified in place; return
   copies. Exceptions are documented in the docstring.
3. **Errors:** domain errors raise typed exceptions (`ConfigError`,
   `ValueError`, `FileNotFoundError`) with actionable messages; no silent
   NaN-swallowing in core event/label paths.
4. **Schema conformance:** outputs are validated against `src/schema.py`
   constants in tests; any deviation is a merge-blocking failure.
5. **Reproducibility:** every pipeline honors `project.random_seed`; no global
   random state without re-seeding.