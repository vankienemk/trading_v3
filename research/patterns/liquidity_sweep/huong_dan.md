# GUIDE: BUILDING A LIQUIDITY SWEEP DETECTION AND SCORING SYSTEM FOR XAUUSD M15

## 1. Project Goal

Build a pipeline capable of:

1. Reading and validating OHLCV data for XAUUSD on the M15 timeframe.
2. Identifying liquidity zones from historical data.
3. Detecting liquidity sweep patterns:
   - Low sweep and reclaim for bullish setups.
   - High sweep and reclaim for bearish setups.
4. Separating:
   - The moment a sweep is detected.
   - The moment of confirmation.
   - The moment of entry.
   - The future data window used to score outcomes.
5. Tracking price behavior after each event.
6. Computing:
   - MFE.
   - MAE.
   - TP/SL outcome.
   - Time to reach TP/SL.
   - Outcomes across multiple horizons and multiple R:R ratios.
7. Producing a one-row-per-event dataset.
8. Scoring using:
   - A rule-based score.
   - A Machine Learning probability score.
9. Backtesting over time while avoiding look-ahead bias and data leakage.
10. Supporting paper trading / live inference in a later phase.

---

# 2. Initial Scope

## 2.1. In scope

The first version focuses on:

- Instrument: XAUUSD.
- Timeframe: M15.
- Input data: OHLCV or OHLC + tick volume.
- Liquidity level types:
  - Rolling high/low.
  - Confirmed swing high/low.
  - Equal highs/equal lows.
  - Optionally extendable to previous day high/low.
- Signal types:
  - Bullish liquidity sweep.
  - Bearish liquidity sweep.
- Outcome scoring:
  - Fixed stop.
  - Fixed reward-to-risk.
  - Time barrier.
  - MFE/MAE.

## 2.2. Out of scope for the first version

Not needed immediately:

- Deep Learning.
- Transformers for OHLC sequences.
- LLMs directly forecasting price.
- Reinforcement Learning.
- A real-money execution engine.
- Multi-asset portfolio optimization.
- Real-time news integration.

These should only be considered after the baseline has passed out-of-sample validation.

---

# 3. Mandatory Principles

## 3.1. Never use future data

At candle `t`, a feature may only use data available up to and including candle `t`.

Correct example:

```python
df["liq_low"] = df["low"].shift(1).rolling(20).min()
```

Incorrect example:

```python
df["liq_low"] = df["low"].rolling(20, center=True).min()
```

Reason:

- `center=True` uses future data.
- Without `shift(1)`, the current candle participates in forming the very level it is sweeping.

## 3.2. Separate setup, confirmation, and label

Must be stored separately:

- `event_time`: the moment the sweep is detected.
- `confirmation_time`: the moment confirmation appears.
- `entry_time`: the moment entry becomes possible.
- `exit_time`: the moment TP, SL, or the time barrier is hit.
- `label`: must only be built from data after entry.

Confirmation features must not be used if the strategy assumes entry right at the close of the sweep candle.

## 3.3. Backtesting must account for costs

Must support at minimum:

- Spread.
- Slippage.
- Commission, if any.
- Stop buffer.

If the dataset only has mid-price, the costs being estimated must be documented explicitly.

## 3.4. Results must be reproducible

Every experiment must save:

- Config.
- Data version.
- Git commit.
- Random seed.
- Feature list.
- Train/validation/test range.
- Model parameters.
- Metrics.
- Output event dataset.

---

# 4. Proposed Directory Structure

```text
xauusd-liquidity-sweep/
├── HUONG_DAN.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── features.yaml
│   ├── labeling.yaml
│   └── model.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_event_review.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── event_charts/
│   ├── metrics/
│   └── experiments/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── resampler.py
│   ├── indicators/
│   │   ├── atr.py
│   │   ├── volume.py
│   │   └── trend.py
│   ├── liquidity/
│   │   ├── rolling_levels.py
│   │   ├── swing_levels.py
│   │   ├── equal_levels.py
│   │   └── level_registry.py
│   ├── events/
│   │   ├── sweep_detector.py
│   │   ├── confirmation.py
│   │   └── deduplication.py
│   ├── features/
│   │   ├── candle_features.py
│   │   ├── level_features.py
│   │   ├── context_features.py
│   │   └── feature_pipeline.py
│   ├── labeling/
│   │   ├── triple_barrier.py
│   │   ├── excursions.py
│   │   └── outcome_builder.py
│   ├── scoring/
│   │   ├── rule_score.py
│   │   ├── expected_value.py
│   │   └── calibration.py
│   ├── modeling/
│   │   ├── dataset.py
│   │   ├── split.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── visualization/
│   │   ├── event_chart.py
│   │   └── reports.py
│   └── pipelines/
│       ├── build_events.py
│       ├── build_dataset.py
│       ├── train_model.py
│       └── paper_trading.py
├── tests/
│   ├── test_data_validator.py
│   ├── test_atr.py
│   ├── test_levels.py
│   ├── test_sweep_detector.py
│   ├── test_confirmation.py
│   ├── test_triple_barrier.py
│   ├── test_no_lookahead.py
│   └── test_time_split.py
└── artifacts/
    ├── models/
    ├── datasets/
    └── feature_schemas/
```

---

# 5. Work Breakdown for the LLM Team of Agents (max 8 members)

Each agent is an independent working session (its own context), receiving only the interface/schema for its own part of the work — no agent shares context with another role.

| # | Agent | Role |
|---|---|---|
| 0 | Project Lead / Integrator | Architecture, schema, merging, integration |
| 1 | Data Engineering | Loading, normalizing, validating OHLCV |
| 2 | Liquidity Level Detection | Rolling/swing/equal levels, causal `known_at` |
| 3 | Sweep Detector & Confirmation | Sweep detection, confirmation, dedup |
| 4 | Feature Engineering | Causal features for each event |
| 5 | Labeling, Trade Simulation & Human Review | Entry/stop/target, triple-barrier, charts, manual review |
| 6 | Scoring (Rule-based + ML) | Rule score, ML dataset, walk-forward, calibration |
| 7 | QA / Bias Auditor | Independent audit of the entire pipeline, including look-ahead |

## 5.1. Agent 0 — Project Lead / Integrator

### Responsibilities

- Manage config and interfaces between modules.
- Enforce naming conventions.
- Merge output from other agents.
- Run integration tests (technical: pipeline runs, output matches schema).
- Ensure the end-to-end pipeline works.
- Operate the CI/CD gate (see section 30.4) — never merge when the gate fails.

### Deliverables

- Repository structure.
- Config schema.
- Data contracts.
- End-to-end CLI or pipeline.
- Final integration test report.

### Must not do

- Change schema unilaterally without updating other agents.
- Optimize the model before the detector and labels have been verified.
- Self-assess look-ahead bias for the parts it integrates — that belongs to Agent 7 (QA), to avoid being both player and referee.

---

## 5.2. Agent 1 — Data Engineering

### Responsibilities

- Read CSV/Parquet data.
- Normalize column names.
- Normalize timezone.
- Sort timestamps.
- Remove duplicates.
- Check for missing candles.
- Check OHLC consistency.
- Produce a data quality report.

### Input

Source data: `dataset.csv`, exported directly from MT5 (MetaTrader 5), already placed in the workspace at `data/raw/dataset.csv`.

```text
timestamp, open, high, low, close, volume
```

`volume` can be real volume or tick volume (MT5 exports tick volume by default — the data quality report must state clearly which type is being used).

### Output

Normalized Parquet file:

```text
data/processed/xauusd_m15.parquet
```

### Required schema

```text
timestamp: datetime64[ns, UTC]
open: float64
high: float64
low: float64
close: float64
volume: float64
```

### Validation rules

```text
high >= max(open, close)
low <= min(open, close)
high >= low
timestamps strictly increasing
no duplicate timestamps
OHLC has no nulls
```

### Deliverables

- `src/data/loader.py`
- `src/data/validator.py`
- `tests/test_data_validator.py`
- `reports/data_quality.json`

---

## 5.3. Agent 2 — Liquidity Level Detection

### Responsibilities

Detect:

1. Rolling high/low.
2. Confirmed swing high/low.
3. Equal highs/equal lows.
4. Level age.
5. Touch count per level.
6. Level status:
   - Active.
   - Swept.
   - Invalidated.

### Key requirement

A swing level must only enter the system at the moment it has actually been confirmed.

Example: a swing low with three bars on the right:

- The swing sits at candle `t`.
- It is only known at `t+3`.
- `known_at = t+3`.

### Proposed output

```text
level_id
level_type
direction
price
origin_time
known_at
touch_count
last_touch_time
age_bars
status
```

### Deliverables

- `rolling_levels.py`
- `swing_levels.py`
- `equal_levels.py`
- `level_registry.py`
- Unit tests for causal behavior.

This is the second-highest leakage-risk module after feature engineering — it must be carefully reviewed by Agent 7 whenever it changes.

---

## 5.4. Agent 3 — Sweep Detector and Confirmation

### Responsibilities

- Detect bullish sweeps.
- Detect bearish sweeps.
- Compute penetration, wick ratio, and reclaim.
- Detect confirmation.
- Remove duplicate events.
- Apply cooldown.

### Output

```text
event_id
event_time
direction
level_id
level_price
event_high
event_low
event_close
penetration_atr
wick_ratio
reclaim_atr
confirmation_time
confirmation_type
confirmation_strength
```

### Deliverables

- `sweep_detector.py`
- `confirmation.py`
- `deduplication.py`
- Tests using synthetic data.

---

## 5.5. Agent 4 — Feature Engineering

### Responsibilities

Build features for each event using only the data available at decision time.

### Feature groups

- Candle features.
- Liquidity level features.
- Volatility features.
- Volume features.
- Session features.
- Higher-timeframe context.
- Confirmation features.
- Distance features.

### Requirements

Every feature needs metadata:

```text
feature_name
description
dtype
available_at
uses_future_data
expected_range
```

`uses_future_data` must always be `false` for any feature used to train the model.

This is the **highest** leakage-risk module on the whole team — every change to Agent 4 must be accompanied by a no-lookahead test (section 25.2) run through CI before Agent 0 merges it, and it sits at the top of Agent 7's audit priority list.

### Deliverables

- Modules under `src/features/`.
- `artifacts/feature_schemas/features.json`
- Anti-look-ahead tests.

---

## 5.6. Agent 5 — Labeling, Trade Simulation, and Human Review

### Responsibilities

- Determine entry.
- Determine stop.
- Determine target.
- Compute MFE/MAE.
- Build triple-barrier labels.
- Handle the case where TP and SL are both hit within the same candle.
- Compute costs.
- Plot charts around each event, showing level, sweep, entry, stop, target.
- Export charts by group: winner/loser/ambiguous/high score/low score.
- Produce a manual review file (`manual_review.csv`) and summarize review results.

### Output

```text
event_id
entry_time
entry_price
stop_price
target_price
risk_price
outcome
exit_time
exit_price
exit_reason
mfe_r
mae_r
bars_held
net_result_r
ambiguous
```

### Manual review schema

```text
event_id
detector_correct
liquidity_level_clear
sweep_clear
confirmation_clear
reviewer_score
reviewer_comment
```

### Deliverables

- `triple_barrier.py`
- `excursions.py`
- `outcome_builder.py`
- `event_chart.py`
- `reports/event_charts/`
- `reports/manual_review.csv`
- Unit tests for TP/SL ordering.

---

## 5.7. Agent 6 — Scoring (Rule-Based + Machine Learning)

### Responsibilities

**Rule-based part:**
- Build a 0–100 score, storing a breakdown for each component.
- Never hard-code weights in source — pull them from config.
- Compare score against outcome.

**ML part:**
- Build the event-level dataset.
- Time-based split, walk-forward validation.
- Train baseline models in order: Logistic Regression → Random Forest → CatBoost → LightGBM/XGBoost.
- Calibrate probability (Platt/Isotonic).
- Evaluate out-of-sample, analyze feature importance.
- No deep learning, transformers, or LLM forecasting on raw OHLC at this stage.

### Test set access

Only Agent 6 is allowed to read the test set, and only **once**, at the final evaluation step (see section 30.4). Agent 7 (QA) may read the test set only in an audit capacity after Agent 6 has published results, and must never use it to suggest changes back into the feature set or model.

### Output

```text
event_id
rule_score
level_score
sweep_score
confirmation_score
context_score
volume_score
model_probability
```

### Deliverables

- `rule_score.py` plus weight config.
- Training pipeline, serialized model, calibration model.
- Win rate/expectancy report by score bucket.
- Test metrics, walk-forward report.

---

## 5.8. Agent 7 — QA / Bias Auditor

### Responsibilities

Independently check the entire pipeline, including what previously belonged to Agent 0:

- Look-ahead bias (in every module: level, feature, confirmation).
- Data leakage.
- Timestamp errors.
- Features computed over the whole dataset instead of a causal rolling window.
- Event overlap.
- Train/test overlap.
- Label contamination.
- Selection bias.
- Overly optimistic same-bar assumptions.
- Trading costs being ignored.

### Deliverables

```text
reports/bias_audit.md
reports/verification_checklist.json
```

### Independence principles

- This agent does not write detector, feature, label, or model code.
- This agent has the authority to require any other agent to rework code if a leakage risk is found, even after Agent 0 has already merged it.
- This agent's sign-off is mandatory before moving to Phase 7 (Machine Learning) and before freezing the baseline (section 28).

---

# 6. Baseline Config

Create the file `configs/baseline.yaml`:

```yaml
project:
  symbol: XAUUSD
  timeframe: M15
  timezone: UTC
  random_seed: 42

data:
  input_path: data/raw/dataset.csv  # exported from MT5, already placed in the workspace
  output_path: data/processed/xauusd_m15.parquet
  timestamp_column: timestamp
  volume_column: volume

indicators:
  atr_period: 14
  volume_zscore_window: 50

liquidity:
  rolling_lookback: 20

  swing:
    left_bars: 3
    right_bars: 3

  equal_levels:
    tolerance_atr: 0.10
    min_touches: 2
    max_age_bars: 200

sweep:
  min_penetration_atr: 0.05
  max_penetration_atr: 0.50
  min_wick_ratio: 0.35
  min_reclaim_atr: 0.00
  cooldown_bars: 4

confirmation:
  enabled: true
  max_wait_bars: 3
  require_break_sweep_extreme: true
  min_body_ratio: 0.60
  min_range_atr: 0.80

entry:
  mode: next_open_after_confirmation
  slippage_price: 0.0
  spread_price: 0.0

stop:
  mode: sweep_extreme
  buffer_atr: 0.10

labeling:
  horizons: [4, 8, 16, 32]
  reward_r_values: [1.0, 1.5, 2.0]
  same_bar_policy: ambiguous
  time_barrier_result: mark_to_market

splitting:
  train_fraction: 0.60
  validation_fraction: 0.20
  test_fraction: 0.20
  embargo_bars: 32

model:
  target: outcome_2r_h16
  probability_calibration: isotonic
```

---

# 7. Data Normalization and Validation

The project's input data is `dataset.csv`, exported from MT5 and already placed in the workspace at `data/raw/dataset.csv`. There is no need to download anything or connect to an MT5 API — Agent 1 reads this file directly. Because the data comes from an MT5 export, keep in mind:

- Timestamps in an MT5 file are usually in broker time (not UTC) — this must be identified and normalized to UTC before saving to Parquet.
- The volume column in an MT5 export defaults to tick volume, not real volume — this must be stated explicitly in `reports/data_quality.json`.
- Number/column-separator formatting can depend on the locale used at export time (comma vs. period) — check this when parsing the CSV.

## 7.1. Basic validation function

```python
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamp must be sorted ascending")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Null OHLC values detected")

    invalid_high = (
        df["high"] < df[["open", "close"]].max(axis=1)
    )

    invalid_low = (
        df["low"] > df[["open", "close"]].min(axis=1)
    )

    invalid_range = df["high"] < df["low"]

    if invalid_high.any():
        raise ValueError("Invalid high values detected")

    if invalid_low.any():
        raise ValueError("Invalid low values detected")

    if invalid_range.any():
        raise ValueError("High below low detected")
```

## 7.2. Checking for missing candles

For M15, the expected spacing is 15 minutes, but watch out for:

- Weekends.
- Broker maintenance.
- Holidays.
- Short closing periods.

Do not auto forward-fill OHLC without a clearly defined rule.

The audit output should include:

```text
row_count
start_time
end_time
duplicate_count
missing_value_count
invalid_ohlc_count
suspected_gap_count
zero_volume_count
```

---

# 8. Computing ATR

```python
import pandas as pd


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()

    previous_close = out["close"].shift(1)

    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["true_range"] = true_range
    out["atr"] = true_range.rolling(period).mean()

    return out
```

The baseline uses a simple rolling ATR. Wilder's ATR can be added later, but the method used must be documented in config.

---

# 9. Identifying Liquidity Levels

## 9.1. Rolling high/low

```python
def add_rolling_liquidity_levels(
    df,
    lookback=20,
):
    out = df.copy()

    out["liq_low"] = (
        out["low"]
        .shift(1)
        .rolling(lookback)
        .min()
    )

    out["liq_high"] = (
        out["high"]
        .shift(1)
        .rolling(lookback)
        .max()
    )

    return out
```

`shift(1)` is mandatory.

## 9.2. Swing high/low

A swing low at `t` with `left_bars=3`, `right_bars=3` must satisfy:

```text
low[t] is lower than the low of the three preceding candles
low[t] is lower than the low of the three following candles
```

However:

```text
origin_time = timestamp[t]
known_at = timestamp[t + right_bars]
```

The detector may only use the level after `known_at`.

## 9.3. Equal highs/equal lows

Two or more levels are considered equal lows if:

```text
abs(low_i - low_j) <= tolerance_atr × ATR_reference
```

Baseline:

```text
tolerance_atr = 0.10
min_touches = 2
```

Each equal-level cluster must store:

- Mean/median price.
- Lowest/highest price.
- Touch count.
- First touch.
- Last touch.
- Age.
- ATR tolerance at the time the cluster was formed.

---

# 10. Defining a Liquidity Sweep

## 10.1. Bullish sweep

A baseline bullish sweep satisfies:

```text
low[t] < liquidity_low
close[t] > liquidity_low
penetration_atr >= min_penetration_atr
penetration_atr <= max_penetration_atr
lower_wick_ratio >= min_wick_ratio
```

Where:

```text
penetration_atr =
    (liquidity_low - low[t]) / ATR[t]

lower_wick =
    min(open[t], close[t]) - low[t]

lower_wick_ratio =
    lower_wick / (high[t] - low[t])

reclaim_atr =
    (close[t] - liquidity_low) / ATR[t]
```

## 10.2. Bearish sweep

```text
high[t] > liquidity_high
close[t] < liquidity_high
penetration_atr >= min_penetration_atr
penetration_atr <= max_penetration_atr
upper_wick_ratio >= min_wick_ratio
```

## 10.3. Baseline detector

```python
import numpy as np


def detect_sweeps(
    df,
    level_lookback=20,
    min_pen_atr=0.05,
    max_pen_atr=0.50,
    min_wick_ratio=0.35,
):
    out = add_atr(df)

    out["liq_low"] = (
        out["low"]
        .shift(1)
        .rolling(level_lookback)
        .min()
    )

    out["liq_high"] = (
        out["high"]
        .shift(1)
        .rolling(level_lookback)
        .max()
    )

    candle_range = (
        out["high"] - out["low"]
    ).replace(0, np.nan)

    lower_body = out[["open", "close"]].min(axis=1)
    upper_body = out[["open", "close"]].max(axis=1)

    out["lower_wick"] = lower_body - out["low"]
    out["upper_wick"] = out["high"] - upper_body

    out["lower_wick_ratio"] = (
        out["lower_wick"] / candle_range
    )

    out["upper_wick_ratio"] = (
        out["upper_wick"] / candle_range
    )

    out["low_pen_atr"] = (
        (out["liq_low"] - out["low"]) / out["atr"]
    )

    out["high_pen_atr"] = (
        (out["high"] - out["liq_high"]) / out["atr"]
    )

    out["low_reclaim_atr"] = (
        (out["close"] - out["liq_low"]) / out["atr"]
    )

    out["high_reclaim_atr"] = (
        (out["liq_high"] - out["close"]) / out["atr"]
    )

    out["sweep_long"] = (
        out["liq_low"].notna()
        & (out["low_pen_atr"] >= min_pen_atr)
        & (out["low_pen_atr"] <= max_pen_atr)
        & (out["close"] > out["liq_low"])
        & (
            out["lower_wick_ratio"]
            >= min_wick_ratio
        )
    )

    out["sweep_short"] = (
        out["liq_high"].notna()
        & (out["high_pen_atr"] >= min_pen_atr)
        & (out["high_pen_atr"] <= max_pen_atr)
        & (out["close"] < out["liq_high"])
        & (
            out["upper_wick_ratio"]
            >= min_wick_ratio
        )
    )

    return out
```

---

# 11. Confirmation

## 11.1. Principle

Confirmation occurs after a sweep and must never be treated as known at the moment of the sweep.

Multiple confirmation types may be supported:

1. Close breaking beyond the sweep candle's extreme.
2. Displacement candle.
3. Break of short-term structure.
4. CHoCH/MSS.
5. FVG or imbalance.

## 11.2. Baseline confirmation

### For a bullish sweep

Within the next three candles at most:

```text
close[k] > high[sweep]
body_ratio[k] >= 0.60
range[k] / ATR[k] >= 0.80
```

### For a bearish sweep

```text
close[k] < low[sweep]
body_ratio[k] >= 0.60
range[k] / ATR[k] >= 0.80
```

Where:

```text
body_ratio =
    abs(close - open) / (high - low)
```

## 11.3. Entry timing

If confirmation appears at candle `k`:

```text
confirmation_time = close time of candle k
entry_time = open time of candle k+1
entry_price = open[k+1] + appropriate costs
```

Entry must never be placed at the open of the confirmation candle itself, since confirmation does not yet exist at that moment.

---

# 12. Removing Duplicate Events

A single level can generate multiple consecutive sweep signals.

The baseline applies a cooldown:

```python
import numpy as np


def apply_cooldown(signal, bars=4):
    accepted = np.zeros(len(signal), dtype=bool)
    last_event = -10**9

    for position, flag in enumerate(signal):
        if flag and position - last_event > bars:
            accepted[position] = True
            last_event = position

    return accepted
```

This can be improved by grouping consecutive events and choosing:

- The first candle.
- The candle with the deepest penetration.
- The candle with the strongest reclaim.

The chosen rule must be documented.

---

# 13. Feature Engineering

## 13.1. Candle features

```text
candle_range
range_atr
body_size
body_ratio
upper_wick
lower_wick
upper_wick_ratio
lower_wick_ratio
close_location
return_1
return_4
```

Close location:

```text
(close - low) / (high - low)
```

## 13.2. Sweep features

```text
penetration_price
penetration_atr
reclaim_price
reclaim_atr
distance_close_to_level_atr
sweep_range_atr
sweep_volume_zscore
```

## 13.3. Liquidity level features

```text
level_type
level_age_bars
level_touch_count
bars_since_last_touch
equal_level_dispersion_atr
level_is_previous_day_high_low
level_is_h1_swing
level_already_partially_swept
```

## 13.4. Volatility features

```text
atr
atr_percentile
range_percentile
rolling_std_return
volatility_regime
```

ATR percentile must be computed causally using a rolling window, never a percentile over the full dataset.

## 13.5. Volume features

```text
volume
volume_zscore
volume_percentile
volume_relative_to_session
```

Example:

```text
volume_zscore =
    (volume - rolling_mean_past) / rolling_std_past
```

## 13.6. Time/session features

```text
hour_utc
day_of_week
session_asia
session_london
session_new_york
session_overlap
minutes_from_session_open
```

Session hours must live in config so they can be changed for timezone and DST policy.

## 13.7. Higher-timeframe context

M15 can be resampled into H1/H4.

Features:

```text
h1_return
h1_trend
h1_distance_to_ema
h1_atr
h4_trend
distance_to_previous_day_high
distance_to_previous_day_low
```

When merging higher-timeframe data down to M15:

- Only use H1/H4 candles that have already closed.
- Never use data from an H1 candle still forming.

## 13.8. Confirmation features

Only used if entry happens after confirmation:

```text
confirmation_delay_bars
confirmation_range_atr
confirmation_body_ratio
confirmation_volume_zscore
structure_break_distance_atr
fvg_size_atr
```

---

# 14. Entry, Stop, and Target

## 14.1. Entry

Baseline:

```text
entry = open of the candle after confirmation
```

If confirmation is not used:

```text
entry = open of the candle after the sweep
```

## 14.2. Stop for long

```text
stop =
    low of the sweep
    - stop_buffer_atr × ATR at the sweep
```

## 14.3. Stop for short

```text
stop =
    high of the sweep
    + stop_buffer_atr × ATR at the sweep
```

## 14.4. Risk

Long:

```text
risk = entry - stop
```

Short:

```text
risk = stop - entry
```

An event must be dropped or marked invalid if:

```text
risk <= 0
```

## 14.5. Target

Long:

```text
target = entry + reward_r × risk
```

Short:

```text
target = entry - reward_r × risk
```

Baseline should generate labels for:

```text
1.0R
1.5R
2.0R
```

---

# 15. Tracking Price After an Event

Baseline horizons:

```text
4 bars  = 1 hour
8 bars  = 2 hours
16 bars = 4 hours
32 bars = 8 hours
```

For each horizon, compute:

```text
MFE
MAE
MFE_R
MAE_R
close_return_R
max_close_return_R
min_close_return_R
```

## 15.1. MFE and MAE for long

```text
MFE =
    max(future_high) - entry

MAE =
    entry - min(future_low)

MFE_R = MFE / risk
MAE_R = MAE / risk
```

## 15.2. MFE and MAE for short

```text
MFE =
    entry - min(future_low)

MAE =
    max(future_high) - entry

MFE_R = MFE / risk
MAE_R = MAE / risk
```

---

# 16. Triple-Barrier Labeling

Every event has three barriers:

1. Stop-loss barrier.
2. Take-profit barrier.
3. Time barrier.

## 16.1. Proposed label

```text
1  = target hit before stop
0  = stop hit before target
-1 = horizon ends with neither target nor stop hit
NaN = undetermined because TP and SL were both hit within the same candle
```

Can also be produced:

```text
exit_reason:
    target
    stop
    time
    ambiguous
```

## 16.2. Baseline long-event labeling

```python
import numpy as np


def label_long_event(
    df,
    event_pos,
    horizon=16,
    stop_buffer_atr=0.10,
    reward_r=2.0,
):
    entry_pos = event_pos + 1

    if entry_pos >= len(df):
        return None

    event = df.iloc[event_pos]
    entry = df.iloc[entry_pos]["open"]

    stop = (
        event["low"]
        - stop_buffer_atr * event["atr"]
    )

    risk = entry - stop

    if not np.isfinite(risk) or risk <= 0:
        return None

    target = entry + reward_r * risk

    end_pos = min(
        entry_pos + horizon - 1,
        len(df) - 1,
    )

    mfe = 0.0
    mae = 0.0
    outcome = -1
    exit_reason = "time"
    exit_pos = end_pos
    ambiguous = False

    for pos in range(entry_pos, end_pos + 1):
        high = df.iloc[pos]["high"]
        low = df.iloc[pos]["low"]

        mfe = max(mfe, high - entry)
        mae = max(mae, entry - low)

        hit_stop = low <= stop
        hit_target = high >= target

        if hit_stop and hit_target:
            outcome = np.nan
            exit_reason = "ambiguous"
            exit_pos = pos
            ambiguous = True
            break

        if hit_stop:
            outcome = 0
            exit_reason = "stop"
            exit_pos = pos
            break

        if hit_target:
            outcome = 1
            exit_reason = "target"
            exit_pos = pos
            break

    return {
        "event_pos": event_pos,
        "entry_pos": entry_pos,
        "exit_pos": exit_pos,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "outcome": outcome,
        "exit_reason": exit_reason,
        "ambiguous": ambiguous,
        "mfe_r": mfe / risk,
        "mae_r": mae / risk,
        "bars_held": exit_pos - entry_pos + 1,
    }
```

An equivalent function must be written for short events.

## 16.3. TP and SL hit within the same candle

M15 data does not reveal intrabar ordering.

Supported policies:

```text
ambiguous:
    Assign NaN and exclude from binary training.

conservative:
    Assume stop was hit first.

optimistic:
    Assume target was hit first.
    Not recommended as the primary result.

lower_timeframe:
    Use M1 or tick data to determine ordering.
```

The primary result should use:

```text
same_bar_policy = ambiguous
```

If M1 data is available, prefer resolving order using M1 data.

---

# 17. Trading Costs

## 17.1. Long

Effective entry can be simulated as:

```text
entry_effective =
    raw_entry + half_spread + slippage
```

Long exit:

```text
exit_effective =
    raw_exit - half_spread - slippage
```

## 17.2. Short

Short entry:

```text
entry_effective =
    raw_entry - half_spread - slippage
```

Short exit:

```text
exit_effective =
    raw_exit + half_spread + slippage
```

Must store all of:

```text
gross_result_r
cost_r
net_result_r
```

Never publish expectancy before costs have been accounted for.

---

# 18. Final Event Dataset

Each event is one row.

Proposed schema:

```text
event_id
event_time
direction

level_id
level_type
level_price
level_origin_time
level_known_at
level_age_bars
level_touch_count

event_open
event_high
event_low
event_close
event_volume
atr

penetration_price
penetration_atr
reclaim_price
reclaim_atr
wick_ratio
range_atr
body_ratio
close_location
volume_zscore

hour_utc
day_of_week
session
atr_percentile
volatility_regime

h1_trend
h4_trend
distance_to_previous_day_high_atr
distance_to_previous_day_low_atr

confirmation_found
confirmation_time
confirmation_delay_bars
confirmation_type
confirmation_strength
confirmation_range_atr
confirmation_body_ratio

entry_time
entry_price
stop_price
risk_price

mfe_r_h4
mae_r_h4
mfe_r_h8
mae_r_h8
mfe_r_h16
mae_r_h16
mfe_r_h32
mae_r_h32

outcome_1r_h4
outcome_1r_h8
outcome_1r_h16
outcome_1r_h32

outcome_1_5r_h4
outcome_1_5r_h8
outcome_1_5r_h16
outcome_1_5r_h32

outcome_2r_h4
outcome_2r_h8
outcome_2r_h16
outcome_2r_h32

bars_to_target
bars_to_stop
ambiguous

gross_result_r
cost_r
net_result_r

rule_score
model_probability
expected_value_r
```

---

# 19. Rule-Based Scoring

## 19.1. Baseline score

0–100 scale:

| Component | Max points |
|---|---:|
| Liquidity level quality | 20 |
| Sweep quality | 25 |
| Reclaim | 15 |
| Confirmation | 20 |
| Higher-timeframe context | 10 |
| Volume and volatility | 10 |
| Total | 100 |

## 19.2. Breakdown example

### Liquidity level: 0–20

```text
Clear equal highs/lows: +8
3 or more touches: +4
H1 or previous-day level: +5
Level never swept before: +3
```

### Sweep quality: 0–25

```text
Penetration within the optimal zone: +8
Large wick ratio: +7
Large range relative to ATR: +5
Close location favors direction: +5
```

### Reclaim: 0–15

```text
Close reclaims the level: +5
Reclaim >= 0.10 ATR: +5
Reclaim >= 0.25 ATR: +5
```

### Confirmation: 0–20

```text
Confirmation within 1–3 candles: +5
Displacement present: +5
Structure break present: +5
High confirmation volume: +5
```

## 19.3. Output requirements

Do not store only the total score. Every component must be stored:

```text
score_level
score_sweep
score_reclaim
score_confirmation
score_context
score_volume
rule_score
```

## 19.4. Score evaluation

Bucket into:

```text
0–39
40–49
50–59
60–69
70–79
80–100
```

For each bucket, report:

```text
event_count
win_rate
loss_rate
ambiguous_rate
average_mfe_r
average_mae_r
average_net_result_r
profit_factor
```

---

# 20. Machine Learning

## 20.1. Main problem

Predict:

```text
P(target hit before stop | features at entry)
```

Example target:

```text
outcome_2r_h16
```

## 20.2. Baseline models

Implement in this order:

1. Logistic Regression.
2. Random Forest.
3. CatBoost.
4. LightGBM/XGBoost.

Do not start with a complex model while event counts are still low.

## 20.3. Amount of data

A dataset of 10,000 M15 candles might only produce a few hundred events.

If the event count is low:

- Reduce the number of features.
- Favor regularization.
- Avoid deep learning.
- Report confidence intervals.
- Do not trust a single train/test split.

## 20.4. Feature selection

Never feed into the model:

```text
outcome
exit_time
exit_price
mfe_r
mae_r
bars_to_target
bars_to_stop
future_return
```

These columns are labels or future data.

## 20.5. Probability calibration

Support:

- Platt scaling.
- Isotonic regression.

Evaluate calibration using:

- Brier score.
- Calibration curve.
- Expected Calibration Error, if implemented.

---

# 21. Expected Value

Given:

```text
p = calibrated win probability
reward_r = target level in R
loss_r = 1
cost_r = cost in R
```

Then:

```text
EV =
    p × reward_r
    - (1 - p) × loss_r
    - cost_r
```

Example:

```text
reward_r = 2
cost_r = 0.05
```

```text
EV =
    p × 2
    - (1 - p)
    - 0.05
```

Break-even point:

```text
p >
    (1 + cost_r)
    / (reward_r + 1)
```

For the example above:

```text
p > 1.05 / 3
p > 0.35
```

The model score should only be used once the probability has been checked for calibration.

---

# 22. Train, Validation, and Test Split

## 22.1. Never split randomly

Do not use:

```python
train_test_split(
    X,
    y,
    shuffle=True,
)
```

A random split lets future data leak into training.

## 22.2. Time-based split

Baseline:

```text
first 60%: train
next 20%: validation
last 20%: test
```

## 22.3. Walk-forward validation

Example:

```text
Fold 1:
Train months 1–3
Validate month 4

Fold 2:
Train months 1–4
Validate month 5

Fold 3:
Train months 1–5
Validate month 6
```

## 22.4. Purging

If an event in train has a horizon extending into validation, that event must be dropped from train.

## 22.5. Embargo

Leave a gap after train before validation/test.

Baseline:

```text
embargo_bars = max_horizon = 32
```

---

# 23. Metrics

## 23.1. Classification metrics

```text
ROC-AUC
PR-AUC
Log loss
Brier score
Accuracy
Precision
Recall
F1
```

For imbalanced data, prefer:

- PR-AUC.
- Brier score.
- Log loss.

## 23.2. Trading metrics

```text
event_count
trade_count
win_rate
average_win_r
average_loss_r
expectancy_r
net_expectancy_r
profit_factor
maximum_drawdown_r
average_mfe_r
average_mae_r
ambiguous_rate
```

## 23.3. Stability metrics

Must be reported by:

- Year/month.
- Session.
- Long/short.
- Volatility regime.
- Score bucket.
- Level type.
- Walk-forward fold.

Never report only a single aggregate number.

---

# 24. Human Review

Before training the ML model, randomly select at least:

```text
100–200 events
```

Including:

- Bullish sweeps.
- Bearish sweeps.
- Winners.
- Losers.
- Ambiguous cases.
- Different level types.

Each chart should show:

- At least 30–50 candles before the event.
- 16–32 candles after the event.
- Liquidity level.
- Sweep candle.
- Confirmation candle.
- Entry.
- Stop.
- Target.
- Outcome.

Review goals:

1. Does the detector actually find the visual pattern it's supposed to?
2. Is the level genuinely clear?
3. Are there many false positives from spread noise?
4. Are any clean patterns being missed?
5. Is confirmation too slow?
6. Is the stop too tight?

Do not train the model if the detector has not passed human review.

---

# 25. Mandatory Testing

## 25.1. Unit tests

Required tests for:

- ATR.
- Rolling levels.
- Swing known-at logic.
- Equal-level clustering.
- Bullish sweep.
- Bearish sweep.
- Confirmation timing.
- Cooldown.
- MFE/MAE.
- Long triple barrier.
- Short triple barrier.
- Same-bar ambiguity.
- Costs.
- Time split.
- Purging.
- Embargo.

## 25.2. No-lookahead test

One testing method:

1. Run the feature pipeline on all data up to `T`.
2. Re-run the pipeline on data truncated at `t < T`.
3. Compare features at timestamps `<= t`.
4. Results must be identical.

Pseudo-test:

```python
full = build_features(df)
partial = build_features(df.iloc[:1000])

columns = FEATURE_COLUMNS

assert_frame_equal(
    full.iloc[:1000][columns],
    partial[columns],
    check_dtype=False,
)
```

If an old feature value changes once future data is added, that feature is at risk of leakage.

## 25.3. Synthetic scenario tests

Build synthetic OHLC sequences to test:

- Long sweep hitting TP first.
- Long sweep hitting SL first.
- Short sweep hitting TP first.
- Short sweep hitting SL first.
- TP and SL hit in the same candle.
- Event near the end of the dataset.
- ATR equal to NaN during warm-up.
- Candle range equal to 0.
- Entry producing an invalid risk value.

---

# 26. Execution Pipelines

## 26.1. Pipeline 1 — Data audit

```text
Raw CSV
→ normalize columns
→ parse timestamps
→ validate OHLC
→ detect gaps
→ save Parquet
→ save data quality report
```

## 26.2. Pipeline 2 — Build events

```text
Processed OHLCV
→ ATR
→ liquidity levels
→ sweep detector
→ confirmation detector
→ deduplication
→ event table
```

## 26.3. Pipeline 3 — Build labeled dataset

```text
Event table
→ event features
→ entry/stop/target
→ triple-barrier labels
→ MFE/MAE
→ costs
→ rule score
→ final event dataset
```

## 26.4. Pipeline 4 — Train model

```text
Event dataset
→ select causal features
→ time split
→ walk-forward validation
→ train baseline models
→ probability calibration
→ evaluate
→ save model artifact
```

## 26.5. Pipeline 5 — Paper trading

```text
New closed M15 candle
→ update indicators
→ update levels
→ detect sweep
→ wait for confirmation
→ calculate features
→ rule score
→ model probability
→ expected value
→ create paper trade
→ monitor future candles
→ finalize outcome
```

---

# 27. Proposed Function Interfaces

## Data

```python
def load_ohlcv(path: str) -> pd.DataFrame:
    ...
```

```python
def validate_ohlcv(df: pd.DataFrame) -> None:
    ...
```

## Indicators

```python
def add_atr(
    df: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    ...
```

## Levels

```python
def build_liquidity_levels(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Events

```python
def detect_sweep_events(
    candles: pd.DataFrame,
    levels: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Confirmation

```python
def attach_confirmations(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Features

```python
def build_event_features(
    candles: pd.DataFrame,
    levels: pd.DataFrame,
    events: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Labels

```python
def build_event_labels(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Scoring

```python
def calculate_rule_scores(
    events: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    ...
```

## Modeling

```python
def train_models(
    dataset: pd.DataFrame,
    config: dict,
) -> dict:
    ...
```

---

# 28. Phased Implementation Process

## Phase 1 — Baseline detector

Implement:

- ATR 14.
- Rolling high/low over 20 candles.
- Min penetration: `0.05 ATR`.
- Max penetration: `0.50 ATR`.
- Min wick ratio: `0.35`.
- Close reclaim of the level.
- Four-candle cooldown.

### Completion criteria

- Detector runs over the entire dataset.
- Event table has no duplicate IDs.
- No look-ahead in the rolling levels.
- Charts exist for at least 100 events.

---

## Phase 2 — Labeling

Implement:

- Entry at the open of the next candle.
- Stop past the sweep extreme by `0.10 ATR`.
- Targets at 1R, 1.5R, 2R.
- Horizons of 4, 8, 16, 32.
- MFE/MAE.
- Same-bar ambiguity handling.

### Completion criteria

- Both long and short are supported.
- Unit tests for TP/SL pass.
- No access beyond the end of the dataset.
- Ambiguous events are recorded separately.

---

## Phase 3 — Human review (round 1)

Owner: Agent 5, coordinating with Agent 7.

- Review 100–200 charts (based on the rolling-levels baseline — before swing/equal levels exist).
- Mark detector correct/incorrect.
- Track false-positive rate.
- Adjust thresholds if needed.
- Freeze the detector baseline (rolling levels) before extending to more complex levels.

### Completion criteria

- `manual_review.csv` exists.
- A detector error report exists.
- Detector changes are versioned.

---

## Phase 4 — Upgrading liquidity levels

Owner: Agent 2, audited by Agent 7.

Add:

- Confirmed swing levels.
- Equal highs/lows.
- Previous day high/low.
- H1 levels.
- Level age and touch count.

### Completion criteria

- Every level has a `known_at`.
- Causal-behavior tests pass.
- Every event can be traced back to its source level.

---

## Phase 5 — Confirmation

Owner: Agent 3, audited by Agent 7.

Add:

- Displacement.
- Close breaking beyond the sweep extreme.
- Structure break.
- Confirmation delay.

Compare two strategies:

```text
A: entry immediately after the sweep
B: entry after confirmation
```

Evaluate:

- Win rate.
- Expectancy.
- Average stop distance.
- Event count.
- MFE/MAE.
- Entry delay.

---

## Phase 5b — Human review (round 2, mandatory)

Owner: Agent 5, audited by Agent 7.

Once swing/equal levels (Phase 4) and confirmation (Phase 5) are working, re-run manual review on the new event set — do not reuse the conclusions from round 1.

- Review another 100–200 charts, prioritizing events that use a swing/equal level or that have confirmation.
- Check specifically: is confirmation delayed unreasonably, are the new levels producing many false positives.
- Agent 7 gives a preliminary bias-audit sign-off here before Phase 6 can proceed.

### Completion criteria

- `manual_review.csv` version 2 exists (does not overwrite version 1).
- Agent 7 confirms no new look-ahead issues have arisen from Phases 4–5.

---

## Phase 6 — Rule score

Owner: Agent 6.

- Build the 0–100 score.
- Store the breakdown.
- Evaluate by score bucket.
- Do not over-tune thresholds on the test set.

---

## Phase 7 — Machine Learning

Owner: Agent 6, final audit by Agent 7.

- Logistic Regression baseline.
- CatBoost/LightGBM.
- Walk-forward validation.
- Probability calibration.
- Expected value filtering.
- The test set is opened by Agent 6 exactly **once**, at the final step (see section 30.4).

### Completion criteria

- The model beats a simple baseline out-of-sample.
- Probability calibration is reasonable.
- Results remain positive after costs.
- No feature leakage.
- Agent 7 has signed off on the final bias audit.

---

## Phase 8 — Paper trading

Run at least a few hundred events if possible.

Track:

```text
signal_time
features
rule_score
model_probability
expected_value
paper_entry
paper_stop
paper_target
spread
slippage
MFE
MAE
outcome
```

Do not move to live trading based on backtesting alone.

---

# 29. Role of LLMs in the Project

LLMs are well suited for:

- Generating code against an interface.
- Writing unit tests.
- Reviewing anti-look-ahead logic.
- Analyzing false positives.
- Summarizing experiments.
- Generating model reports.
- Helping label charts.
- Proposing new features.
- Comparing results across versions.

LLMs should not be the primary forecasting model on raw OHLC because:

- The dataset is small.
- Leakage is hard to control.
- Probability calibration is hard.
- Stability is hard to explain.
- The cost and complexity are unnecessary at the baseline stage.

Preferred order:

```text
Rule-based detector
→ event-level features
→ tabular ML
→ calibrated probability
```

---

# 30. Coordination Rules Between Agents

## 30.1. Never change interfaces arbitrarily

If an agent changes a schema, it must:

1. Update the schema file.
2. Notify the Project Lead.
3. Update tests.
4. Update downstream modules.
5. Log it in the changelog.

## 30.2. Every module must have

- Docstrings.
- Type hints.
- Unit tests.
- Input schema.
- Output schema.
- Error handling.
- No mutation of the input DataFrame unless explicitly documented.

## 30.3. Every pull request must document

```text
Goal
Files changed
Input/output
Assumptions
Look-ahead risk
Tests run
Known limitations
```

## 30.4. Mandatory CI/CD gate before merging

Do not rely on each agent remembering to run checks. Agent 0 operates an automated gate that runs before every merge, at minimum:

```text
lint (ruff)
type check (mypy)
unit tests (pytest)
no-lookahead test (section 25.2) — mandatory for any change in Agent 2 or Agent 4
```

Merges are rejected if any step fails, with no "fix it later" exception. Agent 0 has no authority to override the gate — only Agent 7 may request a revert of a merge that passed the gate if leakage is later found that the gate failed to catch.

## 30.5. Test set access

- Only Agent 6 may read the test set, and only once, at the final evaluation step (Phase 7).
- Agent 7 may read the test set after Agent 6 has published results, strictly for auditing — never to propose changes back to the feature set or model based on it.
- All other agents (0–5) have no access to the test set file for the entire life of the project.

## 30.6. Never simultaneously optimize the detector and the model on the test set

The correct process:

```text
Train:
    build the model

Validation:
    choose features and hyperparameters

Test:
    a single final evaluation
```

If the test set has already been used to tune anything, it is no longer a real test.

---

# 31. Definition of Done

The baseline project is considered complete when:

## Data

- [ ] Data successfully normalized.
- [ ] Timestamps unified.
- [ ] No duplicates.
- [ ] OHLC validation passes.
- [ ] A gap/missing-values report exists.

## Liquidity levels

- [ ] Rolling levels are causal.
- [ ] Swing levels have `known_at`.
- [ ] Equal levels have an ATR tolerance.
- [ ] Every event can be traced to its source level.

## Detector

- [ ] Both long and short sweeps are detected.
- [ ] Penetration and wick ratio are correct.
- [ ] Events are deduplicated.
- [ ] Human review has been done.

## Confirmation

- [ ] Confirmation does not use data before the candle closes.
- [ ] Entry occurs after confirmation.
- [ ] Setups with/without confirmation can be compared.

## Labeling

- [ ] Entry/stop/target are well-defined.
- [ ] MFE/MAE are correct.
- [ ] TP/SL are checked in candle order.
- [ ] Same-bar ambiguity is handled.
- [ ] Multiple horizons and R:R ratios exist.

## Scoring

- [ ] Rule score has a breakdown.
- [ ] Score is evaluated by bucket.
- [ ] ML probability is calibrated.
- [ ] Expected value after costs exists.

## Validation

- [ ] No random split.
- [ ] Walk-forward validation exists.
- [ ] Purging/embargo exist.
- [ ] No-lookahead tests pass.
- [ ] The QA agent has completed the bias audit.

## Reporting

- [ ] Event dataset exists.
- [ ] Model report exists.
- [ ] Chart review exists.
- [ ] Metrics exist by session/direction/regime.
- [ ] The experiment is reproducible.

---

# 32. Common Mistakes to Avoid

1. Using a centered rolling window.
2. Missing `shift(1)` when creating a liquidity level.
3. Identifying a swing at the swing candle itself instead of at its confirmation time.
4. Using an unclosed H1 candle as a feature for M15.
5. Placing entry at the open of the confirmation candle.
6. Using MFE/MAE as a feature.
7. Random train/test split.
8. Not purging events whose future windows overlap.
9. Treating TP as a win when TP and SL are hit in the same candle.
10. Not accounting for spread/slippage.
11. Counting multiple candles of the same sweep as separate independent events.
12. Over-tuning thresholds on the test set.
13. Reporting only win rate without expectancy.
14. Ignoring event count.
15. Using accuracy for an imbalanced target.
16. Using an overly complex model with only a few hundred events.
17. Skipping manual chart review.
18. Changing the detector after seeing test results.
19. Computing percentiles or scaling over the whole dataset.
20. Assuming a model's output probability is already calibrated.

---

# 33. Final Deliverables

The completed pipeline must produce:

```text
data/processed/xauusd_m15.parquet
artifacts/datasets/liquidity_sweep_events.parquet
artifacts/models/model.pkl
artifacts/models/calibrator.pkl
artifacts/feature_schemas/features.json
reports/data_quality.json
reports/manual_review.csv
reports/bias_audit.md
reports/metrics/test_metrics.json
reports/figures/calibration_curve.png
reports/figures/equity_curve_r.png
reports/figures/score_bucket_performance.png
reports/event_charts/
```

---

# 34. Short Execution Plan for the Team of Agents (8 agents)

Sprint ↔ Phase mapping (section 28), to avoid the confusion of "sprint done" meaning "phase done":

| Sprint | Corresponding Phase |
|---|---|
| Sprint 1 | Phase 1 (Baseline detector) |
| Sprint 2 | Phase 2 (Labeling) + Phase 3 (Human review round 1) |
| Sprint 3 | Phase 4 (Level upgrades) + Phase 5 (Confirmation) + Phase 5b (Human review round 2) |
| Sprint 4 | Phase 6 (Rule score) + Phase 7 (Machine Learning) |

A sprint is only considered done once Agent 7 has confirmed the corresponding DoD checklist (section 31) — never move to the next sprint just because time ran out.

## Sprint 1 — Phase 1

Agent 1:

- Data loader.
- Validator.
- Data quality report.

Agent 2:

- Rolling liquidity levels.
- ATR.
- Causal level tests.

Agent 3:

- Baseline sweep detector.
- Cooldown.

Agent 7:

- Design the no-lookahead tests (used project-wide, not just for this sprint).
- First review: rolling levels and sweep detector.

Agent 0:

- Lock down schema/naming conventions once, for the whole project.
- Set up the CI/CD gate (section 30.4).

## Sprint 2 — Phase 2 + 3

Agent 5:

- Entry/stop/target.
- MFE/MAE.
- Triple barrier.
- Event charts + manual review round 1.

Agent 0:

- Assemble the first pipeline (data → level → sweep → label).

Agent 7:

- Audit labeling logic (same-bar policy, cost).
- Confirm the Phase 2–3 DoD before opening Sprint 3.

## Sprint 3 — Phase 4 + 5 + 5b

Agent 2:

- Swing levels.
- Equal highs/lows.
- Previous day levels.

Agent 3:

- Confirmation.
- Displacement.
- Structure break.

Agent 4:

- Feature pipeline (only start once levels + confirmation are stable).

Agent 5:

- Human review round 2 (mandatory — see Phase 5b).

Agent 7:

- Dedicated audit of swing/equal levels and confirmation — leakage risk rises noticeably compared to Sprint 1 here.
- Sign off before opening Sprint 4.

## Sprint 4 — Phase 6 + 7

Agent 6:

- Rule score + score bucket report.
- Time split, walk-forward training, calibration, model evaluation.
- Open the test set exactly once, at the final step (section 30.5).

Agent 7:

- Final bias audit, including auditing the test set after Agent 6 publishes results.
- Final sign-off before Agent 0 freezes the baseline.

Agent 0:

- End-to-end integration.
- Freeze the baseline version.

---

# 35. Conclusion

The project's priority sequence is:

```text
Clean data
→ causal liquidity level
→ sweep detector
→ human review
→ accurate labeling
→ event-level features
→ rule-based score
→ tabular ML
→ probability calibration
→ paper trading
```

The most important factor is not a complex model but:

1. A clearly defined event.
2. Never using future data.
3. Correctly simulating entry timing.
4. Correctly handling TP/SL order.
5. Fully accounting for costs.
6. Validating over time.
7. Evaluating on data never used for tuning.

Only once the baseline has passed all of the above should the system be extended into something more complex.
