# PATTERN_SPECS — double_bottom (P1)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 / §13 Agent 3.
Detector: `research/patterns/double_bottom/detector.py` (class
`DoubleBottomDetector`, version `1.0`).

## 1. Geometry definition

A **double bottom** is a bullish reversal structure on a downtrend approach:

```
        \       /  ← neckline (swing high, H2)
         \  N  /
   D1     \   /    D2
    \______\_/______\_
              ↓ breakout: close > neckline
```

* two swing **lows** `D1`, `D2` at approximately the same level
  (`|D1 − D2| ≤ max_equal_atr × ATR`, default 1.5 ATR);
* a swing **high** `N` (the neckline touch) strictly between them
  (`min_separation_bars` on each side, default 4 bars);
* genuine depth: `N − max(D1, D2) ≥ min_depth_atr × ATR` (default 3.0 ATR)
  so sub-ATR noise cannot satisfy the structure — 3.0 ATR also keeps the
  §8.3 pure-noise FP ≤ 5 %;
* confirmation: after `D2`'s pivot is causally known
  (`D2.bar + right_bars`, §3.2), a close crosses **above** the neckline
  price `N` (the high of the middle swing) within the staleness window.

## 2. Swing formula (per §3.2 / §11 mandatory field)

* shared infra: `research/core/swing_detector.py` — `SwingDetector`,
  used identically by double_bottom / double_top (and later Wedge / H&S);
* left bars: `left_bars = 3` (configurable) — pivot must be strictly more
  extreme than the low/high `left_bars` bars before it;
* right bars: `right_bars = 3` (configurable) — pivot must be strictly more
  extreme than the low/high `right_bars` bars after it;
* **causal existence**: pivot at bar `t` is usable only from
  `known_at_bar = t + right_bars` (§3.2 Right-Bar Rule).  The detector sets
  `detect_time = df.index[D2.bar + right_bars]` and never reads a bar past
  it for structural features;
* price source: pivot lows from the **low** series, pivot highs from the
  **high** series (OHLCV);
* min distance between pivots: `i2 − i1 ≥ min_separation_bars` and
  `i3 − i2 ≥ min_separation_bars` (default 4 bars); no hard max — the
  staleness window acts as the upper bound on the second-leg→confirm span.

## 3. Confirmation rule

* neckline = price of the middle swing (`N` = high of `H2`), known at
  `H2.bar + right_bars`;
* confirmation = first bar `c > D2.bar` whose **close** crosses above `N`;
* because the pattern is not complete until `D2`'s pivot exists,
  `confirm_bar = max(c, D2.bar + right_bars)` — the effective confirm bar;
* staleness (§3.3): if no close-cross fires within
  `max_bars_between_detect_and_confirm` (default 60 bars) after `D2`, the
  candidate is discarded (never emitted as an event).

## 4. Entry / SL / TP

* entry: open of the bar **after** `confirm_bar` (causal — known at that
  bar); if the data ends at the confirm bar, entry = confirm close;
* stop: `min(D1, D2) − stop_buffer_atr × ATR` (default buffer 0.5 ATR)
  — below the lower extreme;
* risk R = `|entry − stop|`; target = `entry + target_r × R`
  (default `target_r = 1.5`);
* all levels are carried in `event.structure_levels`
  (`neckline`, `double_bottom_level`, `extreme1_level`, `extreme2_level`)
  and `event.entry_price/stop_price/target_price`.

## 5. Staleness window (§3.3)

* `max_bars_between_detect_and_confirm = 60` (M15 default; the second leg
  → breakout usually fires within the first leg's rise, well under this);
* config-driven; events discarded when confirmation misses the window —
  the live Event Lake logs them with `attributes.discard_reason = "stale"`
  (engine responsibility, §3.3).
* **Semantics (handoff P1#6 / docs/findings_resolution.md):** the confirm
  close-cross scan is anchored at the LAST EXTREME bar
  (`scan_end = min(n, D2.bar + 1 + max_wait)`), which — because the detect bar
  is the pivot-known bar `D2.bar + right_bars` — makes the effective
  detect→confirm bound `max_wait − right_bars` (57 bars at the 60/3 defaults):
  STRICTER than the config key name reads.  Kept deliberately (conservative:
  discards more, never accepts a late confirm; relaxing would change event
  counts and invalidate trained models).  Pinned by
  `test_staleness_effective_bound_detect_to_confirm`.

## 6. Default config + version

```python
{
  "version": "1.0",            # semantic version — bump on behavior change (§6.2)
  "left_bars": 3, "right_bars": 3,
  "atr_period": 14,
  "min_separation_bars": 4,
  "max_equal_atr": 1.5,
  "min_depth_atr": 3.0,
  "stop_buffer_atr": 0.5,
  "target_r": 1.5,
  "max_bars_between_detect_and_confirm": 60,
  "cooldown_bars": 3,
  "symbol": "XAUUSD", "timeframe": "M15",
}
```

`config_hash = sha1(canonical_json(config) + version + indicators_version)[:12]`
(§6.2, `research/core/config_hash.py`); stamped on every event via
`event.config_hash` + `feature_schema_version = "double-v1.0"`.

## 7. Features (scoring inputs) + available_at (§3.4)

| feature | dtype | available_at | meaning |
|---|---|---|---|
| `atr` | float | detect | ATR(14) at the pivot-known bar (data ≤ detect) |
| `depth_atr` | float | detect | neckline depth in ATR |
| `low_offset_atr` | float | detect | offset between the two lows in ATR (equal-level fidelity) |
| `symmetry_ratio` | float | detect | left/right arm-length symmetry ∈ [0,1] |
| `pattern_length` | int | detect | bars between the two extremes |
| `confirm_reclaim_atr` | float | confirm | confirm close above neckline in ATR |
| `confirm_range_atr` | float | confirm | confirm bar range in ATR |

All features read bars **≤** their `available_at` stamp; `uses_future_data =
False`; runtime validation via `research.core.causal_checks.validate_causality`.

Every declared feature is emitted: `confirm_range_atr` (confirm bar range,
ATR@confirm-bar normalized) is attached to `event.attributes` at the confirm
bar (handoff P1#5 fix, see docs/findings_resolution.md; the historical dead
bare-expression is gone).

`rule_score` (0–1) = 0.40·depth + 0.25·symmetry + 0.20·equal-fidelity +
0.15·reclaim.

## 8. Dataset builder + labeling (§13 Agent 3)

* `research/patterns/double_bottom/dataset.py::label_events` — forward,
  fixed-horizon labeling: for each event, `horizon_bars = 72` bars strictly
  after entry (18 h M15):
  * `mfe_ratio` = best favourable excursion in R over (entry, entry+96];
  * `mae_ratio` = worst adverse excursion in R over (entry, entry+96];
  * `close_ratio` = fixed-horizon exit PnL in R;
  * `label = 1` iff `mfe_ratio ≥ target_r` (1.25R touched inside
    horizon — the label target is a research hyperparameter distinct from
    the detector's 1.5R trade target);
* **no look-ahead**: labels read only bars after `entry_bar`; features only
  bars ≤ `known_at`;
* dataset columns: `research/patterns/double_bottom/dataset.py::labeled_to_frame`;
* §6.3 research gates: `run_gates` / `assert_gates` (n≥300, n_oos≥100,
  balance ∈ [10,90] %, ESS ≥ 60 %, OOS PF bootstrap CI lower > 1.0,
  PR-AUC ≥ 1.05×baseline — logistic scorer fit on TRAIN-only causal
  features, ≥3 consecutive WF folds with PF ≥ 0.8) — report
  in `research/patterns/double_bottom/artifacts/reports/gate_report_xauusd_m15.md`.

## 9. Expected event frequency (data audit, XAUUSD M15)

Numbers below are produced by `scripts/generate_reports.py` on the XAUUSD
M15 dataset (audit run on 2026-09-07, ~99.7k bars 2022–2026):
detected events / labeled events / OOS events — see the gate report file for
the exact audit table.  Default geometry targets a few hundred events per
~4.5 years of M15 data (≥300 total per §6.3 sample-size gate).

## 10. Synthesizer benchmark (§8.3)

* harness: `research/patterns/double_bottom/benchmark.py` +
  `research/core/pattern_synthesizer.py::run_benchmark/assert_acceptance`;
* gates: recall ≥ 80 % (≤2-bar tolerance), robustness drop ≤ 15 pts when
  noise doubles, FP ≤ 5 % on pure-noise OU series — see
  `artifacts/reports/synthesizer_benchmark.md` for measured values.