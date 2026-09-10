# PATTERN_SPECS — double_top (P2, mirror of double_bottom)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P2) / §13
Agent 3.  Detector: `research/patterns/double_top/detector.py` (class
`DoubleTopDetector`, version `1.0`).

**Double Top is the exact MIRROR of Double Bottom and reuses 100 % of P1
infra** (spec §11): the shared `SwingDetector`
(`research/core/swing_detector.py`), the detection machinery
(`DoublePatternDetectorBase` in `research/patterns/double_bottom/detector.py`),
the benchmark harness and the dataset/labeling/gates
(`research/patterns/double_bottom/`).  Every field below is the mirror of the
double_bottom spec — read `research/patterns/double_bottom/PATTERN_SPECS.md`
for the shared conventions.

## 1. Geometry definition

Bearish reversal structure on an uptrend approach: two swing **highs** `U1`,
`U2` at approximately the same level, a swing **low** `N` (neckline touch)
between them, and a close crossing **below** the neckline after `U2`'s pivot
is causally known.

## 2. Swing formula (§3.2 / §11 mandatory field)

* shared `SwingDetector`; `left_bars = 3`, `right_bars = 3` (configurable);
* pivot high/low at bar `t` usable only from `known_at_bar = t + right_bars`
  (§3.2 Right-Bar Rule); `detect_time = df.index[U2.bar + right_bars]`;
* price source: pivot highs from the **high** series, pivot lows from the
  **low** series;
* min pivot distance: `i2 − i1 ≥ min_separation_bars`,
  `i3 − i2 ≥ min_separation_bars` (default 4 bars); upper bound = staleness.

## 3. Confirmation rule

* neckline = price of the middle swing (`N` = low of `H`-mirror swing),
  known at `N.bar + right_bars`;
* confirmation = first bar `c > U2.bar` whose **close** crosses **below** `N`;
* `confirm_bar = max(c, U2.bar + right_bars)`; staleness
  `max_bars_between_detect_and_confirm = 60` bars (discard if missed).

## 4. Entry / SL / TP

* entry: open of the bar after `confirm_bar` (causal);
* stop: `max(U1, U2) + stop_buffer_atr × ATR` (default buffer 0.5 ATR);
* risk R = `|entry − stop|`; target = `entry − target_r × R`
  (default `target_r = 1.5`);
* `structure_levels` = `{neckline, double_top_level, extreme1_level,
  extreme2_level}`.

## 5. Staleness window (§3.3)

Same as P1: `max_bars_between_detect_and_confirm = 60`; missed confirm →
candidate discarded; Event Lake logs `discard_reason = "stale"` (engine).
**Semantics (handoff P1#6):** scan anchored at the last extreme → effective
detect→confirm bound `max_wait − right_bars` (57 at 60/3 defaults), stricter
than the config name reads, intentionally kept — see
`double_bottom/PATTERN_SPECS.md` §5 + `docs/findings_resolution.md`.

## 6. Default config + version

Identical schema to P1 (only `name`/`direction` differ; `min_depth_atr =
1.8`, `target_r = 1.5`, `max_bars_between_detect_and_confirm = 60`):

```python
{
  "version": "1.0", "left_bars": 3, "right_bars": 3, "atr_period": 14,
  "min_separation_bars": 4, "max_equal_atr": 1.5, "min_depth_atr": 3.0,
  "stop_buffer_atr": 0.5, "target_r": 1.5,
  "max_bars_between_detect_and_confirm": 60, "cooldown_bars": 3,
  "symbol": "XAUUSD", "timeframe": "M15",
}
```

`config_hash` (§6.2) + `feature_schema_version = "double-v1.0"` stamped on
every event.

## 7. Features + available_at (§3.4)

Identical schema to P1 (mirrored meaning: `depth_atr` = depth of the middle
low below the highs; `confirm_reclaim_atr` = confirm close **below** neckline
in ATR).  Same detection-time/confirm-time availability split; all
`uses_future_data = False`.

## 8. Dataset builder + labeling

Reuses P1 infra verbatim (`double_bottom/dataset.py`): forward fixed-horizon
MFE/MAE labeling with mirrored convention for shorts (`mfe_ratio` = entry −
min(low) over the horizon, in R; etc.), `label = 1` iff `mfe_ratio ≥ target_r`;
labels read only bars after entry (no look-ahead).  §6.3 gates + report for
double_top live in `double_bottom/artifacts/reports/` (shared tooling).

## 9. Expected event frequency (data audit, XAUUSD M15)

Mirror of P1 — same order of magnitude; exact counts per
`scripts/generate_reports.py` audit (see gate report).

## 10. Synthesizer benchmark (§8.3)

Same harness as P1 (`double_bottom/benchmark.py::run_double_benchmark("double_top")`);
gates: recall ≥ 80 %, robustness drop ≤ 15 pts, FP ≤ 5 % — measured values in
`research/patterns/double_bottom/artifacts/reports/synthesizer_benchmark.md`.