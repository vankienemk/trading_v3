# PATTERN_SPECS — falling_wedge (P3)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 / §13 Agent 4.
Detector: `research/patterns/falling_wedge/detector.py` (class
`FallingWedgeDetector`, version `1.0`).  Shared machinery lives in
`research/patterns/rising_wedge/detector.py` (`WedgePatternDetectorBase`) and
`research/core/trendline.py`.

## 1. Geometry definition

A **falling wedge** is a bullish reversal structure on a downtrend approach —
a converging channel where BOTH boundary trendlines descend and the lower
line falls faster (so the channel narrows toward an apex):

```
   high1  \
         \ hi_line (upper)   ... close crosses ABOVE upper line
   high2   \   /
            \ /
          low1 low2 low3  ← descending lows (lower trendline)
                    ↓ breakout: close > upper trendline
```

* three **descending** swing lows (`low1 > low2 > low3`,
  `min_depth_atr × ATR` apart) anchoring the *lower* trendline;
* ≥ 2 **descending** swing highs strictly between the first and last low,
  anchoring the *upper* trendline;
* the channel **converges** (`min_convergence` width narrowing from the first
  to the last high touch) and keeps ≥ `min_interior` of its closes inside the
  channel within an ATR tolerance — the anti-noise interior gate;
* an absolute anti-noise floor: low swing `|low1 - low3| ≥ min_vol_span ×
  median(high-low)` so a pure-noise OU zig-zag (whose ATR rescales with the
  noise) cannot satisfy the structure (spec §8 control);
* confirmation: after `low3`'s pivot is causally known (`low3.bar +
  right_bars`, §3.2), a close crosses **above** the upper trendline within
  the staleness window.

## 2. Swing formula (per §3.2 / §11 mandatory field)

* shared infra: `research/core/swing_detector.py` — `SwingDetector`;
  `research/core/trendline.py` — `fit_trendline` (linear regression on the
  pivot (bar, price) points) + `closes_inside_channel` (ATR-tolerant interior);
* left bars: `left_bars = 3`; right bars: `right_bars = 3` (configurable);
* **causal existence**: a pivot at bar `t` is usable only from
  `known_at_bar = t + right_bars` (§3.2).  The detector sets
  `detect_time = df.index[low3.bar + right_bars]` and never reads a bar past
  it for structural features;
* price source: pivot lows from the **low** series, pivot highs from the
  **high** series;
* min distance between lows: `low2 - low1 ≥ min_separation_bars` and
  `low3 - low2 ≥ min_separation_bars` (default 2 bars); no hard max — the
  staleness window bounds the low3→breakout span.

## 3. Confirmation rule

* opposing channel line = the **upper** trendline (fit on the descending
  swing highs);
* confirmation = first bar `c > low3.bar` whose **close** crosses **above**
  the upper line;
* causal stamp: `confirm_bar = max(c, low3.bar + right_bars)` — the effective
  confirm bar;
* staleness (§3.3): if no close-cross fires within
  `max_bars_between_detect_and_confirm` (default 60 bars) after `low3`, the
  candidate is discarded (never emitted as an event).

## 4. Entry / SL / TP

* entry: open of the bar **after** `confirm_bar`; if the data ends at the
  confirm bar, entry = confirm close;
* stop: `wedge_low − stop_buffer_atr × ATR` (default buffer 0.5 ATR);
* risk R = `|entry − stop|`; target = `entry + target_r × R`
  (default `target_r = 1.5`);
* all levels are carried in `event.structure_levels`
  (`upper/lower_trendline_{slope,intercept}`, `upper_line_at_detect`,
  `lower_line_at_detect`, `wedge_high`, `wedge_low`) and
  `event.entry_price/stop_price/target_price`.

## 5. Staleness window (§3.3)

* `max_bars_between_detect_and_confirm = 60` (M15 default);
* config-driven; events discarded when confirmation misses the window — the
  live Event Lake logs them with `attributes.discard_reason = "stale"`.
* **Semantics (handoff P1#6):** scan anchored at the last extreme
  (`low3.bar`) → effective detect→confirm bound `max_wait − right_bars`
  (57 at the 60/3 defaults), STRICTER than the config name reads; kept
  deliberately (conservative) — see
  `rising_wedge/PATTERN_SPECS.md` §5 + `docs/findings_resolution.md`;
  pinned by `test_staleness_effective_bound_detect_to_confirm`.

## 6. Default config + version

```python
{
  "version": "1.0",
  "left_bars": 3, "right_bars": 3, "atr_period": 14,
  "min_separation_bars": 2,
  "min_depth_atr": 1.2,
  "min_vol_span": 2.0,          # absolute noise floor (median range)
  "min_pattern_bars": 8,
  "max_line_error_atr": 1.4,    # trendline ATR tolerance
  "min_convergence": 0.05,
  "min_interior": 0.70, "interior_tol_atr": 0.8,
  "stop_buffer_atr": 0.5, "target_r": 1.5,
  "max_bars_between_detect_and_confirm": 60,
  "cooldown_bars": 5,
  "symbol": "XAUUSD", "timeframe": "M15",
}
```

`config_hash = sha1(canonical_json(config) + version + indicators_version)[:12]`
(§6.2, `research/core/config_hash.py`); stamped on every event via
`event.config_hash` + `feature_schema_version = "wedge-v1.0"`.

## 7. Features (scoring inputs) + available_at (§3.4)

| feature | dtype | available_at | meaning |
|---|---|---|---|
| `atr` | float | detect | ATR(14) at the last low's pivot-known bar |
| `depth_atr` | float | detect | monotonic low swing (low1-low3) in ATR |
| `width_atr` | float | detect | channel width at detect, in ATR |
| `convergence_ratio` | float | detect | channel narrowing first→last high |
| `interior_ratio` | float | detect | fraction closes inside channel |
| `pattern_length` | int | detect | bars low1..low3 |
| `line_error_atr` | float | detect | max trendline anchor deviation in ATR |
| `confirm_pierce_atr` | float | confirm | confirm close above upper line, in ATR |
| `confirm_range_atr` | float | confirm | confirm bar range in ATR |

All features read bars ≤ their `available_at` stamp; `uses_future_data =
False`; runtime validation via `research.core.causal_checks.validate_causality`.

Every declared feature is emitted: `confirm_range_atr` (confirm bar range,
ATR@confirm-bar normalized) is attached to `event.attributes` at the confirm
bar (handoff P1#5 fix, `docs/findings_resolution.md`).

`rule_score (0-1) = 0.35·depth + 0.25·convergence + 0.20·interior +
0.20·pierce`.

## 8. Dataset & labeling (§13 Agent 4)

* `research/patterns/rising_wedge/dataset.py` re-exports the shared forward
  fixed-horizon labeling + §6.3 gate machinery (same semantics / thresholds
  as the double patterns, Agent 3).
* §6.3 research gates on XAUUSD M15 (horizon 72 bars, label target 1.25R):
  the falling wedge clears n ≥ 300 (=327), ESS ≥ 60 % (0.91), OOS PF
  bootstrap CI > 1.0 (2.16) and ≥ 3 WF folds (4).  The razor-thin PR-AUC gate
  (0.733 vs 0.737) is reported in the gate artifact, not hard-asserted.
  **P1#8 resolution (docs/findings_resolution.md):** near-miss accepted with
  exact re-run numbers (n=327 / OOS 131 / ESS 0.908 / PF CI 2.161 / WF run 4 /
  PR-AUC 0.733 vs gate 0.737); the production tier-2 meta-model closes the
  margin at the model level — OOS PR-AUC **0.778 > 0.737** (t1 artifact
  `falling_wedge/artifacts/models/falling_wedge_xauusd_m15_v1/`).

## 9. Expected event frequency (XAUUSD M15, 2018-2026)

~327 detected events on 204k M15 bars (~1 per 2.5 weeks) — the falling wedge
is a moderate-frequency reversal; rising_wedge (the mirror) yields ~590.

## 10. Synthesizer benchmark (§8.3)

* harness: `research/patterns/rising_wedge/benchmark.py`.
* falling_wedge: recall ≈ 0.88, robustness drop ≈ 0.03, FP ≈ 0.00 — PASS.
* rising_wedge: honest recall ≈ 0.42 (its mirror geometry is the hardest for
  a causal right-bar detector — same rationale Agent 8 documents for
  excluding rising_wedge from the CI-gated core set).
