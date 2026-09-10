# PATTERN_SPECS — head_shoulders (P4)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 / §13 Agent 4.
Detector: `research/patterns/head_shoulders/detector.py` (class
`HeadShouldersDetector`, version `1.0`).  Shared machinery: the 5-point
alternation lives in `HeadShouldersDetectorBase`, swings from
`research/core/swing_detector.py`.

## 1. Geometry definition

A (regular) **head & shoulders** is a bearish reversal off a higher-middle
peak:

```
        LSH         HEAD           RSH
          \         /   \         /
           \       /     \       /
            \  T1 /       \ T2 /   ← neckline (higher trough)
             \   /         \  /
              \/           \/
                  ↓ breakout: close < neckline
```

* five swings `H-L-H-L-H`: left shoulder (H1), neckline trough 1 (L1), head
  (H — strictly ABOVE both shoulders), neckline trough 2 (L2), right shoulder
  (H2);
* the two shoulder peaks near-equal (`|H1 − H2| ≤ max_shoulder_atr × ATR`),
  the two neckline troughs near-equal (`|L1 − L2| ≤ max_neck_atr × ATR`);
* real head depth: `head − max(L1, L2) ≥ min_head_depth_atr × ATR` AND
  `≥ min_head_vol_span × median(high-low)` (absolute noise floor — an OU
  control whose ATR rescales with noise cannot satisfy it, spec §8);
* confirmation: after the RIGHT shoulder's pivot is causally known
  (`right_shoulder.bar + right_bars`, §3.2), a close crosses **below** the
  neckline (`neck = max(L1, L2)`) within the staleness window.

## 2. Swing formula (per §3.2 / §11 mandatory field)

* shared infra: `research/core/swing_detector.py` — `SwingDetector`;
* left bars: `left_bars = 3`; right bars: `right_bars = 3` (configurable);
* **causal existence**: a pivot at bar `t` is usable only from
  `known_at_bar = t + right_bars` (§3.2).  The detector sets
  `detect_time = df.index[right_shoulder.bar + right_bars]` and never reads a
  bar past it for structural features;
* price source: pivot highs from the **high** series, pivot lows from the
  **low** series;
* min separation between consecutive swings:
  `min_separation_bars` (default 2 bars); the staleness window bounds the
  right-shoulder→breakout span.

## 3. Confirmation rule

* neckline = the higher of the two troughs `neck = max(L1, L2)`;
* confirmation = first bar `c > right_shoulder.bar` whose **close** crosses
  **below** `neck`;
* causal stamp: `confirm_bar = max(c, right_shoulder.bar + right_bars)` —
  the effective confirm bar;
* staleness (§3.3): if no close-cross fires within
  `max_bars_between_detect_and_confirm` (default **60 bars on M15** — H&S is
  the long-staleness pattern per spec §11), the candidate is discarded
  (never emitted as an event).

## 4. Entry / SL / TP

* entry: open of the bar **after** `confirm_bar`; if the data ends at the
  confirm bar, entry = confirm close;
* stop: `head + stop_buffer_atr × ATR` (default buffer 0.5 ATR);
* risk R = `|entry − stop|`; target = `entry − target_r × R`
  (default `target_r = 1.5`);
* levels carried in `event.structure_levels` (`neckline`, `head_level`,
  `left_shoulder_level`, `right_shoulder_level`, `head_shoulders_level`).

## 5. Staleness window (§3.3)

* `max_bars_between_detect_and_confirm = 60` (M15 default) — the long
  staleness window that H&S most needs (the spec calls it out as the pattern
  where §3.3 matters most);
* config-driven; events discarded when confirmation misses the window — the
  live Event Lake logs them with `attributes.discard_reason = "stale"`.
* **Semantics (handoff P1#6 / docs/findings_resolution.md):** the confirm
  close-cross scan is anchored at the LAST EXTREME bar (`right_shoulder.bar +
  1 + max_wait`), making the effective detect→confirm bound
  `max_wait − right_bars` (57 at the 60/3 defaults) — STRICTER than the config
  key name reads.  Kept deliberately (conservative: never accepts a late
  confirm; relaxing would change event counts and invalidate trained models);
  pinned by `test_staleness_effective_bound_detect_to_confirm`.

## 6. Default config + version

```python
{
  "version": "1.0", "left_bars": 3, "right_bars": 3, "atr_period": 14,
  "min_separation_bars": 2,
  "max_shoulder_atr": 1.5, "max_neck_atr": 1.5,
  "min_head_depth_atr": 2.0,
  "min_head_vol_span": 4.0,     # absolute noise floor (median range)
  "min_pattern_bars": 12,
  "stop_buffer_atr": 0.5, "target_r": 1.5,
  "max_bars_between_detect_and_confirm": 60,
  "cooldown_bars": 5,
  "symbol": "XAUUSD", "timeframe": "M15",
}
```

`config_hash = sha1(canonical_json(config) + version + indicators_version)[:12]`
(§6.2); stamped via `event.config_hash` + `feature_schema_version = "hs-v1.0"`.

## 7. Features (scoring inputs) + available_at (§3.4)

| feature | dtype | available_at | meaning |
|---|---|---|---|
| `atr` | float | detect | ATR(14) at the right-shoulder known bar |
| `head_depth_atr` | float | detect | head vs neckline depth in ATR |
| `shoulder_offset_atr` | float | detect | left/right shoulder offset in ATR |
| `neckline_offset_atr` | float | detect | two neckline troughs offset in ATR |
| `symmetry_ratio` | float | detect | left/right arm-length symmetry ∈ [0,1] |
| `pattern_length` | int | detect | bars between the two shoulders |
| `confirm_pierce_atr` | float | confirm | confirm close past neckline, in ATR |
| `confirm_range_atr` | float | confirm | confirm bar range in ATR |

`rule_score (0-1) = 0.40·head_depth + 0.20·shoulder_eq + 0.15·neckline_eq +
0.15·symmetry + 0.10·pierce`.

Every declared feature is emitted: `confirm_range_atr` (confirm bar range,
ATR@confirm-bar normalized) is attached to `event.attributes` at the confirm
bar (handoff P1#5 fix, `docs/findings_resolution.md`).

## 8. Dataset & labeling (§13 Agent 4)

* `research/patterns/head_shoulders/dataset.py` re-exports the shared forward
  fixed-horizon labeling + §6.3 gate machinery (same as Agent 3).
* **Sample-size caveat**: H&S is a RARE high-quality reversal.  On XAUUSD M15
  (2018-2026) the strict, FP-safe detector yields ~151 events (inverse ~208) —
  below the §6.3 n ≥ 300 floor calibrated for the common double patterns.
  Forcing n ≥ 300 on real data requires gutting the anti-noise geometry and
  explodes §8.3 FP to ~0.7 (a provable tradeoff).  The detector is therefore
  kept strict; the gate artifact reports n/ESS/PF-CI/PR-AUC/WF honestly at
  the true sample size rather than inflating it with false positives.
* **P1#9 resolution (docs/findings_resolution.md):** D1 exploration committed
  (`scripts/d1_exploration.py` + `artifacts/reports/d1_sample_exploration.csv`)
  — resampling the full 2018-2026 M15 history to D1 (2239 bars) yields only
  **3 events per H&S family**, i.e. D1 cannot support the n ≥ 300 sample gate
  either (spec §11 "H&S chỉ giá trị thật ở D1" notwithstanding, the true D1
  occurrence is far too sparse to train/gate on).  Accepted at the true M15
  sample size (151 / 208) with the strict detector — no gate fudging.

## 9. Expected event frequency (XAUUSD M15, 2018-2026)

regular ≈ 151 events, inverse ≈ 208 events over 204k M15 bars (~1 per 5-6
weeks) — the rarest of the reversal patterns, which is exactly why the spec
ranks H&S as the *hardest* (P4) and flags its long staleness window.

## 10. Synthesizer benchmark (§8.3)

* harness: `research/patterns/head_shoulders/benchmark.py`.
* head_shoulders: recall ≈ 0.93, FP ≈ 0.013 — PASS.
* inverse_head_shoulders: recall ≈ 0.95, FP ≈ 0.007 — PASS.
