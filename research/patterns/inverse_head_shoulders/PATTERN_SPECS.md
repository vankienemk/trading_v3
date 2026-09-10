# PATTERN_SPECS — inverse_head_shoulders (P4, mirror of head_shoulders)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 / §13 Agent 4.
Detector: `research/patterns/inverse_head_shoulders/detector.py` (class
`InverseHeadShouldersDetector`, version `1.0`).

Inverse Head & Shoulders is the exact mirror of Head & Shoulders and reuses
**100 % of P4 infra** — `HeadShouldersDetectorBase` + `SwingDetector`.

## 1. Geometry definition

A (inverse) **head & shoulders** is a bullish reversal off a lower-middle low:

* five swings `L-H-L-H-L`: left shoulder trough, neckline peak 1, head (strictly
  BELOW both shoulder troughs), neckline peak 2, right shoulder trough;
* shoulder troughs near-equal (`max_shoulder_atr`), neckline peaks near-equal
  (`max_neck_atr`);
* real head depth against the neckline (`min_head_depth_atr × ATR` AND
  `min_head_vol_span × median(high-low)` absolute floor);
* confirmation: after the right shoulder's pivot is causally known
  (`right_shoulder.bar + right_bars`, §3.2), a close crosses **above** the
  neckline (`neck = min(L1, L2)`) within the staleness window.

## 2-7. Swing formula / Confirmation / Entry-SL-TP / Staleness / Config / Features

Mirrored from head_shoulders: `direction = "bullish"`,
`short_name = "IHS"`, `kind_seq = ("L","H","L","H","L")`,
`feature_schema_version = "hs-v1.0"`, same default config, swing formula,
causal existence stamps, staleness default 60 bars (§3.3), same feature
schema (available_at detect / confirm), and the same §8 sample-size caveat.
**P1#5/P1#6 (docs/findings_resolution.md):** `confirm_range_atr` is emitted at
the confirm bar (F1 populated); staleness scan anchored at the last extreme →
effective detect→confirm bound `max_wait − right_bars` = 57 (F2 kept stricter,
documented — see `head_shoulders/PATTERN_SPECS.md` §5).

`rule_score` identical to head_shoulders (head_depth + shoulder_eq +
neckline_eq + symmetry + pierce).

## 8-10. Dataset / Frequency / Benchmark

Same shared dataset + labeling machinery as head_shoulders.  On XAUUSD M15
(2018-2026): ~208 events (rarer than double patterns; §6.3 n ≥ 300 sample gate
reported at true size).  **P1#9 (docs/findings_resolution.md):** the committed
D1 exploration (`head_shoulders/scripts/d1_exploration.py`) shows only 3 IHS
events on 10 years of D1 — D1 cannot support the sample gate; accepted at the
true M15 sample size with the strict detector.  §8.3 synthesizer benchmark:
recall ≈ 0.95, FP ≈ 0.007 — PASS.
