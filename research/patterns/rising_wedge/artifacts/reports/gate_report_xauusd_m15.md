# Research Gate Report (§6.3) — Agent 4 DoD, XAUUSD M15

Dataset: 204133 M15 bars (2018-01-02 09:00:00+00:00 → 2026-09-03 22:45:00+00:00).  Labeling: forward MFE/MAE, fixed 72-bar horizon, label target 1.25R (see PATTERN_SPECS.md §8).

Note: H&S is a rare reversal (true strict-structure events on M15 ~150/208).  The §6.3 n >= 300 sample gate is calibrated for the common double patterns; H&S keeps the strict anti-noise detector and reports its gates at the true sample size (forcing n >= 300 would explode §8.3 FP to ~0.7 — a documented tradeoff).

## falling_wedge
- detectors events: 327 · labeled: 327
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: falling_wedge · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 327 (gate ≥ 300) · n_oos: 131 (gate ≥ 100)
- OOS positive rate: 0.702 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.908 (gate ≥ 0.60)
- OOS PF: 2.949 · bootstrap CI lower: 2.161 (gate > 1.0)
- PR-AUC: 0.733 vs baseline 0.702 (gate ≥ 0.737) -- logistic scorer fit on train only
- walk-forward fold PFs: [1.88, 3.28, 2.95, 3.03] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: FAIL**

## rising_wedge
- detectors events: 594 · labeled: 593
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: rising_wedge · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 593 (gate ≥ 300) · n_oos: 237 (gate ≥ 100)
- OOS positive rate: 0.700 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.776 (gate ≥ 0.60)
- OOS PF: 3.191 · bootstrap CI lower: 2.549 (gate > 1.0)
- PR-AUC: 0.746 vs baseline 0.700 (gate ≥ 0.735) -- logistic scorer fit on train only
- walk-forward fold PFs: [4.25, 5.68, 3.48, 2.9] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: PASS**

## head_shoulders
- detectors events: 151 · labeled: 151
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: head_shoulders · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 151 (gate ≥ 300) · n_oos: 60 (gate ≥ 100)
- OOS positive rate: 0.367 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.967 (gate ≥ 0.60)
- OOS PF: 1.037 · bootstrap CI lower: 0.642 (gate > 1.0)
- PR-AUC: 0.485 vs baseline 0.367 (gate ≥ 0.385) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.77, 1.48, 0.75, 1.2] · longest PF≥0.8 run: 1 (gate ≥ 3)

**Overall: FAIL**

## inverse_head_shoulders
- detectors events: 208 · labeled: 208
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: inverse_head_shoulders · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 208 (gate ≥ 300) · n_oos: 83 (gate ≥ 100)
- OOS positive rate: 0.422 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.904 (gate ≥ 0.60)
- OOS PF: 1.416 · bootstrap CI lower: 0.921 (gate > 1.0)
- PR-AUC: 0.658 vs baseline 0.422 (gate ≥ 0.443) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.78, 1.76, 1.56, 1.2] · longest PF≥0.8 run: 3 (gate ≥ 3)

**Overall: FAIL**
