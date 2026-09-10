# Research Gate Report (§6.3) — Agent 3 DoD, XAUUSD M15

Dataset: 204133 M15 bars (2018-01-02 09:00:00+00:00 → 2026-09-03 22:45:00+00:00).  Labeling: forward MFE/MAE, fixed 72-bar horizon, label target 1.25R (see PATTERN_SPECS.md §8).

## double_bottom
- detectors events: 382 · labeled events: 382
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: double_bottom · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 382 (gate ≥ 300) · n_oos: 153 (gate ≥ 100)
- OOS positive rate: 0.431 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.863 (gate ≥ 0.60)
- OOS PF: 2.106 · bootstrap CI lower: 1.595 (gate > 1.0)
- PR-AUC: 0.509 vs baseline 0.431 (gate ≥ 0.453) -- logistic scorer fit on train only
- walk-forward fold PFs: [1.94, 0.98, 2.24, 2.15] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: PASS**

## double_top
- detectors events: 334 · labeled events: 334
# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: double_top · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 334 (gate ≥ 300) · n_oos: 134 (gate ≥ 100)
- OOS positive rate: 0.433 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.933 (gate ≥ 0.60)
- OOS PF: 1.593 · bootstrap CI lower: 1.188 (gate > 1.0)
- PR-AUC: 0.488 vs baseline 0.433 (gate ≥ 0.454) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.92, 1.12, 2.14, 1.34] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: PASS**
