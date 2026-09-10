# Walk-Forward Training Report (§6.2/§6.4)

- model_id: `double_top_xauusd_m15_v1` · pattern: double_top · symbol: XAUUSD · timeframe: M15
- feature_schema_version: double-v1.0 · config_hash: `bb7129921f24`
- trained_at: 2026-09-07T22:27:52.853860+00:00 · lifecycle_state: validated · gate_passed: True
- label: forward MFE fixed 72-bar horizon, target 1.25R · calibrated: True
- events: 334 · train: 200 · OOS: 134 (purge+embargo applied)

## Held-out OOS metrics (final model fit on train only)

| metric | value |
|---|---|
| accuracy | 0.6194 |
| brier | 0.2550 |
| model_gate_cutoff | 0.2713 |
| model_gated_n | 60.0000 |
| n | 134.0000 |
| oos_pf_model_gated | 1.7095 |
| oos_pf_rule | 1.5933 |
| positive_rate | 0.4328 |
| pr_auc | 0.5924 |
| roc_auc | 0.6470 |

## Walk-forward folds (expanding window, purge+embargo)

| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |
|---|---|---|---|---|---|---|
| 0 | 66 | 66 | 0.3856 | 0.5873 | 0.2125 | 0.924 |
| 1 | 132 | 66 | 0.5841 | 0.7017 | 0.2075 | 1.119 |
| 2 | 196 | 66 | 0.6080 | 0.5886 | 0.2903 | 2.139 |
| 3 | 264 | 66 | 0.5736 | 0.6702 | 0.2295 | 1.337 |

- PR-AUC across folds: mean 0.5378 ± 0.0888 (min 0.3856, max 0.6080)

## §6.3 research gates

# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: double_top · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 334 (gate ≥ 300) · n_oos: 134 (gate ≥ 100)
- OOS positive rate: 0.433 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.933 (gate ≥ 0.60)
- OOS PF: 1.593 · bootstrap CI lower: 1.188 (gate > 1.0)
- PR-AUC: 0.488 vs baseline 0.433 (gate ≥ 0.454) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.92, 1.12, 2.14, 1.34] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: PASS**

## Notes

Tier-2 meta-model (double_top): RandomForest + isotonic calibration, fit on train split only; 25 causal features at detect/confirm bars (≤ known_at); label = forward MFE 72-bar, 1.25R target.
