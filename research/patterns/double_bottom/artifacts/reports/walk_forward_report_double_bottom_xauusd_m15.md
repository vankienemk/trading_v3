# Walk-Forward Training Report (§6.2/§6.4)

- model_id: `double_bottom_xauusd_m15_v1` · pattern: double_bottom · symbol: XAUUSD · timeframe: M15
- feature_schema_version: double-v1.0 · config_hash: `bb7129921f24`
- trained_at: 2026-09-07T22:27:48.140986+00:00 · lifecycle_state: validated · gate_passed: True
- label: forward MFE fixed 72-bar horizon, target 1.25R · calibrated: True
- events: 382 · train: 228 · OOS: 153 (purge+embargo applied)

## Held-out OOS metrics (final model fit on train only)

| metric | value |
|---|---|
| accuracy | 0.6013 |
| brier | 0.2768 |
| model_gate_cutoff | 0.2670 |
| model_gated_n | 56.0000 |
| n | 153.0000 |
| oos_pf_model_gated | 2.5061 |
| oos_pf_rule | 2.1064 |
| positive_rate | 0.4314 |
| pr_auc | 0.5514 |
| roc_auc | 0.5829 |

## Walk-forward folds (expanding window, purge+embargo)

| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |
|---|---|---|---|---|---|---|
| 0 | 76 | 76 | 0.5328 | 0.5767 | 0.2512 | 1.940 |
| 1 | 152 | 76 | 0.5385 | 0.7908 | 0.1865 | 0.976 |
| 2 | 228 | 76 | 0.5924 | 0.6212 | 0.2862 | 2.240 |
| 3 | 301 | 76 | 0.5098 | 0.4842 | 0.2544 | 2.150 |

- PR-AUC across folds: mean 0.5434 ± 0.0303 (min 0.5098, max 0.5924)

## §6.3 research gates

# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: double_bottom · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 382 (gate ≥ 300) · n_oos: 153 (gate ≥ 100)
- OOS positive rate: 0.431 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.863 (gate ≥ 0.60)
- OOS PF: 2.106 · bootstrap CI lower: 1.595 (gate > 1.0)
- PR-AUC: 0.509 vs baseline 0.431 (gate ≥ 0.453) -- logistic scorer fit on train only
- walk-forward fold PFs: [1.94, 0.98, 2.24, 2.15] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: PASS**

## Notes

Tier-2 meta-model (double_bottom): RandomForest + isotonic calibration, fit on train split only; 25 causal features at detect/confirm bars (≤ known_at); label = forward MFE 72-bar, 1.25R target.
