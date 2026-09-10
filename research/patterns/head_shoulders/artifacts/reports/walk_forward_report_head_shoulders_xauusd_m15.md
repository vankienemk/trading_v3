# Walk-Forward Training Report (§6.2/§6.4)

- model_id: `head_shoulders_xauusd_m15_v1` · pattern: head_shoulders · symbol: XAUUSD · timeframe: M15
- feature_schema_version: hs-v1.0 · config_hash: `811747a66c49`
- trained_at: 2026-09-07T22:28:07.993207+00:00 · lifecycle_state: trained · gate_passed: False
- label: forward MFE fixed 72-bar horizon, target 1.25R · calibrated: True
- events: 151 · train: 91 · OOS: 60 (purge+embargo applied)

## Held-out OOS metrics (final model fit on train only)

| metric | value |
|---|---|
| accuracy | 0.6167 |
| brier | 0.2432 |
| model_gate_cutoff | 0.2625 |
| model_gated_n | 47.0000 |
| n | 60.0000 |
| oos_pf_model_gated | 1.2384 |
| oos_pf_rule | 1.0369 |
| positive_rate | 0.3667 |
| pr_auc | 0.4018 |
| roc_auc | 0.5419 |

## Walk-forward folds (expanding window, purge+embargo)

| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |
|---|---|---|---|---|---|---|
| 0 | 30 | 30 | 0.2284 | 0.4201 | 0.1758 | 0.767 |
| 1 | 59 | 30 | 0.3850 | 0.4676 | 0.2877 | 1.484 |
| 2 | 90 | 30 | 0.3453 | 0.5053 | 0.2158 | 0.748 |
| 3 | 120 | 30 | 0.5301 | 0.6227 | 0.2434 | 1.203 |

- PR-AUC across folds: mean 0.3722 ± 0.1078 (min 0.2284, max 0.5301)

## §6.3 research gates

# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: head_shoulders · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 151 (gate ≥ 300) · n_oos: 60 (gate ≥ 100)
- OOS positive rate: 0.367 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.967 (gate ≥ 0.60)
- OOS PF: 1.037 · bootstrap CI lower: 0.642 (gate > 1.0)
- PR-AUC: 0.485 vs baseline 0.367 (gate ≥ 0.385) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.77, 1.48, 0.75, 1.2] · longest PF≥0.8 run: 1 (gate ≥ 3)

**Overall: FAIL**

## Notes

Tier-2 meta-model (head_shoulders): RandomForest + isotonic calibration, fit on train split only; 28 causal features at detect/confirm bars (≤ known_at); label = forward MFE 72-bar, 1.25R target. EXPLORE status: sample n=151 (gate ≥ 300); gate_passed=False — kept strict anti-noise detector per Agent 4 DoD; model registered as 'trained', promotion deferred until t2 verdict.
