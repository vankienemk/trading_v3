# Walk-Forward Training Report (§6.2/§6.4)

- model_id: `falling_wedge_xauusd_m15_v1` · pattern: falling_wedge · symbol: XAUUSD · timeframe: M15
- feature_schema_version: wedge-v1.0 · config_hash: `ff0e26349b0b`
- trained_at: 2026-09-07T22:28:03.433806+00:00 · lifecycle_state: trained · gate_passed: False
- label: forward MFE fixed 72-bar horizon, target 1.25R · calibrated: True
- events: 327 · train: 196 · OOS: 131 (purge+embargo applied)

## Held-out OOS metrics (final model fit on train only)

| metric | value |
|---|---|
| accuracy | 0.6947 |
| brier | 0.2091 |
| model_gate_cutoff | 0.6293 |
| model_gated_n | 63.0000 |
| n | 131.0000 |
| oos_pf_model_gated | 4.0000 |
| oos_pf_rule | 2.9487 |
| positive_rate | 0.7023 |
| pr_auc | 0.7779 |
| roc_auc | 0.6385 |

## Walk-forward folds (expanding window, purge+embargo)

| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |
|---|---|---|---|---|---|---|
| 0 | 65 | 65 | 0.5726 | 0.5153 | 0.2767 | 1.884 |
| 1 | 130 | 65 | 0.8274 | 0.6959 | 0.2189 | 3.277 |
| 2 | 195 | 65 | 0.8064 | 0.7167 | 0.2012 | 2.946 |
| 3 | 260 | 65 | 0.8044 | 0.6270 | 0.2037 | 3.026 |

- PR-AUC across folds: mean 0.7527 ± 0.1044 (min 0.5726, max 0.8274)

## §6.3 research gates

# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: falling_wedge · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 327 (gate ≥ 300) · n_oos: 131 (gate ≥ 100)
- OOS positive rate: 0.702 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.908 (gate ≥ 0.60)
- OOS PF: 2.949 · bootstrap CI lower: 2.161 (gate > 1.0)
- PR-AUC: 0.733 vs baseline 0.702 (gate ≥ 0.737) -- logistic scorer fit on train only
- walk-forward fold PFs: [1.88, 3.28, 2.95, 3.03] · longest PF≥0.8 run: 4 (gate ≥ 3)

**Overall: FAIL**

## Notes

Tier-2 meta-model (falling_wedge): RandomForest + isotonic calibration, fit on train split only; 28 causal features at detect/confirm bars (≤ known_at); label = forward MFE 72-bar, 1.25R target. EXPLORE status: sample n=327 (gate ≥ 300); gate_passed=False — kept strict anti-noise detector per Agent 4 DoD; model registered as 'trained', promotion deferred until t2 verdict.
