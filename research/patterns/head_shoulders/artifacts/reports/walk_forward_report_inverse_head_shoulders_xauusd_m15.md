# Walk-Forward Training Report (§6.2/§6.4)

- model_id: `inverse_head_shoulders_xauusd_m15_v1` · pattern: inverse_head_shoulders · symbol: XAUUSD · timeframe: M15
- feature_schema_version: hs-v1.0 · config_hash: `811747a66c49`
- trained_at: 2026-09-07T22:28:12.591170+00:00 · lifecycle_state: trained · gate_passed: False
- label: forward MFE fixed 72-bar horizon, target 1.25R · calibrated: True
- events: 208 · train: 124 · OOS: 83 (purge+embargo applied)

## Held-out OOS metrics (final model fit on train only)

| metric | value |
|---|---|
| accuracy | 0.6627 |
| brier | 0.2364 |
| model_gate_cutoff | 0.1952 |
| model_gated_n | 66.0000 |
| n | 83.0000 |
| oos_pf_model_gated | 1.3697 |
| oos_pf_rule | 1.4158 |
| positive_rate | 0.4217 |
| pr_auc | 0.5852 |
| roc_auc | 0.6458 |

## Walk-forward folds (expanding window, purge+embargo)

| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |
|---|---|---|---|---|---|---|
| 0 | 41 | 41 | 0.3792 | 0.5676 | 0.1171 | 0.779 |
| 1 | 82 | 41 | 0.3413 | 0.4762 | 0.2653 | 1.761 |
| 2 | 123 | 41 | 0.5958 | 0.6039 | 0.2594 | 1.559 |
| 3 | 164 | 41 | 0.6937 | 0.8128 | 0.1758 | 1.196 |

- PR-AUC across folds: mean 0.5025 ± 0.1470 (min 0.3413, max 0.6937)

## §6.3 research gates

# Research Gate Report (§6.3)

- symbol: XAUUSD · pattern: inverse_head_shoulders · timeframe: M15 · horizon: 72 bars · target: 1.25R
- n_total: 208 (gate ≥ 300) · n_oos: 83 (gate ≥ 100)
- OOS positive rate: 0.422 (gate ∈ [0.10, 0.90])
- ESS ratio: 0.904 (gate ≥ 0.60)
- OOS PF: 1.416 · bootstrap CI lower: 0.921 (gate > 1.0)
- PR-AUC: 0.658 vs baseline 0.422 (gate ≥ 0.443) -- logistic scorer fit on train only
- walk-forward fold PFs: [0.78, 1.76, 1.56, 1.2] · longest PF≥0.8 run: 3 (gate ≥ 3)

**Overall: FAIL**

## Notes

Tier-2 meta-model (inverse_head_shoulders): RandomForest + isotonic calibration, fit on train split only; 28 causal features at detect/confirm bars (≤ known_at); label = forward MFE 72-bar, 1.25R target. EXPLORE status: sample n=208 (gate ≥ 300); gate_passed=False — kept strict anti-noise detector per Agent 4 DoD; model registered as 'trained', promotion deferred until t2 verdict.
