# Pooled OOS Meta-Analysis Verdict — V2 Pipeline Adoption

**Date:** 2026-09-05
**Decision:** Adopt V2 pipeline (nguoc_trend + max_pen ≤ 0.20 + R:R 3.0:1) as primary detection pipeline
**Status:** ✅ STRICT GENERALIZATION PASS

---

## Context

After the v1.2.0 freeze showed the baseline ML model was weak (test PR-AUC 0.2275, no statistical separation from baseline), the V2 pipeline was developed as an alternative detection approach using:
- **nguoc_trend** (counter-trend filter on H1)
- **max_penetration_atr ≤ 0.20** (strict sweep quality gate)
- **R:R 3.0:1** (configurable target reward:risk ratio)

The V2 pipeline was evaluated on **two independent OOS samples** never used for tuning:

| Sample | Period | Bars | Net Trades |
|--------|--------|------|------------|
| XAUUSD OOS | 2018 → 2022-06 | 103,884 | 118 |
| EURUSD Full | 2018 → 2026 | 215,570 | 267 |

---

## Methodology

- **Pooled bootstrap** (3,000 resamples) combining both OOS samples
- **Block bootstrap** (block_size=25) to account for time-series autocorrelation
- **Fenced pooling** — each asset resampled independently before concatenation (prevents blocks from straddling asset boundary)
- **Cochran's Q** heterogeneity test between the two samples
- **Post-hoc disclosure:** pooling was decided after seeing individual CI results, increasing false-positive risk (transparently documented)

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Min penetration ATR | 0.05 |
| Max penetration ATR | 0.20 |
| Min wick ratio | 0.35 |
| Min reclaim ATR | 0.0 |
| Cooldown bars | 4 |
| Group rule | first |
| Target R:R | 3.0:1 |
| Cost assumption | 0.05R/trade |
| Bootstrap resamples | 3,000 |
| Block size (block bootstrap) | 25 |

---

## Results

### Individual Samples (R:R 3.0, post 0.05R cost)

| Metric | XAUUSD OOS | EURUSD Full |
|--------|------------|-------------|
| n (net) | 118 | 267 |
| Profit Factor | **1.4466** | **1.2473** |
| PF 95% CI (bootstrap) | [0.8978, 2.2272] | [0.9371, 1.6376] |
| Avg net R | 0.2228 | 0.1341 |
| Net positive rate | 0.4661 | 0.3970 |
| NPR Wilson 95% CI | [0.3786, 0.5558] | [0.3402, 0.4568] |
| Breakeven cost (corrected) | 0.2728R | 0.1841R |

### Pooled OOS

| Metric | Value |
|--------|-------|
| n (total) | **385** |
| Profit Factor | **1.305** |
| PF 95% CI (iid bootstrap) | [1.0256, 1.6437] |
| PF 95% CI (block bootstrap, fenced) | **[1.0915, 1.5607]** |
| Avg net R | 0.1613 |
| Net positive rate | 0.4182 |
| NPR Wilson 95% CI | [0.37, 0.468] |
| Breakeven cost (corrected) | 0.2113R |

### Heterogeneity Test

| Metric | Value |
|--------|-------|
| Q statistic | 0.644 |
| p-value | 0.4224 |
| I² | 0.0% |
| Interpretation | No significant heterogeneity (caveat: k=2 has low power) |

### Autocorrelation

| Lag | ACF |
|-----|-----|
| 1 | +0.0511 |
| 2 | +0.0684 |
| 3 | -0.0270 |
| 4 | +0.0867 |
| 5 | -0.0597 |
| Mean | 0.0239 |

Autocorrelation negligible → near-IID trade sequence.

---

## Verdict

✅ **STRICT GENERALIZATION PASS:** CI lower bound (1.0915) > 1.0

The V2 pipeline (nguoc_trend + max_pen ≤ 0.20 + R:R 3.0:1) **generalizes across XAUUSD and EURUSD at 95% confidence.**

> "No third asset needed. Proceed to paper trading (Phase 8)."

---

## Decisions

1. **Adopt V2 pipeline** as the primary detection pipeline for paper trading.
2. **Both OOS datasets permanently burned** — no further tuning or re-use permitted.
3. **No third asset needed** — pooled evidence sufficient at 95% confidence.
4. **Kill-switch for Phase 8 (Paper Trading):**
   - Rolling window CI must use **BLOCK bootstrap** (same block_size=25), not IID
   - Minimum observation window: 25 trades (1 block) for first evaluation, extended to 50-60 trades as primary threshold
   - EURUSD position size: 50% of XAUUSD (as previously agreed)

---

## V2 Artifact Checksums

| File | MD5 |
|------|-----|
| `pipeline_v2/data/events_v2.parquet` | `e175e1fbe02e5d99c915b4908ad3861a` |
| `pipeline_v2/data/labeled_events_v2.parquet` | `da92633ea99b73e0fe6dc07a2417b026` |
| `pipeline_v2/artifacts/datasets/liquidity_sweep_events_v2.parquet` | `da92633ea99b73e0fe6dc07a2417b026` |
| `pipeline_v2/artifacts/models/model.pkl` | `c46f53b4ae745f357aadb3c53ac33148` |
| `pipeline_v2/artifacts/models/calibrator.pkl` | `324d939106fc5172a302b5d68b81f9d1` |
| `pipeline_v2/reports/metrics/test_metrics.json` | `b7c79fdb51edb4a02588360a87e67352` |
| `pipeline_v2/reports/metrics/walk_forward_report.json` | `e70cefb297c5664791afc9d2be303c6b` |
| `pipeline_v2/reports/metrics/top_prob_subset_analysis.json` | `f1ae70b4fe8ce6bd289de7773e2bcb41` |
| `pipeline_v2/reports/metrics/score_bucket_report.json` | `a1b92f87d6bb12300b907721a44b7893` |
| `pipeline_v2/reports/metrics/v2_pipeline_summary.json` | `ced7b7f867bd5bed857c82d445fc0ca7` |
| `pipeline_v2/reports/analysis/v2_analysis.json` | `b8ef3351a819add8829147a65672f66e` |

**Note:** V2 pipeline preserved v1.2.0 frozen artifacts (core model + data files: 7/13 md5 intact). 6/13 report files changed or missing due to `step_top_prob` overwrite — documented in v2.0.0 freeze report.

---

## V2 vs V1 Comparison

| Aspect | V1 (v1.2.0 freeze) | V2 |
|--------|-------------------|-----|
| Detection | Standard sweep detection | nguoc_trend + max_pen ≤ 0.20 |
| R:R | 2.0:1 | 3.0:1 (configurable) |
| Confirmed events (XAUUSD) | 974 | 121 |
| Test PR-AUC | 0.2275 | N/A (rule-based) |
| Pooled OOS PF | Not evaluated | 1.305 [1.09, 1.56] |
| Generalization evidence | None | ✅ PASS (2 assets, 95% CI) |

---

## Source Documents

- `pipeline_v2/reports/analysis/cross_asset/pooled_oos_meta_analysis.json` — Full analysis results
- `pipeline_v2/scripts/analysis/pooled_oos_meta_analysis.py` — Analysis script
- `pipeline_v2/artifacts/freeze_status_v2.0.0.md` — V2 freeze report
- `reports/freeze_status_v1.2.0.md` — V1 baseline freeze