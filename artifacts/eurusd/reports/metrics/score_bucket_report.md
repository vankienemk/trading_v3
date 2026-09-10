# Score Bucket Performance Report

**Generated**: 2026-09-06 08:03
**Dataset**: `data/processed/labeled_events.parquet`
**Total events**: 267
**Config**: `configs/baseline.yaml` (scoring.weights + score_buckets)

## Bucket Table

| Bucket | Events | Win Rate | Loss Rate | Ambiguous | Avg MFE (R) | Avg MAE (R) | Avg Net (R) | Profit Factor |
|---|---|---|---|---|---|---|---|---|
| 0-39 | 194 | 18.04% | 50.00% | 0.00% | 1.5372 | 1.6522 | -0.0359 | 0.9343 |
| 40-49 | 70 | 15.71% | 38.57% | 0.00% | 1.3363 | 1.0228 | 0.1520 | 1.3618 |
| 50-59 | 3 | 0.00% | 33.33% | 0.00% | 0.3079 | 0.7388 | -0.1684 | 0.6098 |

## Score Distribution

- Mean rule_score: 34.94
- Std rule_score: 6.79
- Min: 16.00, Max: 52.00

## Outcome Distribution

- **sl**: 125 (46.8%)
- **time**: 96 (36.0%)
- **tp**: 46 (17.2%)

## Figure

`reports/figures/score_bucket_performance.png` — win rate, profit factor,
avg net result, and event distribution by score bucket.

## Notes

- Rule score range: 0-100 (guide §19)
- Buckets: 0-39 / 40-49 / 50-59 / 60-69 / 70-79 / 80-100
- Costs included in net_result_r (guide §17)
- Small sample size per bucket — interpret cautiously