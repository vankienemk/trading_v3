# Walk-Forward Validation Report

**Generated**: 2026-09-06 08:03
**Dataset**: `data/processed/labeled_events.parquet` (267 events)
**Features**: 30 causal features (no look-ahead)
**Config**: `configs/baseline.yaml` (splitting + model sections)
**Seed**: 42 (frozen per baseline v1.1.0-freeze)

## Aggregate Metrics (Out-of-Sample)

| Metric | Mean | Std | Min | Max | N Folds |
|---|---|---|---|---|---|
| accuracy | 0.8000 | 0.0200 | 0.7800 | 0.8200 | 2 |
| brier_score | 0.1481 | 0.0073 | 0.1409 | 0.1554 | 2 |
| f1 | 0.0769 | 0.0769 | 0.0000 | 0.1538 | 2 |
| positive_rate | 0.2100 | 0.0300 | 0.1800 | 0.2400 | 2 |
| pr_auc | 0.3328 | 0.0996 | 0.2332 | 0.4323 | 2 |
| precision | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 2 |
| recall | 0.0417 | 0.0417 | 0.0000 | 0.0833 | 2 |
| roc_auc | 0.6665 | 0.0473 | 0.6192 | 0.7138 | 2 |
| test_size | 50.0000 | 0.0000 | 50.0000 | 50.0000 | 2 |
| train_size | 132.0000 | 25.0000 | 107.0000 | 157.0000 | 2 |
| val_size | 50.0000 | 0.0000 | 50.0000 | 50.0000 | 2 |

## Per-Fold Details

### Fold 0
- **Train size**: 107 events
- **Validation size**: 50 events
- **Best model**: logistic_regression
- **PR-AUC**: 0.2332
- **ROC-AUC**: 0.6192
- **Brier Score**: 0.1409
- **F1**: 0.0000

### Fold 1
- **Train size**: 157 events
- **Validation size**: 50 events
- **Best model**: random_forest
- **PR-AUC**: 0.4323
- **ROC-AUC**: 0.7138
- **Brier Score**: 0.1554
- **F1**: 0.1538


## Figure

`reports/figures/walk_forward_folds.png` — metrics across folds.

## Notes

- Expanding window: each fold trains on all prior data + step expansion
- Purge + embargo applied between train and validation (embargo_bars=32)
- OOS metrics computed on validation set for each fold
- Stability assessed via mean/std of metrics across folds