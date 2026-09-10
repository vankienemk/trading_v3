# BTCUSD M15 Training Run — Acceptance Contract (Requirements)

Status: FROZEN — acceptance criteria for the BTCUSD M15 training run
Base pipeline: `xauusd-liquidity-sweep` V2 (`pipeline_v2`)
Precedent: EURUSD cross-asset run (`eurusd_override.yaml`, `pipeline_v2/artifacts/models/eurusd/`, `pipeline_v2/reports/metrics/eurusd_*`)
Scope (in): `trading_live/research/xauusd-liquidity-sweep/`
Scope (out): live trading code, paper-trading config, risk guard changes.

---

## 1. Data source

| Item | Value |
|---|---|
| Raw file | `data/raw/BTCUSDm_M1_202107030000_202609060000.csv` |
| Resolution | **M1** (2021-07-03 → 2026-09-06; ~2.72M rows) |
| Format | MT5 tab-separated export: `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>` |
| Pipeline timeframe | **M15** after causal M1→M15 resample |
| Processed output | `data/processed/btcusd_m15.parquet` |

## 2. M1 → M15 resampling (causal)

- Aggregation per 15-minute bin: `open = first`, `high = max`, `low = min`, `close = last`, `volume = sum`.
- **Causality**: each M15 candle aggregates only M1 bars **fully inside its bin** (closed M1 bars up to and including bin end). A candle whose bin ends at time `T` is only usable from `T` onward — never while forming.
- Timestamps follow the pipeline convention (bin start time, like `XAUUSD_M15` / `EURUSD_M15` raw feeds); downstream HTF-context merges use the existing close-time relabeling + `shift(1)` so no still-forming candle is visible.
- Existing helper to use: `src/data/resampler.py::resample_ohlcv` (already implements OHLCV aggregation; must be applied on the normalized M1 frame with rule `15min`).
- The audit stage (`xauusd-audit`) validates the **M15** processed frame (15-min spacing, gaps, OHLC invariants).

## 3. Override config

- File: `configs/btcusd_override.yaml` (deep-merged on top of `configs/baseline.yaml`), mirroring `eurusd_override.yaml`:

```yaml
project:
  symbol: BTCUSD

data:
  input_path: data/raw/BTCUSDm_M1_202107030000_202609060000.csv
  output_path: data/processed/btcusd_m15.parquet
```

- All stage CLIs accept `--config baseline.yaml --override btcusd_override.yaml` (repeatable, deep-merged by `src/config.py::load_config`).

## 4. Pipeline stages (run in order, with CLI entry points)

| # | Stage | Entry point | Key args |
|---|---|---|---|
| 1 | **audit** | `xauusd-audit` (`src.pipelines.build_events:data_audit_cli`) | `--config baseline.yaml --override btcusd_override.yaml --report-path reports/data_quality/btcusd_m15_quality.json` |
| 2 | **events** | `python pipeline_v2/scripts/build_events_v2.py --v2` | `--config baseline.yaml --override btcusd_override.yaml --target-r 3.0 --output pipeline_v2/data/events_v2_btcusd.parquet` |
| 3 | **dataset** | `xauusd-dataset` (`src.pipelines.build_dataset:build_dataset_cli`) | `--config baseline.yaml --override btcusd_override.yaml --events pipeline_v2/data/events_v2_btcusd.parquet --output pipeline_v2/artifacts/datasets/liquidity_sweep_events_btcusd.parquet` |
| 4 | **train** | `xauusd-train` (`src.pipelines.train_model:train_model_cli`) | `--config baseline.yaml --override btcusd_override.yaml --dataset pipeline_v2/artifacts/datasets/liquidity_sweep_events_btcusd.parquet --output-dir pipeline_v2/artifacts/models/btcusd --metrics pipeline_v2/reports/metrics/btcusd_test_metrics.json` |

Notes:
- Stage 2 uses the V2 sweep detector (`nguoc_trend` + `max_penetration_atr ≤ 0.20`, `target_r = 3.0` per frozen `v2_frozen.yaml`).
- Per-symbol paths: prefix v2 artifacts with the symbol so XAUUSD/EURUSD outputs are never overwritten: `events_v2_btcusd.parquet`, `liquidity_sweep_events_btcusd.parquet`, `models/btcusd/`.
- Walk-forward report: generated from the btcusd labeled dataset (same style as `scripts/generate_eurusd_reports.py`) into `pipeline_v2/reports/metrics/walk_forward_report.json` (rename/copy as `btcusd_walk_forward_report.json`) plus `btcusd_report_summary.json`.

## 5. Required artifacts (exact relative paths, under in-scope root)

| Artifact | Path |
|---|---|
| Processed M15 candles | `trading_live/research/xauusd-liquidity-sweep/data/processed/btcusd_m15.parquet` |
| Data-quality audit report | `trading_live/research/xauusd-liquidity-sweep/reports/data_quality/btcusd_m15_quality.json` |
| Events table | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/data/events_v2_btcusd.parquet` |
| Labeled dataset | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/artifacts/datasets/liquidity_sweep_events_btcusd.parquet` |
| Trained model | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/artifacts/models/btcusd/model.pkl` |
| Calibrator | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/artifacts/models/btcusd/calibrator.pkl` |
| Feature schema | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/artifacts/models/btcusd/features.json` (copy of the 32-feature `artifacts/feature_schemas/features.json` emitted for the btcusd dataset run) |
| Test metrics | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/reports/metrics/btcusd_test_metrics.json` |
| Walk-forward report | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/reports/metrics/btcusd_walk_forward_report.json` (+ `.md`, fold figure) |
| Report summary | `trading_live/research/xauusd-liquidity-sweep/pipeline_v2/reports/metrics/btcusd_report_summary.json` |

## 6. Metrics

- **Primary: PR-AUC**
  - On test: `pr_auc` in `btcusd_test_metrics.json` (canonical `average_precision_score`).
  - Walk-forward: `aggregate.pr_auc.mean` in `btcusd_walk_forward_report.json` (expanding folds, report per-fold + mean/std/min/max).
- **Secondary** (where produced by `evaluate_on_test` / WF aggregation):
  - ROC-AUC (`roc_auc`), Brier score (`brier_score`), accuracy, precision, recall, F1.
  - Context: `test_size`, `positive_rate`, fold counts (`n_folds`), per-fold metrics.
- Metric JSONs MUST NOT be manually edited; regenerate via the pipeline.

## 7. Full-report deliverable (compiled by reporter role)

Required contents of the BTCUSD M15 report:
1. **Overview**: symbol, timeframe (M15), date range (2021-07-03 → 2026-09-06), pipeline version (v2, frozen params), override config used.
2. **Data quality**: link to `btcusd_m15_quality.json`; row counts M1→M15, gap/duplicate/zero-volume/OHLC-invariant summary, UTC handling.
3. **Pipeline execution log**: the 4 stages run in order with CLI commands and exit codes; artifact paths.
4. **Dataset summary**: event counts after each filter (baseline sweeps → v2 nguoc_trend → confirmed → labeled valid), positive rate.
5. **Model results**: best model class, calibration method, `btcusd_test_metrics.json` table (PR-AUC, ROC-AUC, Brier, accuracy, precision, recall, F1).
6. **Walk-forward**: fold table + aggregate (PR-AUC, ROC-AUC, Brier, accuracy) with mean/std/min/max.
7. **Score buckets / top-prob** subset analysis (if generated).
8. **No-look-ahead attestation**: statement that causal resample, closed-HTF merge, and causal labeling were preserved; link to `tests/test_no_lookahead.py` (marker `no_lookahead`).
9. **Registry**: model registered in `trading_live/model_registry/index.yaml` as `btcusd_v2_h16_20260906`-style id with model_path/calibrator_path/feature_schema pointing at the `pipeline_v2/artifacts/models/btcusd/` files.
10. **Conclusions & limitations**: PR-AUC vs XAUUSD/EURUSD context, event count caveats, cost assumptions (frozen `costs` are zero-cost; BTE/break-even note).

## 8. No-look-ahead constraints (mandatory)

1. **Causal M1→M15 resample**: a bin aggregates only closed M1 bars inside the bin; the M15 candle is unavailable while forming.
2. **Closed HTF merge**: `src/data/resampler.py::closed_higher_timeframe_merge` — bins relabeled to close time, `ffill` + `shift(1)`, so an HTF candle closing at `c` is visible only from `c + 15min` onward.
3. **Causal labeling**: triple-barrier labels from `next_open_after_confirmation` entries, `same_bar_policy: ambiguous`, `time_barrier_result: mark_to_market`; no future-candle access (schema §6 / INTERFACES.md cross-cutting #1).
4. **Purge + embargo** time-based split (`splitting.purge: true`, `embargo_bars: 32`); calibration fit on validation only; a single untouched test evaluation.
5. Regression gate: `pytest -m no_lookahead` must remain green for the pipeline modules touched.

## 9. Acceptance checklist (for verification stage)

- [ ] Exact data path used: `data/raw/BTCUSDm_M1_202107030000_202609060000.csv`
- [ ] M1→M15 resample method causal per §2; processed parquet has 15-min spacing
- [ ] Override config at `configs/btcusd_override.yaml` sets symbol=BTCUSD + both data paths
- [ ] 4 stages ran in order via their CLI entry points (§4) with exit 0
- [ ] All artifacts in §5 exist with exact paths
- [ ] `btcusd_test_metrics.json` has `pr_auc`; walk-forward report has `aggregate.pr_auc.mean`
- [ ] Full report covers all 10 §7 sections
- [ ] No-look-ahead constraints re-checked (§8) and attested