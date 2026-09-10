# XAUUSD Liquidity Sweep — Detection & Scoring System

Phát hiện và chấm điểm **Liquidity Sweep** trên khung **XAUUSD M15**, triển khai
bởi đội 8 agent theo `huong_dan.md`. Nguyên tắc bắt buộc: không dùng dữ liệu
tương lai (no look-ahead), tách setup/confirmation/label, backtest tính đủ chi
phí (spread/slippage/commission/stop buffer), kết quả tái lập được.

## Cấu trúc

```text
configs/            baseline + features + labeling + model (deep-merged by src/config.py)
docs/               SCHEMAS.md (schema lock), INTERFACES.md (module contracts)
src/
  config.py         config loader / merge / path resolution
  config_schema.py  declarative config validation (raises ConfigError)
  schema.py         canonical column constants (machine twin of SCHEMAS.md)
  data/             loader, validator, resampler
  indicators/       atr, volume, trend (all causal)
  liquidity/        rolling, swing, equal levels (known_at causal)
  events/           sweep detector, confirmation, deduplication   [agents 2–3]
  features/         feature pipeline                               [agent 4]
  labeling/         triple barrier, excursions, outcomes           [agent 5]
  scoring/          rule score, expected value, calibration        [agent 6]
  modeling/         dataset, split, train, evaluate, inference     [agent 6]
  pipelines/        CLI entry points: audit / events / dataset / train
tests/              unit + contract + no_lookahead (marker) suites
scripts/ci_gate.sh  mandatory CI/CD gate (ruff + mypy + pytest + no-lookahead)
reports/            data_quality/, integration/, event_charts/, metrics/
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Pipeline 1 — data audit (raw MT5 CSV → normalized Parquet + quality report)
.venv/bin/xauusd-audit --config baseline.yaml

# Pipeline 2 — events (sweep + confirmation, causal group_rule="first")
.venv/bin/xauusd-events --config baseline.yaml

# Pipeline 3 — labeled dataset (features + labels + costs + rule score)
.venv/bin/xauusd-dataset --config baseline.yaml

# Pipeline 4 — train (split + walk-forward + calibration + eval → artifacts)
.venv/bin/xauusd-train --config baseline.yaml

# CI/CD gate before any merge (rule 30.4) — rejects on any failure
scripts/ci_gate.sh --with-no-lookahead
```

Baseline freeze `v1.1.0-freeze`: xem `reports/integration/T16_baseline_freeze.md`
(baseline = strategy B confirmed events; walk-forward OOS PR-AUC 0.960; toàn bộ
12 deliverables §33 hiện diện).

## CI/CD gate

`scripts/ci_gate.sh` chạy: `ruff check` (lint) → `mypy` (type check) →
`pytest` (unit tests) → `pytest -m no_lookahead` (bắt buộc khi có thay đổi
thuộc Agent 2 / Agent 4). Merge bị từ chối nếu bất kỳ bước nào fail; Agent 0
không có quyền override gate, chỉ Agent 7 (QA) có thể yêu cầu revert khi phát
hiện leakage sau merge.

## Schema & contracts

- `docs/SCHEMAS.md` — canonical schema + naming conventions, versioned (rule 30.1).
- `docs/INTERFACES.md` — function contracts giữa các module (rule 30.2).
- `src/schema.py` — column constants; đổi schema phải đổi cả ba chỗ + tests + changelog.

## Data

Đầu vào: `data/raw/XAUUSD_M15_202206020915_202608272245.csv` (MT5 export,
tab-separated, `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`).
Processed: `data/processed/xauusd_m15.parquet`. Test set (`data/processed/test_events.parquet`)
chỉ được đọc bởi Agent 6/7 ở bước cuối (rule 30.5).