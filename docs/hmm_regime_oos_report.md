# HMM Regime Filter — OOS Test Report (LSW / DB / DT)

> **RUN STATUS: FINAL — t10/t11 parity re-run recorded: numbers identical on repaired code (n/PF/totalR/expR/win/maxdd/config/per-state sweep); activation path = DIRECT is_allowed/state_at_confirm_bar (local rule bundle), independent of config_source (F01) and runner fail-closed (F02).**

**Ngày:** 2026-09-08 · **Script:** `research/multi_backtest/scripts/oos_hmm_regime_filter.py`
**Data:** `research/multi_backtest/data/XAUUSD_m15.parquet` — 204133 bars M15 UTC (2018-01-02 09:00:00+00:00 → 2026-09-03 22:45:00+00:00)
**Costs:** CostConfig XAUUSD spread 0.7 / commission 0.5 / slippage 1.0 bps/side (2.2 bps/side); triple-barrier target/stop/horizon 72 bars — `runner.simulate_trade`.

## Protocol (requirements v1.0 §8)

- **HMM fit:** `CausalGaussianHMM` (3 states trending/sideways/high_vol, seed 42, n_iter 100, config_hash §6.2) — train prefix 2018-01-02 → 2023-09-30 (≥1000 bars margin to OOS start 2023-10-12); sanity window uses a strictly-before fit 2018-01-02 → 2019-12-31.
- **States:** causal forward-filtering; state used per event = last closed bar ≤ known_at (`state_at_confirm_bar`, same helper as live ≡ backtest).
- **Gate tuning (LEGACY ONLY):** `allowed_states` x `min_confidence` ∈ {0.00, 0.55, 0.70} swept on legacy window (2018-01-02 → 2022-06-30), config chosen by post-cost PF with n ≥ 15; model threshold swept the same way. **No OOS tuning** (guide §10).
- **Evaluation:** fixed config applied on OOS (2023-10-12 → 2026-09-03) and sanity (2020-01-01 → 2022-06-30, stability only).

## Regime diagnostic — causal HMM state distribution (why the gate matters)

| window | state counts (bars) | share non-dominant |
|---|---|---|
| legacy (2018-01-02 → 2022-06-30) | {'high_vol': 797, 'sideways': 1050, 'trending': 103951} | 1.75% (dominant 98.3%) |
| oos (2023-10-12 → 2026-09-03) | {'trending': 67806} | 0.00% (dominant 100.0%) |
| sanity (2020-01-01 → 2022-06-30) | {'trending': 58836} | 0.00% (dominant 100.0%) |

**Nhận xét (trung thực):** với config canonical (3 states, seed 42, full covariance) trên XAUUSD M15, forward-filter hội tụ về gần như CHỈ MỘT state (`trending`, confidence ≈ 1.0) — OOS window 100% `trending`. HMM regime filter do đó gần như không có selection power trên dữ liệu này: mọi config gate (`allowed_states` x `min_confidence`) giữ gần như toàn bộ trade, kết quả ≈ baseline. Đây là thuộc tính KHÔNG phân biệt regime của chính model (t2 CausalGaussianHMM) với config mặc định, không phải lỗi wiring của script — forward filter deterministic + đã được t2 test incremental==full.

## liquidity_sweep

- **Chosen gate config (legacy):** allowed_states = ['trending'], min_confidence = 0.00
- **Chosen model threshold (legacy):** 0.70

### Window: legacy (2018-01-02 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 121 | 0.888 | -11.43 | -0.094 | 33.1% | 28.14 |
| gate_state | 119 | 0.913 | -8.70 | -0.073 | 33.6% | 25.40 |
| gate_state_conf | 119 | 0.913 | -8.70 | -0.073 | 33.6% | 25.40 |
| gate_top_state | 119 | 0.913 | -8.70 | -0.073 | 33.6% | 25.40 |
| model_gated | 84 | 0.896 | -7.26 | -0.086 | 33.3% | 17.16 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| high_vol | 1 | 0.000 | -1.42 | -1.425 |
| sideways | 1 | 0.000 | -1.31 | -1.307 |
| trending | 119 | 0.913 | -8.70 | -0.073 |

Gate execution audit (anti inert-activation, t7 F01): allowed=119 state_blocked=2 conf_blocked=0 no_regime=0.

### Window: oos (2023-10-12 → 2026-09-03)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 77 | 0.951 | -3.13 | -0.041 | 32.5% | 16.44 |
| gate_state | 77 | 0.951 | -3.13 | -0.041 | 32.5% | 16.44 |
| gate_state_conf | 77 | 0.951 | -3.13 | -0.041 | 32.5% | 16.44 |
| gate_top_state | 77 | 0.951 | -3.13 | -0.041 | 32.5% | 16.44 |
| model_gated | 53 | 0.825 | -7.97 | -0.150 | 28.3% | 14.60 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 77 | 0.951 | -3.13 | -0.041 |

Gate execution audit (anti inert-activation, t7 F01): allowed=77 state_blocked=0 conf_blocked=0 no_regime=0.

### Window: sanity (2020-01-01 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 62 | 0.938 | -3.08 | -0.050 | 33.9% | 14.13 |
| gate_state | 62 | 0.938 | -3.08 | -0.050 | 33.9% | 14.13 |
| gate_state_conf | 62 | 0.938 | -3.08 | -0.050 | 33.9% | 14.13 |
| gate_top_state | 62 | 0.938 | -3.08 | -0.050 | 33.9% | 14.13 |
| model_gated | 47 | 0.931 | -2.59 | -0.055 | 34.0% | 17.16 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 62 | 0.938 | -3.08 | -0.050 |

Gate execution audit (anti inert-activation, t7 F01): allowed=62 state_blocked=0 conf_blocked=0 no_regime=0.

### Verdict (OOS): **NEUTRAL**

**LSW hypothesis (PF ≥ 1.05 hậu cost trên OOS):** NOT ACHIEVED — HMM regime filter không cứu LSW.

## double_bottom

- **Chosen gate config (legacy):** allowed_states = ['trending', 'sideways', 'high_vol'], min_confidence = 0.00
- **Chosen model threshold (legacy):** 0.70

### Window: legacy (2018-01-02 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 178 | 0.798 | -17.17 | -0.096 | 43.3% | 22.71 |
| gate_state | 178 | 0.798 | -17.17 | -0.096 | 43.3% | 22.71 |
| gate_state_conf | 178 | 0.798 | -17.17 | -0.096 | 43.3% | 22.71 |
| gate_top_state | 170 | 0.742 | -21.66 | -0.127 | 41.2% | 22.34 |
| model_gated | 23 | 2.139 | +10.76 | +0.468 | 65.2% | 1.29 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| high_vol | 3 | inf | +2.59 | +0.863 |
| sideways | 5 | 2.743 | +1.90 | +0.380 |
| trending | 170 | 0.742 | -21.66 | -0.127 |

Gate execution audit (anti inert-activation, t7 F01): allowed=178 state_blocked=0 conf_blocked=0 no_regime=0.

### Window: oos (2023-10-12 → 2026-09-03)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 153 | 1.375 | +23.20 | +0.152 | 56.9% | 6.42 |
| gate_state | 153 | 1.375 | +23.20 | +0.152 | 56.9% | 6.42 |
| gate_state_conf | 153 | 1.375 | +23.20 | +0.152 | 56.9% | 6.42 |
| gate_top_state | 153 | 1.375 | +23.20 | +0.152 | 56.9% | 6.42 |
| model_gated | 2 | inf | +2.64 | +1.321 | 100.0% | 0.00 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 153 | 1.375 | +23.20 | +0.152 |

Gate execution audit (anti inert-activation, t7 F01): allowed=153 state_blocked=0 conf_blocked=0 no_regime=0.

### Window: sanity (2020-01-01 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 81 | 0.894 | -3.84 | -0.047 | 46.9% | 13.36 |
| gate_state | 81 | 0.894 | -3.84 | -0.047 | 46.9% | 13.36 |
| gate_state_conf | 81 | 0.894 | -3.84 | -0.047 | 46.9% | 13.36 |
| gate_top_state | 81 | 0.894 | -3.84 | -0.047 | 46.9% | 13.36 |
| model_gated | 12 | 2.298 | +6.11 | +0.509 | 66.7% | 1.29 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 81 | 0.894 | -3.84 | -0.047 |

Gate execution audit (anti inert-activation, t7 F01): allowed=81 state_blocked=0 conf_blocked=0 no_regime=0.

### Verdict (OOS): **NEUTRAL**

## double_top

- **Chosen gate config (legacy):** allowed_states = ['trending'], min_confidence = 0.00
- **Chosen model threshold (legacy):** 0.65

### Window: legacy (2018-01-02 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 171 | 0.640 | -36.50 | -0.213 | 39.8% | 42.97 |
| gate_state | 167 | 0.645 | -35.39 | -0.212 | 39.5% | 42.97 |
| gate_state_conf | 167 | 0.645 | -35.39 | -0.212 | 39.5% | 42.97 |
| gate_top_state | 167 | 0.645 | -35.39 | -0.212 | 39.5% | 42.97 |
| model_gated | 16 | 5.143 | +14.05 | +0.878 | 81.2% | 2.20 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| high_vol | 4 | 0.348 | -1.11 | -0.278 |
| trending | 167 | 0.645 | -35.39 | -0.212 |

Gate execution audit (anti inert-activation, t7 F01): allowed=167 state_blocked=4 conf_blocked=0 no_regime=0.

### Window: oos (2023-10-12 → 2026-09-03)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 101 | 0.700 | -16.76 | -0.166 | 39.6% | 21.09 |
| gate_state | 101 | 0.700 | -16.76 | -0.166 | 39.6% | 21.09 |
| gate_state_conf | 101 | 0.700 | -16.76 | -0.166 | 39.6% | 21.09 |
| gate_top_state | 101 | 0.700 | -16.76 | -0.166 | 39.6% | 21.09 |
| model_gated | 3 | 0.655 | -0.75 | -0.249 | 33.3% | 1.04 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 101 | 0.700 | -16.76 | -0.166 |

Gate execution audit (anti inert-activation, t7 F01): allowed=101 state_blocked=0 conf_blocked=0 no_regime=0.

### Window: sanity (2020-01-01 → 2022-06-30)

| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |
|---|---|---|---|---|---|---|
| baseline | 87 | 0.811 | -8.87 | -0.102 | 42.5% | 14.23 |
| gate_state | 87 | 0.811 | -8.87 | -0.102 | 42.5% | 14.23 |
| gate_state_conf | 87 | 0.811 | -8.87 | -0.102 | 42.5% | 14.23 |
| gate_top_state | 87 | 0.811 | -8.87 | -0.102 | 42.5% | 14.23 |
| model_gated | 11 | 3.230 | +7.56 | +0.687 | 72.7% | 2.20 |

Per-state selection power (post-cost):

| state | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending | 87 | 0.811 | -8.87 | -0.102 |

Gate execution audit (anti inert-activation, t7 F01): allowed=87 state_blocked=0 conf_blocked=0 no_regime=0.

### Verdict (OOS): **NEUTRAL**

## T7-F01/F02 independence (integration review findings)

- **F01 (config_source activation bug)** không ảnh hưởng script này: script KHÔNG gọi `resolve_regime_config`/`RegimeSlot`/`run_symbol_backtest`/`attach_regime_features`; gate được gọi TRỰC TIẾP qua `hard_gate.is_allowed` với rule bundle tự dựng (`{pattern: {allowed_states, min_confidence}}`) — không có đường activation nào có thể silent-inert. Mỗi window in `[gate audit]` (allowed/state_blocked/conf_blocked/no_regime) để chứng minh gate thực sự chạy.
- **F02 (runner thiếu fail-closed §6.3)** không ảnh hưởng: script không đi qua runner scoring; DB/DT model-gated dùng tier-2 cũ (feature_list KHÔNG chứa hmm_*) qua `make_tier2_scorer` — không có model hmm-required nào được score thiếu plugin.
- Bối cảnh: t10 (repair) đã fix cả F01+F02; t11 (review round 2) **PASS**; parity re-run trên code repaired cho numbers **IDENTICAL** (n/PF/totalR/expR/win/maxdd/config/per-state sweep) — `RUN_STATUS` = FINAL (xem header).

## OOS side-by-side — regime-gated vs model-gated vs rule-only

| pattern | variant | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|---|
| liquidity_sweep | rule-only (baseline) | 77 | 0.951 | -3.13 | -0.041 |
| liquidity_sweep | regime-gated (trending, conf=0.00) | 77 | 0.951 | -3.13 | -0.041 |
| liquidity_sweep | model-gated | 53 | 0.825 | -7.97 | -0.150 |
| double_bottom | rule-only (baseline) | 153 | 1.375 | +23.20 | +0.152 |
| double_bottom | regime-gated (trending+sideways+high_vol, conf=0.00) | 153 | 1.375 | +23.20 | +0.152 |
| double_bottom | model-gated | 2 | inf | +2.64 | +1.321 |
| double_top | rule-only (baseline) | 101 | 0.700 | -16.76 | -0.166 |
| double_top | regime-gated (trending, conf=0.00) | 101 | 0.700 | -16.76 | -0.166 |
| double_top | model-gated | 3 | 0.655 | -0.75 | -0.249 |

*Model-gated = legacy LSW calibrator / t1 tier-2 calibrated probs, threshold chọn trên legacy và cố định trên OOS (xem bảng tuning).*

## No-lookahead trong experiment (acceptance #6)

- HMM fit: **chỉ trên train prefix** (2018-01-02 → 2023-09-30 cho legacy/OOS; 2018-01-02 → 2019-12-31 cho sanity) — không bao giờ fit trên vùng sẽ đánh giá.
- State dùng cho event = **state tại known_at** (last closed bar ≤ known_at, qua `state_at_confirm_bar` — cùng helper live ≡ backtest của t3); không bao giờ dùng state tương lai.
- Gate tuning (`allowed_states` x `min_confidence`, model threshold) **CHỈ trên legacy window**; OOS đánh giá với config cố định (không tune trên OOS — guide §10).

## Consistency với các bằng chứng OOS trước

- LSW OOS rule-only: **PF 0.951, n=77, -3.13R** — khớp chính xác `finding/lsw_model_filter_not_enough` (PF 0.951, 77t, -3.13R) và gần `docs/oos_edge_lsw_db.md` (PF 0.9609, 75t — khác vì runner có dedup grouping + caps, script này là per-event simulation như `oos_lsw_model_filter.py`).
- DB OOS rule-only: **PF 1.375, n=153, +23.20R** — sát `docs/oos_edge_lsw_db.md` (PF 1.4065, 151t; 2 trade bị caps trong runner). Edge DB vẫn còn nguyên trên OOS.

## Tuning detail — legacy sweep (allowed_states x min_confidence)

### liquidity_sweep

| config | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending|conf=0.00 | 119 | 0.913 | -8.70 | -0.073 |
| trending|conf=0.55 | 119 | 0.913 | -8.70 | -0.073 |
| trending|conf=0.70 | 119 | 0.913 | -8.70 | -0.073 |
| sideways|conf=0.00 | 1 | 0.000 | -1.31 | -1.307 |
| sideways|conf=0.55 | 1 | 0.000 | -1.31 | -1.307 |
| sideways|conf=0.70 | 1 | 0.000 | -1.31 | -1.307 |
| high_vol|conf=0.00 | 1 | 0.000 | -1.42 | -1.425 |
| high_vol|conf=0.55 | 1 | 0.000 | -1.42 | -1.425 |
| high_vol|conf=0.70 | 1 | 0.000 | -1.42 | -1.425 |
| trending+sideways|conf=0.00 | 120 | 0.901 | -10.01 | -0.083 |
| trending+sideways|conf=0.55 | 120 | 0.901 | -10.01 | -0.083 |
| trending+sideways|conf=0.70 | 120 | 0.901 | -10.01 | -0.083 |
| trending+high_vol|conf=0.00 | 120 | 0.900 | -10.12 | -0.084 |
| trending+high_vol|conf=0.55 | 120 | 0.900 | -10.12 | -0.084 |
| trending+high_vol|conf=0.70 | 120 | 0.900 | -10.12 | -0.084 |
| sideways+high_vol|conf=0.00 | 2 | 0.000 | -2.73 | -1.366 |
| sideways+high_vol|conf=0.55 | 2 | 0.000 | -2.73 | -1.366 |
| sideways+high_vol|conf=0.70 | 2 | 0.000 | -2.73 | -1.366 |
| trending+sideways+high_vol|conf=0.00 | 121 | 0.888 | -11.43 | -0.094 |
| trending+sideways+high_vol|conf=0.55 | 121 | 0.888 | -11.43 | -0.094 |
| trending+sideways+high_vol|conf=0.70 | 121 | 0.888 | -11.43 | -0.094 |

| model threshold | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| thr=0.50 | 85 | 0.879 | -8.60 | -0.101 |
| thr=0.55 | 85 | 0.879 | -8.60 | -0.101 |
| thr=0.60 | 85 | 0.879 | -8.60 | -0.101 |
| thr=0.65 | 85 | 0.879 | -8.60 | -0.101 |
| thr=0.70 | 84 | 0.896 | -7.26 | -0.086 |

### double_bottom

| config | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending|conf=0.00 | 170 | 0.742 | -21.66 | -0.127 |
| trending|conf=0.55 | 170 | 0.742 | -21.66 | -0.127 |
| trending|conf=0.70 | 170 | 0.742 | -21.66 | -0.127 |
| sideways|conf=0.00 | 5 | 2.743 | +1.90 | +0.380 |
| sideways|conf=0.55 | 5 | 2.743 | +1.90 | +0.380 |
| sideways|conf=0.70 | 4 | 2.524 | +1.66 | +0.416 |
| high_vol|conf=0.00 | 3 | inf | +2.59 | +0.863 |
| high_vol|conf=0.55 | 3 | inf | +2.59 | +0.863 |
| high_vol|conf=0.70 | 3 | inf | +2.59 | +0.863 |
| trending+sideways|conf=0.00 | 175 | 0.768 | -19.76 | -0.113 |
| trending+sideways|conf=0.55 | 175 | 0.768 | -19.76 | -0.113 |
| trending+sideways|conf=0.70 | 174 | 0.765 | -19.99 | -0.115 |
| trending+high_vol|conf=0.00 | 173 | 0.773 | -19.07 | -0.110 |
| trending+high_vol|conf=0.55 | 173 | 0.773 | -19.07 | -0.110 |
| trending+high_vol|conf=0.70 | 173 | 0.773 | -19.07 | -0.110 |
| sideways+high_vol|conf=0.00 | 8 | 5.118 | +4.49 | +0.561 |
| sideways+high_vol|conf=0.55 | 8 | 5.118 | +4.49 | +0.561 |
| sideways+high_vol|conf=0.70 | 7 | 4.899 | +4.25 | +0.608 |
| trending+sideways+high_vol|conf=0.00 | 178 | 0.798 | -17.17 | -0.096 |
| trending+sideways+high_vol|conf=0.55 | 178 | 0.798 | -17.17 | -0.096 |
| trending+sideways+high_vol|conf=0.70 | 177 | 0.796 | -17.40 | -0.098 |

| model threshold | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| thr=0.50 | 41 | 1.795 | +14.08 | +0.343 |
| thr=0.55 | 38 | 1.889 | +14.30 | +0.376 |
| thr=0.60 | 30 | 1.700 | +9.60 | +0.320 |
| thr=0.65 | 25 | 2.032 | +10.94 | +0.438 |
| thr=0.70 | 23 | 2.139 | +10.76 | +0.468 |

### double_top

| config | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| trending|conf=0.00 | 167 | 0.645 | -35.39 | -0.212 |
| trending|conf=0.55 | 167 | 0.645 | -35.39 | -0.212 |
| trending|conf=0.70 | 167 | 0.645 | -35.39 | -0.212 |
| sideways|conf=0.00 | 0 | nan | +0.00 | +nan |
| sideways|conf=0.55 | 0 | nan | +0.00 | +nan |
| sideways|conf=0.70 | 0 | nan | +0.00 | +nan |
| high_vol|conf=0.00 | 4 | 0.348 | -1.11 | -0.278 |
| high_vol|conf=0.55 | 4 | 0.348 | -1.11 | -0.278 |
| high_vol|conf=0.70 | 4 | 0.348 | -1.11 | -0.278 |
| trending+sideways|conf=0.00 | 167 | 0.645 | -35.39 | -0.212 |
| trending+sideways|conf=0.55 | 167 | 0.645 | -35.39 | -0.212 |
| trending+sideways|conf=0.70 | 167 | 0.645 | -35.39 | -0.212 |
| trending+high_vol|conf=0.00 | 171 | 0.640 | -36.50 | -0.213 |
| trending+high_vol|conf=0.55 | 171 | 0.640 | -36.50 | -0.213 |
| trending+high_vol|conf=0.70 | 171 | 0.640 | -36.50 | -0.213 |
| sideways+high_vol|conf=0.00 | 4 | 0.348 | -1.11 | -0.278 |
| sideways+high_vol|conf=0.55 | 4 | 0.348 | -1.11 | -0.278 |
| sideways+high_vol|conf=0.70 | 4 | 0.348 | -1.11 | -0.278 |
| trending+sideways+high_vol|conf=0.00 | 171 | 0.640 | -36.50 | -0.213 |
| trending+sideways+high_vol|conf=0.55 | 171 | 0.640 | -36.50 | -0.213 |
| trending+sideways+high_vol|conf=0.70 | 171 | 0.640 | -36.50 | -0.213 |

| model threshold | n_trades | PF | totalR | expR/trade |
|---|---|---|---|---|
| thr=0.50 | 34 | 2.509 | +18.78 | +0.552 |
| thr=0.55 | 26 | 2.252 | +12.79 | +0.492 |
| thr=0.60 | 23 | 2.721 | +13.70 | +0.595 |
| thr=0.65 | 16 | 5.143 | +14.05 | +0.878 |
| thr=0.70 | 12 | 3.553 | +8.66 | +0.721 |

## OOS model-threshold diagnostic (transparency — threshold đã cố định từ legacy, KHÔNG phải quyết định)

| pattern | thr=0.50 | thr=0.55 | thr=0.60 | thr=0.65 | thr=0.70 |
|---|---|---|---|---|---|
| liquidity_sweep | 0.825 (n=53) | 0.825 (n=53) | 0.825 (n=53) | 0.825 (n=53) | 0.825 (n=53) |
| double_bottom | 4.305 (n=11) | 5.605 (n=7) | 4.446 (n=6) | 4.016 (n=5) | inf (n=2) |
| double_top | 0.778 (n=9) | 0.778 (n=9) | 0.645 (n=4) | 0.655 (n=3) | 0.655 (n=3) |

## Optional §8.5 — DB model trained WITH hmm_* features

- Train events (legacy window): **178**; OOS events: **153**.
- OOS PR-AUC (hmm-feature RandomForest + isotonic): **0.548**.
- OOS gated PF tại threshold **0.50** (sweep TRÊN CHÍNH OOS events, max-PF, n≥15): **1.466** (n=24, totalR=+4.28R).
- ⚠️ Trung thực (T8-F02): đây là **OOS-optimistic exploration**, KHÔNG phải config chọn trên legacy, **KHÔNG dùng cho bất kỳ verdict gate nào** — chỉ là bằng chứng phụ §8.5 cho biết hmm_* features có giúp model mới hay không; threshold sweep trên chính OOS nên con số PF bị optimistic bias.
- So sánh: model DB t1 tier-2 cũ (không hmm features) trên OOS window — xem bảng `model_gated` ở trên (threshold chọn trên legacy).

## Kết luận trung thực

### Verdict summary (OOS window, §8.4)

| pattern | baseline PF (n) | gated PF (n) | verdict | LSW hypothesis |
|---|---|---|---|---|
| liquidity_sweep | 0.951 (n=77) | 0.951 (n=77) | NEUTRAL | NOT ACHIEVED |
| double_bottom | 1.375 (n=153) | 1.375 (n=153) | NEUTRAL | n/a |
| double_top | 0.700 (n=101) | 0.700 (n=101) | NEUTRAL | n/a |

**Kết luận:** với config canonical của `CausalGaussianHMM` (3 states, seed 42, full covariance) trên XAUUSD M15, regime filter **không cải thiện post-cost edge** trên bất kỳ pattern nào trong OOS — lý do chính là model không phân biệt regime trên dữ liệu này (OOS 100% một state), nên gate ≈ no-op. Điều này KHÔNG có nghĩa plugin sai — forward filtration đúng causal (t2 tests) — mà là config mặc định không có selection power trên XAUUSD M15; cần tuning HMM config (n_states, input_features, covariance) TRƯỚC khi kỳ vọng gate giúp.

- Báo cáo này ghi NHẬN kết quả thực tế, kể cả khi bộ lọc HMM regime KHÔNG cải thiện post-cost edge (bài học `lsw_model_filter_not_enough`: model/regime filter có thể có selection power nhưng không đủ bù cost thật).
