# OOS Edge Test — LSW + DB (XAUUSD, sau phase 2)

**Ngày:** 2026-09-08 · **Runner:** `research/multi_backtest/runner.py` (§12, backtest ≡ live CorrelationManager path)
**Window (OOS):** 2023-10-12 04:45 UTC → 2026-09-03 (boundary = entry của event đầu tiên trong tail 40% dataset double_bottom; chính xác với gate §6.3)
**Setup:** `--mode dedup --costs spread/commission/slippage` (XAUUSD 0.7/0.5/1.0 bps/side) · `--attach-models` (DB tier-2 model; LSW rule-only — runner chỉ attach t1 artifacts, legacy xauusd_v2 KHÔNG được attach)

## Kết quả per-pattern (XAUUSD OOS)

| pattern | events | trades | PF | expectancy R | total R | maxDD R | PR-AUC (OOS ev) |
|---|---|---|---|---|---|---|---|
| **double_bottom** | 153 | 151 | **1.4065** | **+0.1628** | **+24.59** | 6.42 | 0.705 (models) / 0.734 (rule) |
| **liquidity_sweep** | 77 | 75 | **0.9609** | **−0.0323** | **−2.42** | 15.32 | 0.473 |

- Portfolio (dedup): 226 trades · PF **1.181** · exp +0.098R · total **+22.17R** · maxDD 15.83R · 4 rejected by caps
- Correlation matrix §4.2: DB↔LSW **0.00** — không bao giờ cùng group trên OOS → không có synergy dedup
- §12 item-5 (OOS 2025-10-21→2026-09-04): portfolio +4.00R (56t) vs best-single double_bottom **+7.33R** → **REVIEW NEEDED: multi không thắng best-single** (LSW âm kéo xuống)

## Sensitivity

1. **`--no-models` (XAUUSD OOS):** trade set **giống hệt** (151/75 trades, cùng PF/total R) — model tier-2 được attach làm score/ranking nhưng **không threshold-gate** trade nào; PR-AUC events 0.705→0.734 (rule thuần). → Model filter hiện tại KHÔNG đóng góp vào việc chọn trade; muốn lọc thực sự cần threshold config.
2. **EURUSD OOS (`--no-models`, cross-asset):** double_bottom PF **0.74** (−17.5R/110t), liquidity_sweep PF **0.57** (−36.4R/81t) — **cả hai đều âm** → edge phát hiện là **XAUUSD-specific**; DB không generalize cross-asset bằng rule-only (cần train model riêng cho EURUSD như phase-2 mới chỉ train XAUUSD).

## Verdict edge

- ✅ **double_bottom: CÓ EDGE trên XAUUSD OOS** — PF 1.41, +24.6R, expectancy +0.16R, rolling PF chủ yếu >1 (10t:0.77→61t:2.26→151t:1.71). Đồng nhất với gate §6.3 (PF CI 1.595) và model validated. Đây là pattern live chính.
- ❌ **liquidity_sweep: KHÔNG có edge trên OOS gần đây** (rule-only, cost 2.2bps/side) — PF 0.96, −2.4R, rolling PF quanh 1. So với legacy backtest PF 1.305 (có model, khác cost model) → cần: (a) gắn lại legacy model xauusd_v2 để test, hoặc (b) chấp nhận LSW rule-only giảm edge dưới ngưỡng sau costs.
- ⚠️ **Cặp LSW+DB:** dương PF 1.18 nhưng **một mình DB thắng** (item-5 REVIEW NEEDED). Khuyến nghị vận hành: chỉ DB live (hoặc DB+shadow), tạm ẩn LSW khỏi assignment tới khi test lại với legacy model / điều chỉnh detector.

## Evidence files
- `research/multi_backtest/reports/oos_lsw_db/XAUUSD_multi_backtest.{md,json}` (attach-models)
- `research/multi_backtest/reports/oos_lsw_db_nomodels/XAUUSD_multi_backtest.{md,json}`
- `research/multi_backtest/reports/oos_lsw_db_eurusd/EURUSD_multi_backtest.{md,json}`