# LSW — Chẩn đoán PF legacy 1.305 vs runner OOS 0.96 (2026-09-08)

**Câu hỏi:** Legacy LSW từng PF 1.305 (pooled OOS, n=385) — sau upgrade runner OOS chỉ PF 0.96. Có gì vỡ không?

**Trả lời: KHÔNG vỡ.** Detector bit-identical (golden 15/15 re-run trong t10, config_hash `02ab91d2bb28` y hệt legacy; F1/F2 của phase-2 làm đúng theo yêu cầu, golden không đổi). Sự chênh lệch là do **đánh giá khác nhau**, không phải detector khác nhau.

## Bảng đối chứng (XAUUSD LSW, rule-only, runner triple-barrier 3R target)

| Thí nghiệm | Window | Cost model | Trades | PF | Total R |
|---|---|---|---|---|---|
| Legacy pooled OOS meta (XAUUSD-OOS) | 2018→2022-06 | **flat 0.05R/trade** | 118 | **1.4466** | +0.223R avg |
| Runner zero-cost | 2018→2022-06 | 0.0 | 121 | **1.2912** | +23.23R |
| Runner default-cost | 2018→2022-06 | 2.2 bps/side | 121 | **0.8885** | −11.43R |
| Runner zero-cost | 2023-10→2026-09 | 0.0 | 77 | **1.3106** | +15.84R |
| Runner default-cost | 2023-10→2026-09 | 2.2 bps/side | 75 | **0.9609** | −2.42R |

## Nguyên nhân chênh lệch (3 yếu tố, KHÔNG phải lỗi detector)

1. **Cost model (quan trọng nhất):** legacy dùng `COST_BASE = 0.05` → flat **0.05R/trade**; runner dùng bps thật: `cost_r = 2 × (total_bps/1e4) × entry_price/risk`. Với XAUUSD price ~2000 và stop M15 hẹp (~0.15%), **2.2 bps/side ≈ 0.25–0.29R/trade** — gấp ~6× giả định legacy. Đo trực tiếp: legacy-window drag 34.66R/121t ≈ 0.286R/trade (PF 1.29 → 0.89); recent drag ≈ 0.24R/trade (PF 1.31 → 0.96).
2. **Window:** chỉ số legacy chủ yếu từ XAUUSD-OOS **2018→2022-06** (PF 1.4466 độc lập); test mới là **2023-10→2026**. (Raw edge thực ra KHÔNG suy giảm: zero-cost recent 1.31 ≈ legacy-window 1.29.)
3. **Exit/horizon:** runner dùng horizon cap 72 bars cho target 3R (legacy transform RR3 không cap rõ ràng) → một phần TP 3R trễ không được tính — chênh nhỏ (1.29 vs 1.45 cùng window/cost≈0).

## Kết luận

- **Upgrade không phá LSW** — detector + raw edge còn nguyên (zero-cost PF ~1.29–1.31 trên cả 2 window, ngang legacy).
- **Legacy PF 1.305 (và 1.4466 XAUUSD) từng được đo với giả định cost 0.05R — rẻ hơn thực tế ~6×** → là con số lạc quan. Với cost bps thật, LSW rule-only **hòa vốn (PF ~0.9–0.96)** trên cả 2 window.
- LSW chỉ sống được nếu: (a) model filter (legacy xauusd_v2 hoặc tier-2 mới) nâng win-rate đủ bù cost ≥0.25R/trade, hoặc (b) stop rộng hơn (risk lớn → cost_r nhỏ hơn tương đối), hoặc (c) chấp nhận breakeven và giữ vì đa dạng hóa.

## Evidence files
- `research/multi_backtest/reports/diag_lsw_legacy_win/XAUUSD_multi_backtest.{md,json}` (default cost, legacy window)
- `research/multi_backtest/reports/diag_lsw_legacy_win_zero/XAUUSD_multi_backtest.md` (zero cost, legacy window)
- `research/multi_backtest/reports/diag_lsw_recent_zero/XAUUSD_multi_backtest.md` (zero cost, recent window)
- Legacy nguồn: `trading_live/.../pipeline_v2/reports/analysis/cross_asset/pooled_oos_meta_analysis.json` (COST_BASE=0.05)

---

## PHỤ LỤC — Hướng (a): gắn model filter legacy (2026-09-08)

**Cách làm (script `research/multi_backtest/scripts/oos_lsw_model_filter.py`):** chạy legacy chain 1 lần (242 events full 2018–2026) → 32-feature frame per event → score bằng **calibrator legacy** đúng contract live `_compute_model_prob` (30 numeric features, drop categoricals, fillna 0.0) → gate theo calibrated prob → `simulate_trade` (triple-barrier 3R/1R/h72, cost 2.2bps/side) → tổng hợp theo window × threshold.

**Kết quả (sanity: thr=0.0 tái lập chính xác số runner: legacy-win PF 0.888/−11.43R/121t):**

| Window | Threshold | Trades | PF | Total R |
|---|---|---|---|---|
| legacy (2018→2022H1) | 0.0 (rule) | 121 | 0.888 | −11.43R |
| legacy | 0.5–0.65 | 85 | 0.879 | −8.60R |
| legacy | 0.7 | 84 | 0.896 | −7.26R |
| recent (2023-10→2026) | 0.0 (rule) | 77 | 0.951 | −3.13R |
| recent | ≥0.5 | 53 | 0.825 | −7.97R |
| recent | 0.7 | 53 | 0.825 | −7.97R |

**Selection power (calibrated prob top/bottom halves, post-cost):**
| Window | Half | n | PF | Total R |
|---|---|---|---|---|
| legacy | bottom (prob=0.0) | 60 | 0.719 | −15.71R |
| legacy | top (prob=1.0) | 61 | 1.092 | +4.27R |
| recent | bottom (prob=0.0) | 38 | 0.873 | −4.03R |
| recent | top (prob=1.0) | 39 | **1.029** | +0.91R |

*(Calibrated probs cực phân cực: gần như 0/1 — isotonic trên LR; model hoạt động như hard-classifier.)*

**Kết luận hướng (a): model filter legacy CÓ selection power** (top-prob luôn tốt hơn bottom-prob: 1.09 vs 0.72 legacy, 1.03 vs 0.87 recent) **nhưng KHÔNG đủ bù cost thật** — top concentration trên OOS recent chỉ đạt **PF 1.03 (n=39) ≈ breakeven**, không phải edge khai thác được. Legacy LR (PR-AUC OOS 0.47) + cost 0.25–0.29R/trade ⇒ model không cứu được LSW. Hướng sống còn còn lại: (b) stop rộng hơn (giảm cost_r tương đối) hoặc (c) chấp nhận breakeven như diversification, hoặc redesign detector với chi phí-aware.