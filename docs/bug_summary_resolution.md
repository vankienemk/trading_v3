# Báo cáo xử lý lỗi — `trading_v3_bug_summary.md`

**Ngày xử lý:** 2026-09-10
**Phạm vi:** 11 mục (A1–A4, B1–B5, F1–F2) trong `trading_v3_bug_summary.md`.
**Nguyên tắc:** mỗi mục phải **kiểm chứng bằng code + đo lường thực nghiệm** trước khi sửa. Mục nào đo ra không có lợi (hoặc trái với thiết kế đã cố ý) thì **KHÔNG sửa** và ghi lại bằng chứng thay vì sửa cho có.

---

## Bảng kết quả

| ID | Kết luận | Bằng chứng / thay đổi |
|----|----------|------------------------|
| **A1** | ✅ **ĐÃ SỬA** | `DebugDrawer.mq5` — màu text theo `model_prob` (`<0.40` cam-đỏ, `<0.60` vàng, còn lại xanh/đỏ theo hướng) |
| **A2** | ✅ **ĐÃ SỬA** | `tb = t_confirm + 2 * PeriodSeconds()` (fallback M15) — hết hardcode `2*3600` |
| **A3** | ✅ **ĐÃ SỬA** | `export_debug_events.py::_structure_box` — `end_bar = max(end_bar, confirm_bar)`; ô vàng nay phủ tới nến breakout |
| **A4** | ✅ **ĐÃ SỬA** | Anchor text `ANCHOR_RIGHT_UPPER` — label không còn đè lên vùng lệnh bên phải |
| **B1** | ✅ **ĐÃ SỬA** (opt-in) | Gate `min_model_prob` trong `_group_to_candidate` + `_resolve_min_model_prob`; mặc định **tắt** (không đổi hành vi cũ) |
| **B2** | ✅ **ĐÃ SỬA** | `_normalize_rule_score` theo từng pattern + `_combine_score`; sửa cả 2 call site |
| **B3** | ⚖️ **ĐO RỒI — KHÔNG SỬA** | Đo toàn bộ 204.133 nến: non-strict cho swings 40.126→40.357 (+0.6%), DT 334→**333**, DB 382→**376** (ít hơn!) |
| **B4** | ⚖️ **ĐO RỒI — KHÔNG SỬA** | Dedupe theo `extreme1_bar`: DB 382→**382**, DT 334→**334** — **không đổi gì** |
| **B5** | ⚠️ **XÁC NHẬN LÀ LỖI THẬT — cần quyết định của bạn** | `signal_polling_engine_v2.py:215,222` (live) vẫn gọi `create_symbol_engine` = **chỉ LSW**; `MultiPatternEngine` chỉ dùng trong test/backtest |
| **F1** | ✅ đã sửa trước đó | giữ nguyên |
| **F2** | ✅ documented | giữ nguyên |

---

## A. Lớp vẽ (`docs/debug_drawer/`) — đã sửa & upload

**DebugDrawer.mq5 v2.0** (13032 bytes, đã upload vào `MQL5\Scripts\`):

```cpp
// A2 — vùng lệnh dài đúng 2 NẾN của timeframe đang xem
int periodSec = PeriodSeconds();
if(periodSec <= 0) periodSec = 15 * 60;
datetime tb = t_confirm + 2 * periodSec;

// A1 + A4 — màu theo model_prob + neo phải để không đè vùng lệnh
double pNum = (prob != "" ? StringToDouble(prob) : -1.0);
color txtCol = isRej ? clrGray
             : (pNum < 0.0 ? clrWhite
             : (pNum < 0.40 ? clrOrangeRed
             : (pNum < 0.60 ? clrYellow
                            : (isBuy ? clrLimeGreen : clrRed))));
ObjectSetInteger(0, tname, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
```

**A3 — kiểm chứng cụ thể** (`export_debug_events.py`):

| Event | Trước A3 (structure box) | Sau A3 |
|---|---|---|
| DT 16/07/2026 | `13:00 → 18:30`, giá `4026.71–4081.45` | `13:00 → 07:00 (+1d)`, giá `4022.85–4081.45` |
| DT 17/07/2026 | `03:15 → 06:30`, giá `3975.20–4008.57` | `03:15 → 12:30`, giá `3965.51–4008.57` |

→ Ô vàng **nay bao gồm nến phá neckline** (đúng chỗ lệnh được kích hoạt), không còn "rời khỏi breakout" như mô tả trong báo lỗi.

---

## B. Lớp engine — đã sửa

### B2 — `combined_score` (HIGH, ảnh hưởng toàn pipeline) ✅

**Xác nhận gốc:** hai thang điểm khác nhau tồn tại thật trong repo:

| Pattern | Khai báo | Quan sát thực tế (`debug_events.txt`) |
|---|---|---|
| `liquidity_sweep` | feature schema ghi *"weighted rule score 0..100"* | 32 … 51 |
| `double_bottom` / `double_top` | clamp `min(1.0, …)` | 0.51 … 0.99 |

Vì vậy `rule_score / 100.0` **đúng cho LSW** nhưng **sai cho DB/DT**. Cách sửa không phải "bỏ /100" (sẽ phá LSW) mà là **chuẩn hoá theo từng pattern**:

```python
_RULE_SCORE_SCALE = {"liquidity_sweep": 100.0}   # còn lại auto-detect theo độ lớn

def _normalize_rule_score(pattern_name, rule_score) -> float:
    ...
    scale = _RULE_SCORE_SCALE.get(str(pattern_name).lower())
    if scale is None:
        scale = 100.0 if abs(v) > 1.0 else 1.0     # tự nhận diện plugin mới
    return max(0.0, min(1.0, v / scale))

def _combine_score(rule_score, model_prob) -> float:
    return round((rule_score + model_prob) / 2.0, 4)
```

Áp dụng ở **cả 2 call site**: `_check_new_bar` (legacy LSW — giá trị không đổi) và `_group_to_candidate` (đa pattern — đây là chỗ lỗi).

**Đo trước/sau** cho đúng event trong báo lỗi (DB rule 0.91, prob 0.16):

| | `combined_score` |
|---|---|
| Công thức lỗi cũ `(0.91/100 + 0.16)/2` | **0.0846** |
| Sau khi sửa | **0.5350** |

### B1 — gate `min_model_prob` (opt-in) ✅

Thêm vào `_group_to_candidate` ngay sau khi tính `model_prob`:

```python
prob_min = _resolve_min_model_prob(a)
if prob_min is not None and model_prob < prob_min:
    rep.attributes["discard_reason"] = "low_probability"   # §7.1
    return None
```

- Nguồn ngưỡng: `a.config["min_model_prob"]` (YAML per-symbol/pattern) **hoặc** `_DEFAULT_MIN_MODEL_PROB` (mặc định **rỗng**)
- **Mặc định không lọc** ⇒ hành vi cũ được giữ nguyên (giống cách plugin HMM mặc định OFF), tránh làm thay đổi live ngoài ý muốn
- Bật lên thì event như `p=0.033` (ảnh 1 bạn gửi) sẽ bị chặn và ghi rõ `discard_reason="low_probability"`

---

## B3/B4 — ĐO RỒI, KHÔNG SỬA (kèm bằng chứng)

### B3 — `SwingDetector` strict-low
Báo lỗi giả định đổi `<` → `<=` sẽ **tăng recall**. Đo trên **toàn bộ 204.133 nến XAUUSD M15**:

| | swings | DT events | DB events |
|---|---|---|---|
| Hiện tại (strict) | 40.126 | **334** | **382** |
| Non-strict (đề xuất B3) | 40.357 | **333** | **376** |

→ Thay đổi **không tăng recall**, thậm chí giảm nhẹ 1 DT và 6 DB, đồng thời **phá invariant đã documented** (docstring `swing_detector.py:60–62`: strict để "plateaus do not produce spurious pivots") và làm lệch các con số gate đã pin (DB n=382, DT n=334 trong `PATTERN_SPECS.md`/gate report). **Quyết định: giữ nguyên.**

### B4 — `_dedupe` theo `extreme1_bar`
Áp thử bản vá đề xuất rồi đo: DB **382 → 382**, DT **334 → 334** — **không thay đổi gì** (cooldown theo `confirm_bar` đã bao trùm các ca này). **Quyết định: giữ nguyên**, không thêm code không mang lại hiệu quả.

---

## B5 — lỗi thật, cần bạn quyết định (không tự ý sửa)

**Xác nhận:** đường chạy live thật sự là `signal_polling_engine_v2.py` → `create_symbol_engine()` → pipeline **chỉ LSW**:

```
live/engine/signal_polling_engine_v2.py:215   return create_symbol_engine(symbol, mcp_client=...)
live/engine/signal_polling_engine_v2.py:222   return create_symbol_engine(symbol, ..., symbol_cfg_override=...)
```

Trong khi `MultiPatternEngine` (7 pattern) chỉ được dùng trong **test** và **research/multi_backtest** (grep toàn repo: `tests/test_hmm_regime_integration.py`, `research/multi_backtest/*`). Nghĩa là: **live hiện chỉ scan LSW, DB/DT không được scan live** dù đã có detector + model.

**Vì sao tôi không tự sửa:** đổi đường này = đổi **tập pattern được giao dịch live** (thay đổi hành vi trading thật) — cần bạn quyết định, và cần chạy lại smoke live + gate. Hai phương án đã sẵn sàng:

1. **Thin wrapper (an toàn nhất):** `create_symbol_engine` trả về `MultiPatternEngine(...).check_new_bar` — giữ nguyên chữ ký cho mọi call site hiện có.
2. **Deprecate:** đổi tên thành `_create_legacy_lsw_engine`, cập nhật call site sang `MultiPatternEngine`, thêm cờ config `multi_pattern.enabled`.

Cần bạn chọn (1) hay (2) — tôi sẽ implement kèm test + smoke.

---

## Kiểm thử (verification)

**Test mới:** `tests/test_bug_summary_b1_b2.py` — **23 passed**
- Chuẩn hoá `rule_score`: LSW 40→0.40, DB 0.91→0.91, DT 0.51→0.51, plugin lạ 55→0.55 (auto-detect), giá trị rác → 0.0, clamp [0,1]
- `combined_score`: DB 0.535 (không còn 0.0846), LSW **không đổi** 0.45
- Gate B1: lấy ngưỡng từ config/module, giá trị không hợp lệ bị bỏ qua, biên 0.499/0.5

**Hồi quy (chạy từng file):**

| File | Kết quả |
|---|---|
| `test_liquidity_sweep_golden.py` | **15 passed** (bit-identical giữ nguyên) |
| `test_hmm_regime_plugin.py` | 34 passed, 2 skipped |
| `test_hmm_regime_integration.py` | 33 passed |
| `test_hmm_regime_no_lookahead.py` | 39 passed, 2 skipped |
| `test_hmm_regime_adversarial.py` | 34 passed |
| `test_double_bottom.py` / `test_double_top.py` / `test_core_contracts.py` / `test_correlation.py` / `test_event_lake.py` / `test_head_shoulders.py` / `test_adversarial_lookahead.py` / `test_lifecycle.py` | 173 passed (chạy gộp) |
| `test_multi_backtest.py` | 12 passed |
| `test_live_engine_integration.py` | 10 passed |
| `test_model_registry.py` | 14 passed |
| `test_synthesizer.py` | 34 passed |
| `test_wedges.py` | 26 passed |

**Lưu ý môi trường (không phải lỗi code):** chạy **toàn bộ suite trong 1 process** bị `SIGABRT` (exit 134) do tích luỹ bộ nhớ (golden LSW một mình đã 168 s, đọc full history) và các test **boot GUI PySide6 offscreen** (`test_live_smoke.py::test_assignment_store_is_the_multi_pattern_source` gọi `gui_smoke()`, `test_gui_onboarding_registry.py`). Từng file chạy riêng đều PASS — đây là giới hạn tài nguyên của môi trường, không phải hồi quy từ các bản vá này.

---

## Việc còn lại (cần bạn quyết)

1. **B5** — chọn phương án (1) hay (2) để đưa DB/DT vào scan live.
2. **B1** — có muốn **bật** gate `min_model_prob` không? Nếu có, cho tôi ngưỡng mong muốn (ví dụ DB/DT ≥0.20 hoặc ≥0.40). Hiện tại đang **tắt**.
3. Sau khi bạn kéo lại `DebugDrawer v2.0` trên chart: xác nhận A1 (màu label theo prob) và A3 (ô vàng phủ tới nến breakout) đúng ý.
