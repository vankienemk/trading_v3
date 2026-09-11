# Pattern rework — kiểm chứng và kết quả

**Ngày:** 2026-09-11
**Spec tham chiếu:** `trading_v3_pattern_rework_spec.md`
**Phương pháp:** mỗi đề xuất được (1) đọc code xác minh, (2) cài đặt như một tuỳ chọn
bật/tắt được, (3) **đo trên toàn bộ lịch sử XAUUSD M15** (2018-06 → 2026-09, 204.133 nến)
trước khi quyết định có bật mặc định hay không.

Script đo: `docs/rework_measure.py` → `docs/rework_measure_baseline.json`,
`docs/rework_measure_after.json`.

---

## 0. Kết luận ngắn

| Đề xuất | Verdict | Ghi chú |
|---|---|---|
| §1.3.1 NMS theo structure window | ✅ **ĐÃ CÀI** (bật mặc định) | Diệt 1,7–1,9% event trùng cấu trúc — **không phải 30–40%** như spec dự đoán |
| §1.3.2 sweep `min_separation_bars` | ⏸️ **KHÔNG LÀM** | Cần train-set riêng + chọn theo symbol/TF; chưa có protocol chống overfit |
| §1.3.3 `drop_opposite_overlap` | ✅ **ĐÃ CÀI** (mặc định TẮT, có cờ) | Bắt đúng ca người dùng báo 6/7; phải TẮT mặc định để giữ parity §12 |
| §1.3.4 DebugDrawer skip overlap | ⏸️ **KHÔNG LÀM** | Trùng chức năng §1.3.1/§1.3.3; vẽ ít hơn sẽ che mất bằng chứng |
| §2.3.1 structure stop (neckline) | ⚠️ **CÀI NHƯNG TẮT MẶC ĐỊNH** | **Đo được: làm KẾT QUẢ XẤU ĐI** (winrate 48%→23%) |
| §2.3.2 capped target + min_rr | ⚠️ **CÀI NHƯNG TẮT MẶC ĐỊNH** | Cặp 4 ATR cap + min_rr=1.0 **xoá 96% event** |
| §2.3.3 `v2_target_r` 3.0→2.0 | ⏸️ **CHƯA ĐỔI** | Chờ quyết định của người dùng (đổi hành vi lệnh thật) |
| §2.3.4 partial TP + BE trailing | ⏸️ **CHƯA LÀM** | Cần MCP broker-side partial fill; ngoài phạm vi phiên này |
| §2.3.5 dynamic stop buffer | ⏸️ **KHÔNG LÀM** | Phụ thuộc §2.3.1 — mà §2.3.1 đã đo là có hại |
| §3 re-run backtest | ✅ **ĐÃ ĐO** | Xem §4 — kỳ vọng của spec **không đạt** |

**Tóm tắt một câu:** hai đề xuất về *cấu trúc* (§1.3.1, §1.3.3) là đúng và đã cài;
hai đề xuất về *SL/TP* (§2.3.1, §2.3.2) **đã được đo và chứng minh là phản tác dụng**,
nên được cài dưới dạng tuỳ chọn TẮT để không phá hệ thống đang chạy.

---

## 1. Kiểm chứng code — spec đúng gì, sai gì

### 1.1. Đúng ✅

| Khẳng định spec | Xác minh |
|---|---|
| `_detect_double` quét mọi cặp `(s1,s2,s3)`, không NMS | ✅ `detector.py` vòng `for k in range(len(sw)-2)`, chỉ lọc `kind_seq` + `min_separation_bars` |
| `_dedupe` chỉ theo `confirm_bar` + `cooldown_bars` | ✅ `_dedupe` gom theo `confirm_bar`, bỏ event trong `cooldown` |
| `target_r=1.5` hardcode trong plugin | ✅ `get_default_config` → `"target_r": 1.5` |
| Live engine dùng `v2_target_r=3.0` | ✅ `signal_engine_v2.py:281` default 3.0, đọc từ `v2_frozen.yaml`; dùng ở dòng ~473/493 |
| H&S & wedges cũng `target_r=1.5` | ✅ `head_shoulders/detector.py:144`, `rising_wedge/detector.py:170` |
| `stop_buffer_atr=0.5` cố định | ✅ cả 4 plugin |
| TP thực tế rất xa | ✅ **đo được**: DB median **6,85 ATR**, DT **6,67 ATR** từ entry |

### 1.2. Sai / không chính xác ❌

| Khẳng định spec | Thực tế đo được |
|---|---|
| "overlap event giảm 30–40%" (§7) | NMS chỉ diệt **1,7% (DB) / 1,9% (DT)**. Overlap thật rất thấp: DB/DT chỉ **6/360** event có structure window giao nhau. Spec phóng đại ~20× |
| "`debug_events.txt` 618 → ~400 event" (§3.3) | File thật có **433 event**, không phải 618 (đã kiểm ở phiên trước) |
| "TP = 3R = 9 ATR" (§2.1) | Đúng về *live* (3.0R × risk), nhưng risk thật ~4,5 ATR ⇒ TP ~13 ATR, **xa hơn** spec nói |
| "§2.3.2 cap 4 ATR + min_rr 1.0 sẽ tăng winrate lên 42–48%" (§3.1) | **Sai hoàn toàn.** Cặp này xoá **96% event** (DB 354→15). Bật lên winrate **48%→23%**, expectancy **+0,00R→−0,47R** |
| "§2.3.1 structure SL làm SL gần entry hơn ⇒ tăng winrate" (§2.3.1) | **Sai.** SL gần hơn 5,2× (4,56→0,87 ATR) nhưng winrate **48%→23%**, expectancy **−0,01R→−0,47R** |
| `research/core/dedupe.py` "đã đề cập" (§1.3.3) | File **không tồn tại** trước phiên này; đã tạo mới |
| sweep trong `multi_backtest/runner` (§1.3.2) | Không có cơ chế sweep nào; `multi_backtest` là runner chứ không phải sweeper |

### 1.3. Lỗi toán học gốc rễ của §2.3

Spec đề xuất hai điều **tự mâu thuẫn**:

```
risk = |entry − stop| = depth + buffer  ≥  min_depth_atr(3.0) + 0.5 = 3.5 ATR
                                      đo thực tế: median 4.56 ATR

§2.3.2 đặt cap target = 4.0 × ATR
⇒ realized_rr = 4.0 / (risk/ATR) = 4.0 / 4.56 = 0.88  <  1.0
⇒ vi phạm chính gate min_rr=1.0 mà §2.3.1 đề xuất
⇒ 62% event bị loại; đo được: DB 354 → 15 event
```

Nói cách khác: **cap phải nhỏ hơn risk mới "gần", nhưng gate R:R lại đòi target ≥ risk.**
Hai yêu cầu không thể đồng thời thoả khi stop rộng 4,5 ATR. Spec không phát hiện ra.

---

## 2. Đã cài đặt

### 2.1. §1.3.1 — NMS theo structure window — `research/patterns/double_bottom/detector.py`

`DoublePatternDetectorBase._nms_structure_overlap()`, bật qua `nms_overlap=True` (mặc định).

* Hai candidate coi là **cùng một ý tưởng** khi `[extreme1_bar, extreme2_bar]` giao nhau.
* Giữ event `rule_score` cao hơn; hoà thì theo `extreme1_bar` ⇒ **deterministic**.
* Chạy **trước** `_dedupe`.
* DT thừa hưởng tự động (mirror class) — đúng như spec §4.2 dự đoán.

**Đo được:** DB 360→354, DT 316→310 (diệt 6 event mỗi bên). Cửa sổ sau NMS **đôi một rời nhau**
(được pin bằng test).

### 2.2. §1.3.3 — `drop_opposite_overlap` — `research/core/dedupe.py` (file mới)

Chặn trường hợp §4 grouping không bao giờ bắt: **hai detector đọc cùng một cấu trúc theo hai
chiều ngược nhau** (double top + double bottom quanh cùng swing giữa).

Điều kiện conflict (cả 3 phải đúng):
1. ngược `direction`,
2. `structure_span` giao nhau,
3. `neckline` của bên này nằm trong `span` của bên kia (dùng chung chuỗi pivot).

Giữ bên `rule_score` cao hơn.

**Xác minh trên đúng ca người dùng báo** (chart §0.1 #2, 6/7/2026):
```
double_bottom bullish span=(3718,3730) neck=3724 score=0.877  → GIỮ
double_top    bearish span=(3708,3724) neck=3718 score=0.652  → LOẠI
```
Trên toàn bộ dữ liệu: 664 → 621 event (loại 43 ca ngược chiều dùng chung cấu trúc).

**Quan trọng — mặc định TẮT.** Bật ở live engine làm vỡ hợp đồng **parity §12**
(`test_parity_group_decisions_same_as_live_engine` fail: runner và live ra tập event khác nhau).
Đã sửa đúng cách: thêm cờ `opposite_overlap_guard` ở **cả hai phía**
(`MultiPatternEngine` và `run_symbol_backtest`), mặc định `False`, kèm test pin rằng hai bên
loại **cùng một tập event** khi bật.

### 2.3. §2.3.1/§2.3.2 — stop & target — cài dưới dạng tuỳ chọn, TẮT mặc định

Thêm vào `get_default_config`:
```python
"stop_mode": "legacy",              # hoặc "neckline"
"structure_target_atr": 0.0,        # >0 = target theo ATR thay vì R
"target_cap_atr": 0.0,              # >0 = trần khoảng cách target
"min_rr": 0.0,                      # >0 = loại event có R:R thấp
```
Event nay mang thêm `realized_rr` và `target_capped`.

Hai bug tự phát hiện khi viết test, đã sửa:
* `_nms_structure_overlap` crash `KeyError: 'confirm_bar'` với event không có attribute đó
  → thêm fallback về `extreme2_bar`.
* Stop `neckline` có thể **nằm trên entry** (2/354 ca) vì entry khớp *sau khi* giá đã vượt
  neckline → thêm ràng buộc "chỉ siết, không bao giờ vượt entry".

---

## 3. Kết quả đo — vì sao KHÔNG bật §2.3

### 3.1. Baseline (trước, `stop_mode=legacy`)

| Pattern | n | winrate | expectancy | target/ATR | risk/ATR | R:R |
|---|---|---|---|---|---|---|
| double_bottom | 360 | 48,3% | **+0,000R** | 6,85 | 4,57 | 1,50 |
| double_top | 316 | 40,5% | −0,171R | 6,67 | 4,45 | 1,50 |
| falling_wedge | 310 | 41,6% | −0,239R | 3,02 | 2,01 | 1,50 |
| head_shoulders | 150 | 38,7% | −0,131R | 6,36 | 4,24 | 1,50 |
| inverse_h&s | 203 | 45,3% | −0,033R | 6,58 | 4,38 | 1,50 |
| rising_wedge | 577 | 39,0% | −0,882R | 2,45 | 1,63 | 1,50 |

### 3.2. Áp dụng §2.3.1 + §2.3.2 như spec viết

| Pattern | n | winrate | expectancy | risk/ATR | R:R |
|---|---|---|---|---|---|
| double_bottom | 350 | **23,4%** ↓ | **−0,474R** ↓ | 0,87 | 4,33 |
| double_top | 301 | **22,9%** ↓ | **−0,594R** ↓ | 0,89 | 4,04 |

### 3.3. Quét toàn miền để chắc chắn không bỏ sót phương án tốt hơn

Expectancy theo độ rộng stop `k` (ATR), target = 1,5R:

| stop k (ATR) | DB winrate | DB expectancy | DT winrate | DT expectancy |
|---|---|---|---|---|
| 0,87 (neckline) | 42,4% | **−0,415R** | 38,4% | **−0,519R** |
| 1,50 | 42,1% | −0,225R | 37,1% | −0,348R |
| 2,00 | 42,4% | −0,144R | 36,5% | −0,299R |
| 3,00 | 44,6% | −0,055R | 39,4% | −0,196R |
| **4,56 (legacy)** | **47,2%** | **−0,004R** | **40,6%** | **−0,159R** |
| 5,50 | 49,2% | +0,013R | 41,0% | −0,159R |

**Quan hệ đơn điệu: stop càng rộng, expectancy càng tốt.** Không có giá trị trung gian nào
tốt hơn legacy. Đây là bằng chứng quyết định rằng §2.3.1 đi ngược dữ liệu.

### 3.4. Cơ chế — tại sao spec sai

Đo MFE/MAE theo đơn vị R cũ (risk ≈ 4,56 ATR), horizon 16 nến:

```
MFE: p25=0,36R  p50=0,80R  p75=1,44R  p90=1,65R
MAE: p25=0,33R  p50=0,74R  p75=1,05R  p90=1,22R
```

* Stop neckline rộng **0,19R** ⇒ chỉ cần giá đi ngược 0,19R là dính stop.
  `P(MAE ≥ 0,19R) = 84,7%` — khớp đúng tỉ lệ thua đo được.
* Target 4 ATR = **0,88R** ⇒ `P(MFE ≥ 0,877R) = 46,6%` — khớp tỉ lệ thắng.

Biên độ giá sau breakout chỉ ~0,8R (≈3,6 ATR). **Stop rộng chính là thứ giúp trade sống sót
qua biên độ hẹp đó** — nó không phải lỗi thiết kế, mà là hệ quả của `min_depth_atr=3.0`.
Siết stop làm tăng tần suất dính stop nhanh hơn mức tăng của winrate.

---

## 4. Kiểm thử

**Test mới:** `tests/test_pattern_rework.py` — **20 test, pass hết**.

Bao gồm test *cố ý ghi lại* rằng combo literal của spec phá hệ thống
(`test_spec_combo_would_destroy_the_signal_set`), để không ai áp lại nhầm.

**Hồi quy:** toàn bộ `tests/` (trừ `test_live_smoke.py` boot GUI PySide6) — xem kết quả ở
mục cuối. Riêng `test_multi_backtest.py` từng **fail** ở bước cài §1.3.3 mặc định; đã sửa
bằng cờ parity hai phía và nay pass.

**Lint:** `ruff` sạch trên toàn bộ `research/`, `tests/`, `live/engine/signal_engine_v2.py`.

**Sự cố môi trường đã xử lý:** venv `/tmp/ptv2_venv` bị hỏng gói `pytest`
(`pytest/` và `_pytest/` rỗng) giữa phiên; đã cài lại vào `/tmp/pytest_fix` và chạy bằng
`PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages`.

---

## 5. Việc còn lại — cần bạn quyết

1. **Có bật `opposite_overlap_guard` không?** Đo được: bỏ 43/664 event là hai chiều ngược nhau
   dùng chung cấu trúc. Bật thì phải bật **cả hai phía** (live + backtest).
2. **`v2_target_r`: 3.0 → 2.0?** Spec §2.3.3 đề xuất. Đây là **thay đổi lệnh thật**, tôi không
   tự đổi. Đo baseline cho thấy target xa là *có thật* (6,7 ATR), nhưng siết nó lại làm
   expectancy xấu đi — cần bàn kỹ trước khi đổi.
3. **`§1.3.2` sweep `min_separation_bars`** — chỉ nên làm khi có protocol train/test tách bạch,
   nếu không sẽ là overfit trên chính dữ liệu đang đánh giá.
4. **`§1.3.4` DebugDrawer skip overlap** — có muốn tôi làm không? Nó sẽ ẩn bớt hình vẽ, và
   theo tôi là *phản tác dụng cho việc debug*.

---

## 6. Kết luận trung thực

Đề xuất **cấu trúc** của spec (§1.3.1, §1.3.3) là hợp lý và đã được cài — nhưng quy mô vấn đề
nhỏ hơn spec mô tả khoảng 20 lần (1,8% chứ không phải 30–40%).

Đề xuất **SL/TP** (§2.3.1, §2.3.2) nghe hợp lý trên chart nhưng **sai khi đo**: chúng làm
winrate giảm một nửa và expectancy chuyển từ hoà vốn sang lỗ rõ rệt. Nguyên nhân là spec
đề xuất cap target 4 ATR trong khi stop rộng 4,5 ATR — hai điều kiện không thể đồng thời thoả.
Tôi giữ chúng dưới dạng tuỳ chọn **tắt mặc định**, có test pin cả hai mặt, để nếu sau này có
bằng chứng mới thì bật lên chỉ là đổi một dòng.
