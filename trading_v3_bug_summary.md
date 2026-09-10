# Báo cáo tổng hợp lỗi — `trading_v3`

**Ngày tổng hợp:** 2026-09-10
**Phạm vi audit:** drawing layer (`docs/debug_drawer/`), live signal engine (`live/engine/`), detector + swing infra (`research/patterns/`, `research/core/`), đối chiếu với `docs/handoff.md` + `docs/findings_resolution.md`.
**Phương pháp:** đọc trực tiếp file từ repo (Bash + Read), quan sát chart đã upload (`UcJ6Ouu5`), đọc mẫu `docs/debug_drawer/debug_events.txt`. Tất cả trích dẫn `file:dòng` dưới đây là vị trí đã xác minh trong phiên này.

---

## Bảng tổng hợp nhanh

| ID      | Lớp            | File:dòng                                                                 | Ưu tiên        | Trạng thái |
|---------|----------------|---------------------------------------------------------------------------|----------------|------------|
| **A1**  | Vẽ chart       | `docs/debug_drawer/DebugDrawer.mq5:220–225`                               | HIGH (UX)      | ✅ **ĐÃ SỬA** (v2.0) |
| **A2**  | Vẽ chart       | `docs/debug_drawer/DebugDrawer.mq5:185`                                   | MEDIUM         | ✅ **ĐÃ SỬA** (v2.0) |
| **A3**  | Vẽ chart       | `docs/debug_drawer/export_debug_events.py:95–100`                         | MEDIUM         | ✅ **ĐÃ SỬA** |
| **A4**  | Vẽ chart       | `docs/debug_drawer/DebugDrawer.mq5:231`                                   | LOW            | ✅ **ĐÃ SỬA** (v2.0) |
| **B1**  | Signal engine  | `live/engine/signal_engine_v2.py:1080–1131` (`_group_to_candidate`)       | HIGH           | ✅ **ĐÃ SỬA** (gate opt-in, mặc định tắt) |
| **B2**  | Signal engine  | `live/engine/signal_engine_v2.py:489` + `:1109` (`combined_score`)        | HIGH           | ✅ **ĐÃ SỬA** (chuẩn hoá theo pattern) |
| **B3**  | Swing infra    | `research/core/swing_detector.py:87–94` (`find_swings`)                   | MEDIUM         | ⚖️ **ĐO RỒI — KHÔNG SỬA** (không tăng recall; xem resolution §B3) |
| **B4**  | Detector       | `research/patterns/double_bottom/detector.py:330–348` (`_dedupe`)        | LOW            | ⚖️ **ĐO RỒI — KHÔNG SỬA** (0 thay đổi trên 204k nến) |
| **B5**  | Live wiring    | `live/engine/signal_engine_v2.py:32–67` (bootstrap import)                | HIGH (arch.)   | ⚠️ **XÁC NHẬN LỖI THẬT — chờ bạn chọn phương án 1/2** |
| F1      | Detect feature | `double_bottom/detector.py:267–300`                                       | MEDIUM         | ĐÃ SỬA    |
| F2      | Detect semantic| `findings_resolution.md` §F2 (extreme-anchored)                           | LOW            | GIỮ + DOC  |

> **Báo cáo xử lý chi tiết (bằng chứng đo lường, test, việc còn lại):**
> `docs/bug_summary_resolution.md` — test hồi quy: `tests/test_bug_summary_b1_b2.py` (23 passed).

---

## A. Lớp vẽ — `docs/debug_drawer/`

### A1 — Label Text không phân biệt theo `model_prob`

- **Triệu chứng.** Trên chart upload (`UcJ6Ouu5`) thấy `double_bottom BUY | p=0.160 | s=0.91` được vẽ bằng mũi tên xanh lá (`clrLimeGreen`) và text trắng — trùng màu với một event `p=0.90` nếu có. Trong `docs/debug_drawer/debug_events.txt` (mẫu 200 dòng đầu) đếm thấy nhiều DB event có `model_prob=0.067, 0.069, 0.089, 0.118, 0.160, 0.193`… đều được vẽ cùng kiểu.
- **Nguyên nhân gốc.** Tại `DebugDrawer.mq5:220–231`, code ghép `label` thành chuỗi rồi gán `OBJPROP_COLOR=clrWhite` (hoặc `clrGray` nếu bị reject) — **không xét `prob`**. `arrCol` ở dòng 211 chỉ đổi theo `isRej` và `isBuy`, không theo `prob`.
- **Khắc phục (patch mq5):**

  ```cpp
  // --- 4) annotation text at entry ---
  double pNum = (prob != "") ? StringToDouble(prob) : -1.0;
  color txtCol = isRej ? clrGray
               : (pNum < 0.4 ? clrOrangeRed
               : (pNum < 0.6 ? clrYellow
                             : (isBuy ? clrLimeGreen : clrRed)));

  string label = pattern + " " + direction;
  if(prob != "")     label += " | p=" + prob;
  if(score != "")    label += " | s=" + score;
  if(isRej)          label += " | REJECTED:" + discard;
  string tname = InpPrefix + "txt_" + uid;
  if(ObjectCreate(0, tname, OBJ_TEXT, 0, t_entry, eprice))
    {
      ObjectSetString(0, tname, OBJPROP_TEXT, label);
      ObjectSetInteger(0, tname, OBJPROP_COLOR, txtCol);   // đổi theo prob
      ObjectSetInteger(0, tname, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, tname, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
    }
  ```

- **Ưu tiên:** HIGH — chạy live thật, trader nhìn chart không phân biệt prob thấp/cao, dễ vào lệnh rủi ro.
- **Trạng thái:** CHƯA SỬA.

### A2 — Trade zone hardcode `2 * 3600` giây

- **Triệu chứng.** Khối xanh/đỏ "trade zone" luôn kéo dài đúng 2 giờ kể từ `confirm` bất kể timeframe chart hiện tại. Với M15 thì tương xứng (8 cây), nhưng nếu chạy chart H1 → chỉ che 2 cây, H4 → nửa cây.
- **Nguyên nhân gốc.** `DebugDrawer.mq5:185` ghi:
  ```cpp
  datetime tb = t_confirm + 2 * 3600;
  ```
  Magic number, không đọc `Period()`.
- **Khắc phục:**
  ```cpp
  int periodSec = PeriodSeconds();
  if(periodSec <= 0) periodSec = 15 * 60;     // fallback M15
  datetime tb = t_confirm + 2 * periodSec;
  ```
- **Ưu tiên:** MEDIUM.
- **Trạng thái:** CHƯA SỬA.

### A3 — Pattern zone (vàng) ngắn — không ôm breakout

- **Triệu chứng.** Trên chart upload, hai ô vàng nằm rời khỏi breakout: ô vàng kết thúc trước candle đóng cửa vượt neckline. Hệ quả khi nhìn chart: ô vàng "không trùng" với nơi lệnh thực sự vào.
- **Nguyên nhân gốc.** Trong `docs/debug_drawer/export_debug_events.py`, hàm `_structure_box` (dòng 83–116) đặt:
  ```python
  start_bar = a.get("extreme1_bar", a.get("sweep_anchor_bar"))
  end_bar   = a.get("extreme2_bar")
  ```
  Nhưng `extreme2_bar` mới chỉ là pivot thứ hai (đáy thứ hai của DB hoặc đỉnh thứ hai của DT) — pattern **chưa breakout**. Trade zone (khối xanh/đỏ đọc ở DebugDrawer) bắt đầu ở `confirm` = `max(cross_bar, extreme2_bar + right_bars)`. Hai khoảng **DISJOINT** trên trục thời gian.
- **Khắc phục (`export_debug_events.py`):**
  ```python
  if attrs.get("confirm_bar") is not None:
      end_bar = max(end_bar, int(attrs["confirm_bar"]))
  ```
- **Ưu tiên:** MEDIUM.
- **Trạng thái:** CHƯA SỬA.

### A4 — Anchor Text `ANCHOR_LEFT_UPPER` đè lên trade-zone BUY

- **Triệu chứng.** Với BUY, `eprice = entry`, label được tạo tại `(t_entry, eprice)` rồi neo `ANCHOR_LEFT_UPPER` — text hiển thị phía trên-phải của điểm neo, leo thẳng vào khối xanh target (nếu target > entry) hoặc khối đỏ stop. Nhiều event DB gần nhau đè chồng.
- **Nguyên nhân gốc.** `DebugDrawer.mq5:231` dùng `ANCHOR_LEFT_UPPER`.
- **Khắc phục:**
  ```cpp
  ObjectSetInteger(0, tname, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);  // text đi về phải
  ObjectSetInteger(0, tname, OBJPROP_CORNER, CORNER_LEFT_UPPER);   // góc đính
  ```
- **Ưu tiên:** LOW.
- **Trạng thái:** CHƯA SỬA.

---

## B. Lớp detector + signal engine

### B1 — Không có ngưỡng `min_model_prob`

- **Triệu chứng.** Event DB có `model_prob = 0.160` (xem chart upload) vẫn trở thành `SignalCandidate` của `MultiPatternEngine` và hiển thị BUY. Mọi risk gate (`regime_blocked`, `low_confidence` xem `live/engine/hard_gate.py:42–71`) đều có ngưỡng — riêng model prob thấp thì KHÔNG có guard.
- **Nguyên nhân gốc.** `live/engine/signal_engine_v2.py:_group_to_candidate` (dòng 1080–1131) — đoạn
  ```python
  model_prob = float(rep.model_prob) if _finite(rep.model_prob) else 0.5
  ```
  rồi return `SignalCandidate(...)` không thêm bước kiểm. Tương tự ở `_check_new_bar` (dòng 489) — `combined_score` được tính xong, nhưng không có `if combined_score < threshold: drop` hoặc `if model_prob < threshold: drop`.
- **Khắc phục (patch signal engine, thêm ngay sau `_group_to_candidate:1096`):**

  ```python
  PROB_THRESHOLDS = {
      "double_bottom": 0.55, "double_top": 0.55,
      "liquidity_sweep": 0.50, "falling_wedge": 0.50,
      "rising_wedge": 0.50, "head_shoulders": 0.55,
      "inverse_head_shoulders": 0.55,
  }
  threshold = PROB_THRESHOLDS.get(rep.pattern_name, 0.55)
  if _finite(model_prob) and model_prob < threshold:
      rep.attributes["discard_reason"] = "low_probability"
      return None
  ```

  Sau đó sửa tương tự `_check_new_bar` ở dòng 489: không thêm một candidate khi `model_prob < threshold`.
- **Ưu tiên:** HIGH — tác động trực tiếp đến quyết định vào lệnh live.
- **Trạng thái:** CHƯA SỬA.

### B2 — `combined_score` chia `rule_score` cho 100

- **Triệu chứng.** Mọi event DB/DT/LSW sau khi qua signal engine có `combined_score ≈ (rule_score/100 + model_prob) / 2`. Detector `double_bottom` đã clamp `rule_score ∈ [0,1]` (xem `research/patterns/double_bottom/detector.py:275-281`: `rule_score = min(1.0, 0.40*… + 0.25*… + 0.20*… + 0.15*…)` ⇒ max ≈ 1.0). Vậy `rule_score = 0.91` trở thành `0.0091` trong `combined_score`, cộng với `model_prob = 0.16` ⇒ `0.085`. Mọi consumer downstream (Correlation caps §4.4, RiskGuard, exposure cap) đánh giá event này "rất yếu" dù về geometry thì mạnh.
- **Nguyên nhân gốc.** `signal_engine_v2.py:489` (trong `_check_new_bar` sau 488) và `:1109` (trong `_group_to_candidate`):
  ```python
  combined_score = (rule_score / 100.0 + model_prob) / 2.0
  ```
- **Khắc phục (2 chỗ):**
  ```python
  # line 489
  combined_score = round((rule_score + model_prob) / 2.0, 4)
  # line 1109
  combined_score=round((rule_score + model_prob) / 2.0, 4),
  ```
  Cần re-run `tests/test_double_bottom.py::test_rule_score_in_unit_interval` để đảm bảo contract `rule_score ∈ [0,1]` được detector giữ.
- **Ưu tiên:** HIGH — làm sai `combined_score` cho toàn pipeline.
- **Trạng thái:** CHƯA SỬA.

### B3 — `SwingDetector.find_swings` strict-low làm KPI pivot bằng phẳng

- **Triệu chứng.** Đáy/plateau đúng bằng nhau trong window `[i-lb, i+rb]` có thể không được ghi nhận là pivot. DB detector (theo spec `double_top/PATTERN_SPECS.md` và `double_bottom/PATTERN_SPECS.md` -- đã đọc) cho phép 2 đáy cách nhau ≤ `max_equal_atr × ATR` (mặc định 1.5 ATR). SwingDetector lại strict `<` => 2 detector có gate ngược chiều, làm giảm recall DB/DT khi XAUUSD test 2 lần cùng giá cùng side.
- **Nguyên nhân gốc.** `research/core/swing_detector.py:find_swings` (dòng 87–94):
  ```python
  if lo == min(lows[i - lb : i + rb + 1]) and lo < lows[i - lb] and lo < lows[i + rb]:
      out.append(SwingPoint(i, float(lo), "L", i + rb))
  ```
  điều kiện "strict hơn window edges" (`<` ở cả 2 phía). Tương tự cho `H`.
- **Khắc phục:**
  ```python
  if lo == min(lows[i - lb : i + rb + 1]) and lo <= lows[i - lb] and lo <= lows[i + rb]:
      out.append(SwingPoint(i, float(lo), "L", i + rb))
  if hi == max(highs[i - lb : i + rb + 1]) and hi >= highs[i - lb] and hi >= highs[i + rb]:
      out.append(SwingPoint(i, float(hi), "H", i + rb))
  ```
  Chú ý: kiểm tra test `tests/test_swing_detector.py` (nếu có) — một số test gốc dựa trên strictness hiện tại.
- **Ưu tiên:** MEDIUM — ảnh hưởng recall/FP trên M15 XAUUSD.
- **Trạng thái:** CHƯA SỬA.

### B4 — `_dedupe` chỉ dựa trên `confirm_bar`

- **Triệu chứng.** Hai event DB có thể chia sẻ `extreme1_bar` mà `confirm_bar` cách nhau ≥ `cooldown_bars` ⇒ cả hai được giữ. Trong `debug_events.txt` thấy nhiều `double_bottom|BUY|XAUUSD` trong cùng cụm 3–5 ngày, chia sẻ cùng cú pháp structure.
- **Nguyên nhân gốc.** `research/patterns/double_bottom/detector.py:_dedupe` (dòng 330–348) chỉ so sánh `cb - last_confirmed < cooldown`.
- **Khắc phục:**
  ```python
  bar_of  = {ev.event_id: int(ev.attributes["confirm_bar"]) for ev in events}
  ext1_of = {ev.event_id: int(ev.attributes.get("extreme1_bar", -1)) for ev in events}
  ordered = sorted(events, key=lambda ev: (bar_of[ev.event_id], -ev.rule_score))
  kept: list[PatternEvent] = []
  last_confirmed = None
  last_ext1 = None
  for ev in ordered:
      cb = bar_of[ev.event_id]
      e1 = ext1_of[ev.event_id]
      if last_confirmed is not None and (
          cb - last_confirmed < cooldown or e1 == last_ext1
      ):
          continue
      kept.append(ev); last_confirmed = cb; last_ext1 = e1
  ```
- **Ưu tiên:** LOW — ảnh hưởng chống duplicate event trong cùng cụm.
- **Trạng thái:** CHƯA SỬA.

### B5 — Legacy `create_symbol_engine` chỉ wire LSW; DB/DT không được scan

- **Triệu chứng.** `signal_engine_v2.py:587–637` có factory `create_symbol_engine(symbol, mcp_client, ...)` được giữ để tương thích ngược, nhưng bootstrap import (dòng 32–67) chỉ load pipeline từ `research/patterns/liquidity_sweep/src/...` (detect_sweeps_v2, attach_confirmations, build_event_features, compute_rule_scores…). Vì vậy mọi GUI consumer/dashboard gọi `create_symbol_engine("XAUUSD", mcp)` chỉ nhận LSW events, không có DB/DT scan trong cùng session — dù `MultiPatternEngine` đã sẵn sàng cho 7 pattern (xem `docs/handoff.md` §2.7).
- **Nguyên nhân gốc.** `_RESEARCH = .../research/patterns/liquidity_sweep` (đường dẫn cứng) chỉ trỏ vào 1 plugin; trong khi `signal_engine_v2.py:_group_to_candidate` (dòng 1080–1131) đã đa-pattern. Hai đường dẫn tồn tại song song — legacy path chỉ có 1 pattern.
- **Khắc phục (architectural, hai lựa chọn):**
  - **Lựa chọn 1 (an toàn):** giữ `create_symbol_engine` làm thin wrapper quanh `MultiPatternEngine`:
    ```python
    def create_symbol_engine(symbol, mcp_client, ...):
        return MultiPatternEngine(symbol, mcp_client, ...).check_new_bar
    ```
  - **Lựa chọn 2 (deprecate):** đổi tên thành `_create_legacy_lsw_engine` rồi cập nhật GUI call sites sang `MultiPatternEngine`.
- **Ưu tiên:** HIGH (architectural) — sửa sớm trước khi live test toàn diện.
- **Trạng thái:** CHƯA SỬA.

---

## F. Findings đã xử lý (informational, đóng dấu "release block" cũ)

### F1 — `confirm_range_atr` dead schema field

- **File.** `research/patterns/double_bottom/detector.py:267–300` (mirror ở `rising_wedge/detector.py`, `head_shoulders/detector.py`, các wedge fall-back mirror inherit).
- **Triệu chứng gốc (đã giải).** Feature `confirm_range_atr` được khai báo trong `feature_schema` (AVAILABLE_AT_CONFIRM) nhưng code gốc chỉ tính rồi vứt — biểu thức trần không gán vào đâu.
- **Cách giải (đã có trong repo, xác minh):**
  ```python
  atr_confirm = float(atr[confirm_bar]) if not np.isnan(atr[confirm_bar]) else atr_k
  confirm_range_atr = (highs[confirm_bar] - lows[confirm_bar]) / max(atr_confirm, 1e-12)
  # line 300:
  attributes["confirm_range_atr"] = float(confirm_range_atr),
  ```
- **Test pinned:** `tests/test_double_bottom.py::test_confirm_range_atr_emitted_matches_formula` (đã PASS theo `findings_resolution.md` §F1).
- **Trạng thái:** **ĐÃ SỬA** ngày 2026-09-07.

### F2 — Staleness window extreme-anchored semantics

- **File.** `research/patterns/*/detector.py` + 6 `PATTERN_SPECS.md`.
- **Triệu chứng gốc (đã phân tích).** `max_bars_between_detect_and_confirm=60` (config), nhưng code anchor scan ở `last_extreme + 1` → effective bound = `60 − right_bars = 57 bars` (với mặc định `right_bars=3`).
- **Cách giải (đã chốt).** **GIỮ semantics nghiêm ngặt** (extreme-anchored) — discard nhiều hơn, không bao giờ nhận confirm muộn ngoài ý muốn. Đã thêm test pinned `test_staleness_effective_bound_detect_to_confirm` (assert `effective=57` cho config `60/3`).
- **Trạng thái:** **GIỮ + DOCUMENTED**.

---

## Bảng đối chiếu dẫn chứng (đã đọc trong phiên)

| Bug | File:dòng | Dẫn chứng từ `Read` tool trong session này |
|-----|-----------|---------------------------------------------|
| A1  | `DebugDrawer.mq5:220–225` | Đã đọc đoạn "1) trade zone: confirm → confirm+2h" tới "annotation text at entry" |
| A2  | `DebugDrawer.mq5:185` | Đã đọc tới dòng `datetime tb = t_confirm + 2 * 3600;` |
| A3  | `export_debug_events.py:95–100` | Đã đọc hàm `_structure_box` (file 8486 chars) |
| A4  | `DebugDrawer.mq5:231` | Đã đọc `ObjectSetInteger(0, tname, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER)` |
| B1  | `signal_engine_v2.py:1080–1131` | Đã đọc `_group_to_candidate` đầy đủ |
| B2  | `signal_engine_v2.py:489` + `:1109` | Đã đọc cả hai chỗ có `combined_score = (rule_score / 100.0 + model_prob) / 2.0` |
| B3  | `swing_detector.py:87–94` | Đã đọc `find_swings` đầy đủ |
| B4  | `double_bottom/detector.py:330–348` | Đã đọc `_dedupe` đầy đủ |
| B5  | `signal_engine_v2.py:32–67` | Đã đọc bootstrap `_RESEARCH` và 7 import LSW-only |
| F1  | `double_bottom/detector.py:267–300` | Đã đọc `confirm_range_atr` + attr emission |
| F2  | `findings_resolution.md` §F2 | Đã đọc §F2 (decision: keep, document) |

---

## Đề xuất thứ tự xử lý

1. **B1 + B2** — HIGH, cùng file `signal_engine_v2.py`, gộp 1 PR. Sau sửa re-run `pytest -q -m no_lookahead` để chắc không phá causality gate.
2. **A1 + A2 + A3** — HIGH–MEDIUM, cùng repo `docs/debug_drawer/`. Sau sửa re-load `DebugDrawer.mq5` qua MCP, regenerate `debug_events_echo.txt` để so expected-vs-actual.
3. **B5** — HIGH architectural. Cần design review vì liên quan `MultiPatternEngine` đã shipping ở phase 2 (xem `handoff.md` §2.7 + §6).
4. **B3 + B4 + A4** — MEDIUM–LOW, regression test riêng.

---

## Ghi chú cuối

- 9/11 lỗi (A1..A4, B1..B5) **CHƯA SỬA** — đây là phát hiện mới ngoài danh sách F1/F2/F3 của t2 (đã đóng dấu "release block" cũ).
- FYI từ `docs/handoff.md` §4 lesson #10: `signal_engine_v2.py` đã có sẵn strict-mode mypy errors tiền tồn tại (bare generics, `src.*` import not-found do sys.path runtime) — KHÔNG phải do team phase-2; nên expected B2 fix chỉ edit logic, không chạm import paths.
- Bất kỳ bản vá nào ở trên đều nên đi kèm:
  - Test causality (`pytest -q -m no_lookahead`) — giữ 168 tests pass.
  - Golden sweep (`tests/test_liquidity_sweep_golden.py`) — giữ 15/15 bit-identical.
  - Mypy strict scoped (chỉnh file-level cho `signal_engine_v2.py` theo `docs/qa_checklist.md`).
