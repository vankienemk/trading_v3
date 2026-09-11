# Captain decision memo — trend/HMM rework

**Ngày:** 2026-09-11
**Team:** `trend-hmm-rework`
**Request:** `rework/trading_v3_rework_request_trend_hmm.md`
**Trạng thái:** 2/3 đề xuất chính đã được ĐO và cho kết quả **không như kỳ vọng**. Quyết định bên dưới dựa trên số đo, không dựa trên suy luận.

---

## 0. Vì sao có memo này

Bản yêu cầu §4 đặt ra nguyên tắc "đo trước, bật sau". Team đã tuân thủ đúng — và chính vì vậy **phát hiện ra bản yêu cầu có 2 tiền đề sai**. Memo này ghi lại bằng chứng và chốt hướng đi, để không ai tiêu effort vào một gate đã chứng minh là vô hiệu.

Ba nguồn đo **độc lập** cùng cho một kết quả:
- `measurement_analyst` (t6, đo trên 195.893 nến)
- `trend_gate_engineer` (t4, sweep tham số riêng)
- **captain** (đo lại độc lập, xem §2 bên dưới)

---

## 1. §2.2 `max_pattern_length_bars` — TIỀN ĐỀ SAI, gate vô hiệu

**Bản yêu cầu §1.2 nói:** "range 55+ nến vẫn được chấp nhận như một double-top/double-bottom hợp lệ".

**Đo thực tế** (`pattern_length = extreme2_bar - extreme1_bar`):

| Pattern | n | min | p50 | p95 | **MAX** | số event > 40 nến |
|---|---|---|---|---|---|---|
| double_bottom | 354 | 8 | 14 | 22 | **29** | **0** |
| double_top | 310 | 8 | 14 | 21 | **26** | **0** |

**Không có event nào vượt 29 nến.** Nguyên nhân: `min_separation_bars` cộng với staleness window
(`max_bars_between_detect_and_confirm=60`) đã chặn span từ trước. Đặt gate ở 40–60 như yêu cầu
đề xuất ⇒ **xoá 0 event, thay đổi 0 metric**.

> ⚠️ **ĐÍNH CHÍNH (code_verifier, t1 — finding HIGH #3):** `pattern_length` **KHÔNG tồn tại**
> trong `PatternEvent.attributes`. `detector.py:100-103` chỉ *khai báo* nó trong `feature_schema`;
> `_detect_double` chỉ tính `left_len`/`right_len`. Con số trong bảng trên là **đúng** vì được
> tính trực tiếp bằng `i3 - i1`, nhưng **gate_engineer phải tự tính `extreme2_bar - extreme1_bar`**,
> KHÔNG được đọc `attributes["pattern_length"]` — đọc sẽ `KeyError`.
> Đây cũng là một defect thật: feature đã khai báo `AVAILABLE_AT_DETECT` nhưng chưa bao giờ
> được populate. Cần ghi nhận như một finding riêng.

**Quyết định:** vẫn cài gate như *defence-in-depth* với default `30` (biên an toàn chứng minh
được là 0-drop), nhưng **phải ghi rõ trong code và trong báo cáo rằng nó không cải thiện gì** —
không được ngụ ý rằng nó sửa được triệu chứng "sideways range".

---

## 2. §2.1 trend-context gate — gate cắt 2/3 event nhưng KHÔNG có selection power

Đây là phát hiện quan trọng nhất. Gate **có hoạt động** (nó loại được event), nhưng nó **loại
tín hiệu và nhiễu với tỉ lệ như nhau** — tức không phân biệt được event tốt và event xấu.

**Đo của captain** (lookback=30, min_slope_atr=0.05, min_r2=0.3), so sánh expectancy của
event ĐƯỢC GIỮ vs event BỊ LOẠI:

| Pattern | giữ lại | exp(giữ) | bị loại | exp(loại) | Kết luận |
|---|---|---|---|---|---|
| double_bottom | 118 | **−0.0101R** | 236 | **−0.0097R** | gần như **giống hệt** |
| double_top | 105 | **−0.1599R** | 205 | **−0.1695R** | gần như **giống hệt** |

Nếu gate có selection power thật, expectancy của nhóm giữ lại phải **tốt hơn rõ rệt**. Ở đây
chênh lệch nằm trong nhiễu (0.0004R và 0.0096R). Gate chỉ làm **mất 2/3 cơ hội giao dịch** mà
không đổi chất lượng.

**Baseline (mọi gate tắt), đo độc lập — khớp với measurement_analyst:**

| Pattern | n | winrate | expectancy |
|---|---|---|---|
| double_bottom | 354 | 0.4802 | **−0.0099R** |
| double_top | 310 | 0.4065 | **−0.1662R** |

**Chi phí cụ thể** — 66 event DB và 52 event DT bị loại oan, trong đó có những event
`rule_score ≥ 0.75` **và thắng**:
- DB 2025-11-12 rule 0.9456 → **netR +1.453**
- DT 2026-01-29 rule 0.8639 → **netR +1.486**

**Nguyên nhân gốc:** `min_r2` mới là yếu tố quyết định, không phải slope. Trên event thật:
median |slope|/ATR ≈ **0.09**, median R² ≈ **0.44**. Đặt `min_r2=0.3` là ngưỡng nằm ở
**~35th percentile** ⇒ đương nhiên cắt 65% event bất kể chất lượng.

**Quyết định:** gate ở dạng opt-in, **default TẮT**. Phải ghi rõ trong báo cáo rằng nó
**không có bằng chứng cải thiện**, kèm bảng trên. Nếu sau này ai bật, họ phải đọc được lý do
phản đối ngay tại chỗ.

---

## 3. §4.2 OOS ≥ 100 — FAIL sau khi áp gate ngữ nghĩa

Bản yêu cầu §4.2 yêu cầu OOS ≥ 100 event, **đo sau khi** đã lọc ngữ nghĩa.

| Kịch bản | DB OOS | DT OOS | Đạt? |
|---|---|---|---|
| Baseline (chưa gate gì) | 151 | **98** | DT **đã fail sẵn** |
| Sau stack đầy đủ (len≤40 + rule≥0.6 + trend lb30/r2 0.3) | 43 | 38 | ✗ cả hai |
| Sau stack nới (len≤30 + trend lb20/r2 0.2) | 66 | 54 | ✗ per-pattern |
| Tốt nhất đạt được (combined, lb20/ms0.05/r2 0.2) | 120 combined | | ✓ chỉ khi gộp |

**Phát hiện phụ quan trọng:** `double_top` **đã fail** ngưỡng OOS ≥ 100 ngay ở baseline (98).
Nghĩa là ngưỡng này trong bản yêu cầu không nhất quán với chính dữ liệu hiện có — không gate
nào gây ra chuyện đó.

**Về câu hỏi per-pattern vs combined:** bản yêu cầu không nói rõ. **Không có setting nào đạt
100 OOS cho từng pattern riêng lẻ.** Cách duy nhất đạt là đọc combined (DB+DT), mà cách đó
che mất việc DT yếu hơn hẳn DB.

**Quyết định:** báo cáo **FAIL** trung thực theo đúng luật của bản yêu cầu. KHÔNG nới tham số
để đủ 100. Ghi rõ đây là gate **không đạt**, và nêu việc DT vốn đã dưới ngưỡng.

---

## 4. §2.3 AND-gate — đề xuất ĐÚNG, giữ lại

Khác với §2.1/§2.2, phần này **đúng cả về chẩn đoán lẫn cơ chế**:
- `hard_gate.py` thật sự **không có** ngưỡng nào cho `model_prob`/`rule_score` (đã verify).
- Cơ chế trung bình `(rule_score + model_prob)/2` thật sự **che giấu trục yếu** — đúng như §1.3 mô tả.
- Cách sửa bằng **AND độc lập từng trục** (không dùng `combined_score`) là đúng.

**Nhưng số liệu ngưỡng thì phải đo lại, không copy:**
- `rule_score` p5 đã là **0.603 (DB) / 0.616 (DT)** ⇒ ngưỡng 0.6 gần như **không cắt gì**
  (chỉ 16/354 và 10/310). Đây là tin tốt: floor 0.6 an toàn.
- `model_prob` phân bố rất thấp: DB p10 = 0.0694, median 0.2163; DT min 0.0333, median 0.2565.
  Ngưỡng 0.55 mà bản yêu cầu đề xuất (§6 tự cảnh báo là copy từ tài liệu cũ) sẽ **xoá phần lớn
  event** — phải đo và chốt lại, không dùng nguyên số cũ.

**Quyết định:** giữ §2.3, default TẮT (theo "đo trước bật sau"), ngưỡng để config-driven và
kèm bảng phân bố thực đo.

---

## 5. Việc phải sửa trong cách hiểu bản yêu cầu

| Mục | Bản yêu cầu nói | Thực tế đo được |
|---|---|---|
| §1.2 | range 55+ nến được nhận là pattern | **SAI** — max 29 nến (DB) / 26 (DT) |
| §2.2 | gate này loại "range dài giả dạng pattern" | **Không loại gì** — 0 event |
| §2.1 | gate trend sẽ cải thiện chất lượng tín hiệu | **SAI** — cắt 2/3 event, expectancy không đổi |
| §4.2 | OOS ≥ 100 sau gate | **FAIL** — và DT đã < 100 từ baseline |
| §1.4 | regime_wiring đọc `volatility_ratio` | **SAI** — field này không tồn tại trong repo |
| §1.3 | hard_gate không có ngưỡng prob/rule | **ĐÚNG** — xác nhận |

---

## 4b. Hai BLOCKER từ t1 (code_verifier) — phải sửa cách hiểu trước khi code

`code_verifier` trả verdict `needs_revision` với 2 blocker + 2 high finding. Hai blocker đều
đúng và làm thay đổi phạm vi công việc:

### BLOCKER 1 — `volatility_ratio` không tồn tại
§1.4 và §2.4 step 1 giả định regime pipeline output field này. **Không có match nào trong toàn bộ
file `.py`**; cả 6 hit repo-wide đều là văn xuôi markdown. Bề mặt thật của emitter là
`hmm_state`, `hmm_prob_<state_name>`, `hmm_confidence` (`feature_emitter.py:172-175`).
Đề xuất `stop_buffer_atr(volatility_ratio)` trong spec cũ **chưa bao giờ được build**
(`resolution.md:29` ghi "KHÔNG LÀM").
→ **t5 tuyệt đối không được code dựa trên `volatility_ratio`.**

### BLOCKER 2 — HMM regime đang INERT ở live, không phải "hạ tầng đã kiểm chứng"
Điều này **vô hiệu hoá lý do chọn Phương án C ở §4.5**. Bằng chứng:
- `configs/plugins/hmm_regime.yaml:20` `plugin.enabled: false`; `:36` `regime_filter.enabled: false`
- Thư mục config symbol **của live** là `research/configs/symbols/`, và `XAUUSD.yaml` ở đó
  **không có** key regime/hmm nào; `:74` `model_path` rỗng ⇒ `create_symbol_engine` raise
- `signal_polling_engine_v2.py` **không hề** tham chiếu regime/`MultiPatternEngine`; nó dùng
  closure LSW legacy (`:215` → `:676`)
- Toàn bộ đường regime chỉ nằm trong `MultiPatternEngine` — mà **live không bao giờ khởi tạo**

→ §2.4 **không phải** việc mở rộng 2–3 ngày. Muốn HMM trend chạy thật ở live thì cần: bật plugin
switch, thêm regime block vào **đúng** thư mục config, và migrate live path sang `MultiPatternEngine`.

### HIGH — `rule_score` có HAI thang đo
LSW dùng 0..100 (`_RULE_SCORE_SCALE`), DB/DT dùng 0..1. Một floor thô `0.6` sẽ
**xoá sạch mọi event liquidity_sweep** (giá trị thật ~32–51).
→ Floor **bắt buộc** áp lên giá trị đã normalize. Đây là blocker cho t3.

### HIGH — B1 `min_model_prob` đã tồn tại
Nó đã có sẵn ở dạng opt-in với map rỗng (`signal_engine_v2.py:75`, gate `:1145-1152` ghi
`"low_probability"`), **không phải** `PROB_THRESHOLDS` hardcode như §2.3 mô tả.
→ Nên **tái sử dụng `"low_probability"`**, không tạo `"low_model_prob"` mới.

### Điểm cộng: t5 vẫn khả thi
Có thể emit state hướng trend **mà không phá contract đã freeze**: `get_feature_names` sinh tên
động từ `state_names` (`base.py:116-128`), `_resolve_config` chỉ kiểm tra số lượng/tính duy nhất
(`gaussian_hmm.py:293-296`). Nhưng volatility và trend **không thể cùng nằm trong một**
`state_name` ⇒ dùng **instance HMM thứ hai**, giữ `base.py` nguyên vẹn.

---

## 6. Quyết định cuối cùng

1. **§2.2**: cài với default 30, ghi rõ là no-op đã đo. Không quảng cáo là cải thiện.
2. **§2.3**: cài đầy đủ (detector + live), default TẮT, ngưỡng config-driven + bảng phân bố.
   Đây là phần **duy nhất của bản yêu cầu được đo là có cơ sở đúng**.
3. **§2.1**: cài dạng opt-in default TẮT, và **ghi rõ trong docstring rằng gate không có
   selection power** kèm số đo. Giữ lại vì cơ chế causal của nó đúng và có thể có ích khi
   kết hợp HMM — nhưng không được bật dựa trên bằng chứng hiện có.
4. **§2.4 (HMM trend)**: vẫn làm, vì đó là cách duy nhất còn lại để có một trend signal
   **học từ dữ liệu** thay vì ngưỡng cứng. Đây là hướng có triển vọng nhất, và cần đo độc lập
   trước khi so với §2.1.
5. **Không nới tham số để đạt OOS 100.** Báo cáo FAIL.

**Nguyên tắc giữ nguyên:** bản yêu cầu §3 cấm áp lại `stop_mode="neckline"` và cap 4 ATR —
lệnh cấm này vẫn hiệu lực và đã được nhúng vào mọi task.
