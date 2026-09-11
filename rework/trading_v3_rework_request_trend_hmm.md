# Rework Request — Trend-Context Gate, Pattern Length Gate, Live Probability Gate & HMM Regime Integration

**Ngày viết:** 2026-09-11
**Repo:** `vankienemk/trading_v3`
**Phạm vi:** `research/patterns/double_bottom/detector.py` (base class, ảnh hưởng cả `double_top`), `live/engine/signal_engine_v2.py`, `live/engine/hard_gate.py`, `live/engine/regime_wiring.py`, `live/engine/feature_emitter.py`, `live/db/configs/v2_frozen.yaml`
**Người yêu cầu:** Kiên
**Nguồn:** phiên review 3 chart export (`double_top SELL p=0.033`, `liquidity_sweep BUY`, `double_bottom BUY p=0.069/p=0.069`), đối chiếu với `trading_v3_pattern_rework_spec.md` và `trading_v3_bug_summary.md` đã có sẵn trong repo.

---

## 0. Tóm tắt điều hành

Kết luận của phiên review: **bộ lọc hình học/nhân quả (causal) của detector đang hoạt động đúng thiết kế** — right-bar rule, NMS theo structure window, `drop_opposite_overlap`, staleness window, depth gate (`min_depth_atr`) — tất cả đã đo và pin bằng test. Vấn đề không nằm ở "detector bị bug", mà ở chỗ **detector đang thiếu 2 gate ngữ nghĩa (semantic)** khiến nó bắt "đúng hình học trên giấy, sai hình học trên thị trường":

1. Không có gate về **bối cảnh xu hướng trước pattern** — double_top/double_bottom về định nghĩa cổ điển là *reversal pattern*, phải có downtrend/uptrend đáng kể trước đó. Hai tín hiệu trong ảnh đều nằm giữa một sideways range dài, không phải sau một trend.
2. Không có **giới hạn độ dài pattern tối đa** — range 55+ nến vẫn được chấp nhận như một double-top/double-bottom hợp lệ.

Thêm vào đó, **live engine không có ngưỡng chặn tín hiệu xác suất thấp**: cả hai signal trong ảnh có `rule_score` rất cao (0.93 / 0.77 — vì công thức score chỉ đo depth/symmetry/offset, không đo trend context) nhưng `model_prob` rất thấp (0.180 / 0.272 — model đã "biết" chúng yếu). Vì `combined_score = (rule_score + model_prob) / 2`, một trục mạnh che giấu một trục yếu, kết quả (`0.55` / `0.52`) vẫn đủ để tạo `PendingSignal` vì **`hard_gate.py` không có threshold nào cho `model_prob` hay `combined_score`**.

> **[REWORK-VERIFIED — 2026-09-11, code_verifier t1]**
> Đoạn này **SAI về mặt kỹ thuật**. `volatility_ratio` **KHÔNG tồn tại** trong bất kỳ file `.py`
> nào của repo — cả 6 lần xuất hiện đều là văn xuôi markdown mô tả một đề xuất chưa từng được
> build (`pattern_rework_resolution.md:29` ghi rõ đề xuất §2.3.5 "KHÔNG LÀM").
> Bề mặt thật của emitter là: `hmm_state`, `hmm_prob_<state_name>`, `hmm_confidence`
> (`live/engine/feature_emitter.py:172-175`), cùng `hmm_known_at` / `hmm_config_hash` (`:178-179`).
> Ngoài ra, HMM regime hiện **KHÔNG chạy ở live** (xem annotation ở §1.4) — nên đây không phải
> "hạ tầng đã kiểm chứng" như câu văn ngụ ý.

Cuối cùng, hệ thống **đã có hạ tầng HMM** (dùng để tính `volatility_ratio` cho `stop_buffer_atr` dynamic — xem `live/engine/regime_wiring.py` đọc từ `live/engine/feature_emitter.py`), nhưng regime hiện tại chỉ phân loại **volatility** (trending/ranging theo `atr_fast/atr_slow`), chưa được dùng để phân loại **trend direction** — đúng thứ mà gate #1 ở trên cần. Đây là cơ hội để không phải viết một heuristic slope/ATR mới (dễ overfit vào đúng 2 ảnh mẫu) mà tận dụng lại pipeline regime đã có, mở rộng nó thay vì tạo song song một cơ chế trend-detection khác.

**Không đụng vào (đã đo, có hại):** `stop_mode="neckline"` (winrate đo được giảm 48%→23%) và target cap 4 ATR + `min_rr` (xoá 96% event) — theo `pattern_rework_resolution.md`. Hai đặc tính này (stop rộng ~4.5 ATR) là đặc trưng cần thiết của pattern family này, không phải bug.

---

## 1. Vấn đề đã xác nhận

### 1.1 — Thiếu trend-context gate trước pattern

- **Triệu chứng.** `double_bottom BUY` bên trái ảnh: sideways range ~55+ nến giật nhiễu, hai "đáy" cách nhau rất xa và chênh giá rõ, đỉnh giữa (neckline) nằm sát mép phải box — về cổ điển đây là accumulation range, không phải đáy đôi sau downtrend. `double_top SELL` bên phải: tương tự, range đi ngang với một đỉnh spike lẻ; lệnh SELL confirm đúng lúc giá đang ở đáy range và sắp bật lên — tức bán đuổi tại vùng hỗ trợ.
- **Nguyên nhân gốc.** Detector chỉ gate theo hình học nội bộ pattern (`max_equal_atr`, `min_separation_bars`, `min_depth_atr`) — không có điều kiện nào yêu cầu context TRƯỚC `extreme1_bar`.
- **Cần team verify dòng cụ thể** trong `get_default_config` / `_detect_double` (base class dùng chung cho double_top/double_bottom) — phiên review này không có quyền đọc trực tiếp source nên không thể trích line number chính xác cho phần mới.

### 1.2 — Không giới hạn độ dài pattern tối đa

- **Triệu chứng.** `pattern_length` đã được tính và emit như một feature, nhưng không được dùng làm điều kiện loại (gate). Pattern cổ điển quá dài (range 55+ nến) thường đã thoái hóa thành sideways range, không còn ý nghĩa reversal pattern.
- **Cần team verify** vị trí feature `pattern_length` được tính, để chèn gate ngay cạnh.

### 1.3 — Live engine không gate `model_prob`/`rule_score`, và average che giấu trục yếu

- **Triệu chứng.** `combined_score = (rule_score + model_prob) / 2` cho phép một trục cực mạnh (hình học) bù cho một trục cực yếu (model) — `0.93` và `0.180` trung bình ra `0.55`, đủ "đậu" một threshold `combined ≥ 0.5` nếu có, và **hiện tại còn không có threshold nào cả** (`hard_gate.py` — grep không ra ngưỡng nào cho `model_prob`/`combined_score`).
- **Rủi ro nếu chỉ sửa bằng threshold trên `combined_score`:** không giải quyết được vấn đề, vì cơ chế trung bình vẫn che giấu trục yếu ở mọi mức threshold hợp lý. Cần AND-gate độc lập từng trục, không phải threshold trên giá trị trung bình.

### 1.4 — HMM regime filter đã tồn tại nhưng chưa dùng cho trend detection

- **Hiện trạng.** `live/engine/feature_emitter.py` + `live/engine/regime_wiring.py` đã có pipeline đọc regime hiện tại từ HMM emitter, nhưng chỉ output `volatility_ratio` (dùng cho `stop_buffer_atr` dynamic ở đề xuất 2.3.5 trong `pattern_rework_spec.md`). Không có state nào phân loại **trend direction** (uptrend/downtrend/range).

> **[REWORK-VERIFIED — 2026-09-11, code_verifier t1]**
> **SAI.** `volatility_ratio` **không tồn tại** — 0 lần xuất hiện trong bất kỳ file `.py` nào.
> Bề mặt thật của emitter: `hmm_state`, `hmm_prob_<state_name>`, `hmm_confidence`
> (`feature_emitter.py:172-175`) + lineage `hmm_known_at` / `hmm_config_hash` (`:178-179`).
> **Thứ hai, quan trọng hơn:** HMM regime **đang KHÔNG chạy ở live**, nên mô tả "đã có pipeline"
> là gây nhầm. Bằng chứng: `configs/plugins/hmm_regime.yaml:20` `plugin.enabled: false`;
> `:36` `regime_filter.enabled: false`; thư mục config symbol của live là
> `research/configs/symbols/` và `XAUUSD.yaml` ở đó **không có** key regime/hmm nào;
> `live/engine/signal_polling_engine_v2.py` **không hề** tham chiếu regime hay
> `MultiPatternEngine` — nó gọi `create_symbol_engine` (`:215`) → closure LSW legacy (`:676`).
> Toàn bộ đường regime chỉ tồn tại trong `MultiPatternEngine`, mà live **không bao giờ** khởi tạo.
> **Hệ quả:** §2.4 **không phải** việc mở rộng 2–3 ngày. Cần: bật plugin switch + thêm regime
> block vào **đúng** thư mục config + migrate live path sang `MultiPatternEngine`.
- **Cơ hội.** Thay vì viết một heuristic trend-context riêng (rủi ro overfit — xem 2.1 bên dưới), nên mở rộng HMM emitter đã có sẵn để output thêm trend-regime state, và dùng chính state đó làm gate 1.1.

---

## 2. Đề xuất rework

### 2.1 — Trend-context gate: 2 phương án, khuyến nghị phương án B

**Phương án A — heuristic slope/ATR đơn giản (rủi ro overfit).**

```python
def has_trend_context(closes, atr_k, extreme1_bar, lookback, min_trend_atr=3.0):
    """So 2 điểm đầu-cuối lookback — dễ bị 1 nến outlier kéo lệch."""
    ref = extreme1_bar - lookback
    if ref < 0:
        return False
    move = closes[extreme1_bar] - closes[ref]
    return abs(move) >= min_trend_atr * atr_k
```

Rủi ro: `lookback` và hệ số `min_trend_atr` là 2 tham số tự do dễ bị calibrate (vô tình hoặc cố ý) khớp đúng 2 ảnh mẫu này, mà không tổng quát hoá tốt trên toàn bộ 204k nến lịch sử. So-2-điểm cũng có thể vô tình thỏa mãn nếu 2 mốc lookback rơi trúng 2 đỉnh/đáy của chính cái range đang muốn loại.

**Phương án B — dùng linear regression slope + R² trên N bars trước `extreme1_bar` (ổn định hơn A, vẫn không cần đụng hạ tầng khác):**

```python
def has_trend_context(closes, atr_k, extreme1_bar, lookback, min_slope_atr=0.05, min_r2=0.3):
    window = closes[extreme1_bar - lookback : extreme1_bar]
    if len(window) < lookback:
        return False
    x = np.arange(len(window))
    slope, intercept = np.polyfit(x, window, 1)
    residuals = window - (slope * x + intercept)
    r2 = 1 - (residuals**2).sum() / max(((window - window.mean())**2).sum(), 1e-12)
    return abs(slope) / atr_k >= min_slope_atr and r2 >= min_r2
```

**Phương án C (khuyến nghị) — dùng HMM regime state đã có sẵn (xem 2.4), thay vì heuristic mới.** Vì hạ tầng HMM đã tồn tại cho volatility regime, mở rộng nó để phân loại trend state là cách tận dụng lại pipeline đã kiểm chứng (không phải một cơ chế song song mới cần calibrate từ đầu), đồng thời tránh rủi ro overfit của phương án A/B vì HMM học regime từ toàn bộ lịch sử thay vì một ngưỡng cứng cục bộ quanh `extreme1_bar`.

### 2.2 — `max_pattern_length_bars`

```python
pattern_length = extreme2_bar - extreme1_bar
if pattern_length > max_pattern_length_bars:  # đề xuất 40-60 bars cho M15
    continue  # hoặc discard_reason="pattern_too_long"
```

Gate này độc lập, rẻ, cùng mục tiêu với 2.1 (loại range dài giả dạng pattern), nên có thể merge cùng PR.

### 2.3 — AND-gate `rule_score`/`model_prob` độc lập, cả ở detector lẫn live engine

**Ở detector** (fail-closed, log Event Lake):

```python
if rule_score < min_rule_score:  # đề xuất 0.6
    event.attributes["discard_reason"] = "low_rule_score"
    continue
```

**Ở `hard_gate.py`** (trước khi tạo `PendingSignal`) — AND riêng từng trục, KHÔNG dùng `combined_score` trung bình để quyết định drop:

```python
PROB_THRESHOLDS = {
    "double_bottom": 0.55, "double_top": 0.55,
    "liquidity_sweep": 0.50, "falling_wedge": 0.50,
    "rising_wedge": 0.50, "head_shoulders": 0.55,
    "inverse_head_shoulders": 0.55,
}
RULE_SCORE_FLOOR = 0.6

threshold = PROB_THRESHOLDS.get(pattern_name, 0.55)
if model_prob < threshold or rule_score < RULE_SCORE_FLOOR:
    reason = "low_model_prob" if model_prob < threshold else "low_rule_score"
    candidate.attributes["discard_reason"] = reason
    return None  # không tạo PendingSignal
```

`combined_score` vẫn có thể giữ lại làm feature/ranking cho RiskGuard, exposure cap — nhưng KHÔNG được dùng làm gate duy nhất quyết định có tạo lệnh hay không.

### 2.4 — Mở rộng HMM regime emitter để phân loại trend (kế hoạch tích hợp)

1. **Xác nhận API hiện tại** của `feature_emitter.py` / `regime_wiring.py`: input hiện dùng (returns? ATR fast/slow?), output hiện có (`volatility_ratio`).

> **[REWORK-VERIFIED — 2026-09-11, code_verifier t1]**
> Bước 1 này **đã được thực hiện**, kết quả khác với giả định trong câu hỏi:
> - Output hiện có **KHÔNG** phải `volatility_ratio` (field này không tồn tại). Bề mặt thật:
>   `hmm_state`, `hmm_prob_<state_name>`, `hmm_confidence` (`feature_emitter.py:172-175`),
>   cộng `hmm_known_at` / `hmm_config_hash` để truy vết (`:178-179`).
> - Tên feature sinh **động** từ `state_names` (`research/regime/base.py:116-128`), và
>   `_resolve_config` chỉ kiểm tra số lượng + tính duy nhất (`gaussian_hmm.py:293-296`)
>   ⇒ **có thể thêm vocabulary state mới mà không phá contract đã freeze**.
> - Input feature là **vocabulary đóng** (`CANONICAL_INPUT_FEATURES`, `base.py:55-59`, raise ở
>   `:130-134`) ⇒ slope **phải** được cấp như một cột trong `df`.
> - Volatility và trend **không thể** cùng nằm trong một `state_name`
>   ⇒ dùng **instance HMM thứ hai**, giữ `base.py` / `gaussian_hmm.py` nguyên vẹn.
> - Bước 4 (wire vào gate) **chưa thể làm ở live** cho tới khi xử lý xong blocker ở §1.4
>   (plugin đang tắt + sai thư mục config + live chưa dùng `MultiPatternEngine`).
2. **Mở rộng HMM** (multivariate, ví dụ 3-state: uptrend / downtrend / range) trên feature vector gồm return trung bình N bars + slope, huấn luyện lại trên cùng tập dữ liệu XAUUSD M15 đã dùng cho volatility regime, để tránh lệch phân phối giữa 2 mô hình.
3. **Expose `trend_regime_state`** vào `PatternEvent.attributes` tại thời điểm `extreme1_bar` (không lookahead — dùng state đã biết TẠI bar đó, giữ đúng `no_lookahead` contract).
4. **Wire vào gate 2.1**: double_bottom chỉ pass nếu `trend_regime_state == "downtrend"` tại `extreme1_bar` (và ngược lại cho double_top).
5. **Feature schema versioning**: nếu `trend_regime_state` là feature mới ảnh hưởng đến train_walkforward, cần bump `feature_schema_version` (`double-v1.0` → `double-v1.1`) theo đúng lưu ý §5 trong `pattern_rework_spec.md`, và re-run `config_hash` gate.

---

## 3. Không đụng vào (đã đo, có hại)

| Đề xuất | Kết quả đo | Quyết định |
|---|---|---|
| `stop_mode="neckline"` | Winrate đo được giảm 48% → 23% | KHÔNG áp dụng |
| Target cap 4 ATR + `min_rr` | Xoá 96% event | KHÔNG áp dụng |

Nguồn: `pattern_rework_resolution.md`. Stop rộng (~4.5 ATR) là đặc tính cần thiết của pattern family này trên XAUUSD M15, không phải bug.

---

## 4. Backtest / đo lường bắt buộc trước khi bật mặc định

Theo đúng protocol "đo trước, bật sau" đã áp dụng cho các gate khác trong dự án:

1. Chạy `docs/rework_measure.py` (hoặc tương đương) trên **toàn bộ lịch sử XAUUSD M15 (204k nến)** cho từng gate mới (2.1, 2.2, 2.3) — đo riêng lẻ và đo cộng dồn.
2. Yêu cầu sample-size OOS ≥ 100 events sau khi áp gate — **đo sau khi** cả 2 gate ngữ nghĩa (2.1, 2.2) đã lọc, không đo trước, để tránh con số OOS bị nhiễu bởi chính các range-pattern giả đang muốn loại.
3. Re-run `pytest -q -m no_lookahead` (168 tests hiện có) + golden LSW (15/15 bit-identical) sau mỗi thay đổi.
4. Nếu 2.4 (HMM trend) thêm feature mới → bump `feature_schema_version`, re-run `config_hash` gate, re-train `models/double_bottom_xauusd_m15_v1/model.pkl`.
5. Đo riêng A1 (heuristic) vs A2 (regression) vs Phương án C (HMM) nếu muốn so sánh trước khi chốt — không bắt buộc phải chọn HMM ngay nếu effort không cho phép trong sprint này; có thể ship 2.1 bằng Phương án B trước (rẻ hơn), rồi thay bằng Phương án C ở sprint sau khi HMM trend-regime đã được huấn luyện và đo riêng.

---

## 5. Trình tự implementation đề xuất

| Bước | Nội dung | Effort ước tính | Ghi chú |
|---|---|---|---|
| 5.1 | Gate 2.3 (AND rule_score/model_prob) ở `hard_gate.py` + detector | 0.5–1d | Không đụng calibration đã pin, rẻ nhất, nên làm trước tiên |
| 5.2 | Gate 2.2 (`max_pattern_length_bars`) | 0.5d | Độc lập, thêm test `test_pattern_too_long` |
| 5.3 | Gate 2.1 Phương án B (regression slope) | 1d | Ship trước để có gate hoạt động ngay trong sprint này |
| 5.4 | Đo A/B toàn bộ 204k nến cho 5.1–5.3 cộng dồn | 0.5–1d | Bắt buộc trước khi bật mặc định live |
| 5.5 | Thiết kế mở rộng HMM trend-regime (2.4) | 2–3d | Cần xác nhận API `feature_emitter.py` hiện tại trước, effort phụ thuộc mức độ refactor |
| 5.6 | Huấn luyện + đo riêng HMM trend-regime, so với 5.3 | 1–2d | Nếu tốt hơn đáng kể, thay 5.3 bằng 2.4 ở sprint sau; nếu không, giữ 5.3 |
| 5.7 | Regression đầy đủ (no_lookahead, golden, config_hash nếu đổi schema) | 0.5d | Bắt buộc trước merge bất kỳ bước nào ở trên |

**Tổng ước tính:** ~6–9 ngày-sprint tuỳ có làm 5.5–5.6 (HMM) trong sprint này hay để sprint sau.

---

## 6. Việc cần team xác nhận trước khi bắt đầu

Phiên review này **không có quyền đọc trực tiếp source code** của repo (chỉ đọc được 2 file markdown đã publish: `trading_v3_pattern_rework_spec.md`, `trading_v3_bug_summary.md`), nên các mục sau cần team tự verify bằng cách đọc code trực tiếp trước khi implement, để tránh sai lệch dòng/API so với những gì tài liệu này giả định:

- Vị trí chính xác (`file:dòng`) để chèn gate 2.1 và 2.2 trong base class của `double_bottom/double_top` detector.
- API hiện tại của `feature_emitter.py` / `regime_wiring.py` — input/output hiện có, để biết mức độ refactor cần cho 2.4.
- Ngưỡng `PROB_THRESHOLDS` đề xuất ở 2.3 — đây là copy lại từ đề xuất B1 cũ trong `bug_summary.md`, cần xác nhận vẫn còn đúng ngữ cảnh hiện tại của repo (B1 có thể đã có thay đổi từ lúc đó).
- Định dạng chính xác của `discard_reason` trong Event Lake schema hiện tại, để 3 lý do mới (`low_rule_score`, `pattern_too_long`, `low_trend_context`) khớp convention sẵn có.

---

## 7. Tài liệu tham chiếu

- `trading_v3_pattern_rework_spec.md` (repo, đã publish) — NMS, SL/TP rework
- `trading_v3_bug_summary.md` (repo, đã publish) — B1 (min_model_prob), B2 (combined_score), B3 (swing strict), B5 (legacy wiring)
- `pattern_rework_resolution.md` (repo) — kết quả đo `stop_mode="neckline"` và target cap 4 ATR (đã xác nhận có hại)
- 3 chart export phân tích trong phiên này: `double_top SELL p=0.033|s=0.79` (XAUUSD, ~7-8 Jul), `liquidity_sweep BUY` chồng lấn double_top/double_bottom (28 Aug), `double_bottom BUY p=0.069|s=0.9x` (27 Aug)
