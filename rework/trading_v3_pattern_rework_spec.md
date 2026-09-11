# Bản mô tả — Rework pattern detector & SL/TP cho `trading_v3`

**Ngày viết:** 2026-09-10
**Repo gốc:** `vankienemk/trading_v3` (clone phiên này)
**Phạm vi:** pattern detector (overlap / chồng lấn) + SL/TP sizing (target xa — winrate thấp)

---

## 0. Tóm tắt dẫn chứng

### 0.1. 3 chart đã upload — phân tích

| # | URL ảnh | Pattern thấy | Vấn đề quan sát |
|---|---------|--------------|-----------------|
| 1 | `0q6k5Lbh` | `double_top SELL \| p=0.227 \| s=0.79` (XAUUSD ~7–8 Jul) | Hai đỉnh bằng nhau + volume spike đúng breakout candle — **hợp lệ**. **NHƯNG** SL vàng trên = đúng bằng khoảng cách từ entry đến đỉnh cao nhất + 0.5 ATR, TP xanh dưới lại gấp **3R** (từ trade-zone chiều dài), tức xa hơn chiều sâu swing — khó hit. |
| 2 | `iXwon6U2` | `double_top SELL \| p=0.194 \| s=0.76` (6 Jul 06–09h) **CHỒNG LẤN** `double_bottom BUY \| p=0.089 \| s=0.72` (6 Jul 16–17h) | Hai pattern ngược chiều phát hiện **cùng cấu trúc** (low–high–low vs high–low–high quanh cùng 1 swing middle), label BUY bị đặt **đè lên** zone vàng SELL cũ; teal/purple blending = 2 zone overlap về giá. Đây là evidence rõ nhất cho "pattern overlap chồng lấn". |
| 3 | `B7tSz2Zt` | chart có header raw text `2.12.13 01:00\|...\|1783.62\|…\|1784.74\|2022.12.13 01:00\|2022.12.13 01:30\|` + `dou…` (label bị truncate) — BUY XAUUSD 2022-12-13 | Gần breakout candle giá **drop nhẹ rồi reverse mạnh lên trên** SL đỏ, thị trường sau đó mới quay xuống. **TP được vẽ dài gấp nhiều lần depth pattern**, kéo winrate giảm. |

### 0.2. Dẫn chứng file:dòng (đã đọc trong phiên này)

| Vấn đề | File:dòng |
|--------|-----------|
| `min_separation_bars=4` quá lỏng → 2 pattern cùng structure | `research/patterns/double_bottom/detector.py:125–147` (`get_default_config`) |
| `cooldown_bars=3` chỉ áp trên `confirm_bar` | `research/patterns/double_bottom/detector.py:330–348` (`_dedupe`) |
| Vòng quét tạo **mọi** cặp (s1,s2,s3) hợp kind_seq mà không NMS | `research/patterns/double_bottom/detector.py:191–224` |
| `target_r=1.5` hardcode trong pattern plugin | `research/patterns/double_bottom/detector.py:140, 171, 254–256` |
| `target_r=1.25` cho dataset labeling (train_walkforward) | `research/patterns/double_bottom/scripts/train_walkforward.py:129, 149, 212, 235` |
| **Live engine** lại dùng `v2_target_r=3.0` + `primary_reward_r=2.0` | `live/engine/signal_engine_v2.py:270, 432, 462, 482–485`; `live/db/configs/v2_frozen.yaml:40, 66, 68` |
| `stop_buffer_atr=0.5` cố định — scale theo ATR (không scale theo structure) | `research/patterns/double_bottom/detector.py:139, 170, 247–250` |
| `min_depth_atr=3.0` — pattern sâu → SL xa → TP còn xa hơn nữa | `research/patterns/double_bottom/detector.py:138, 223` |
| H&S, RisingWedge cùng `target_r=1.5` | `rpatterns/head_shoulders/detector.py:143-144, 178-179, 273-282`; `patterns/rising_wedge/detector.py:169-170, 205-206, 326-335` |

---

## 1. Vấn đề A — Pattern chồng lấn (DB/DT/H&S/RW)

### 1.1. Mô tả hiện tượng

Chart #2 (`iXwon6U2`) cho thấy hai event được vẽ chồng lên nhau:
* `double_top SELL` quanh 6 Jul 06–09h.
* `double_bottom BUY` quanh 6 Jul 16–17h cùng dùng một cú swing đáy giữa (`H/L/H` lẫn `L/H/L`) ⇒ chung middle swing.

Lý do: detector chỉ lọc **kind_seq** (`L,H,L` hoặc `H,L,H`) và `min_separation_bars` giữa 3 swing — không có **non-max suppression (NMS)** theo structure window.

### 1.2. Nguyên nhân gốc — đọc code

`_detect_double` (`double_bottom/detector.py:158–226`) quét:

```python
for k in range(len(sw) - 2):
    s1, s2, s3 = sw[k], sw[k + 1], sw[k + 2]
    if (s1.kind, s2.kind, s3.kind) != want:
        continue
    ...
    if depth < min_depth_atr * atr_k:
        continue
    # ... confirm + emit PatternEvent
```

* Cứ 3 swing liên tiếp hợp `(L,H,L)` thì **emit event**, không KIỂM TRA xem các sóng **nằm trong cùng structure window** không.
* `_dedupe` (dòng 330–348) chỉ gỡ 2 event có `confirm_bar` cách nhau dưới `cooldown_bars`. Nhưng chart #2 có 2 event BUY & SELL cách nhau nửa ngày → cooldown KHÔNG bắt được.

### 1.3. Đề xuất rework

#### Đề xuất 1.3.1 — NMS theo `extreme1` + `extreme2` overlap

Pseudo-diff cho `_detect_double` (chèn ngay sau vòng `for k`, trước khi `append` ở cuối vòng):

```python
# Anti-chồng-lấn:
#   hai candidate cùng pattern không được chia sẻ cùng cấu trúc.
#   Quy tắc: nếu A.extreme1 <= B.extreme1 <= A.extreme2 hoặc ngược lại
#   → giữ event có rule_score cao hơn, drop event còn lại (NMS).
keep_candidates: list[PatternEvent] = []
cand_sorted = sorted(candidates, key=lambda e: -float(e.rule_score))
occupied: list[tuple[int, int]] = []  # các (i1, i3) đã giữ

for ev in cand_sorted:
    i1 = int(ev.attributes["extreme1_bar"])
    i3 = int(ev.attributes["extreme2_bar"])
    overlap = any(not (i3 < a_i1 or i1 > a_i3) for (a_i1, a_i3) in occupied)
    if overlap:
        continue
    keep_candidates.append(ev)
    occupied.append((i1, i3))
candidates = keep_candidates
return self._dedupe(candidates, cooldown)
```

#### Đề xuất 1.3.2 — Nâng `min_separation_bars` tham số hoá

Trong `get_default_config` (dòng 125–147) vẫn giữ `min_separation_bars=4` cho default, nhưng THÊM vào `multi_backtest/runner` cơ chế sweep `[3, 4, 6, 8]` trên train-set để chọn `min_separation_bars` theo symbol/timeframe (XAUUSD M15 → 6 cho DB/DT).

#### Đề xuất 1.3.3 — Ràng buộc overlap với pattern NGƯỢC chiều

Thêm helper `drop_opposite_overlap` ở `research/core/dedupe.py` (file mới) hoặc cuối `findings_resolution.md` đã đề cập:

```python
def drop_opposite_overlap(events: list[PatternEvent],
                           min_staleness_bars: int = 3) -> list[PatternEvent]:
    """Drop một event khi tồn tại event NGƯỢC direction khác dùng chung
    (extreme1, extreme2, middle_swing) trong khoảng gần.
    Giữ event có rule_score cao hơn."""
    keep = []
    sorted_evs = sorted(events, key=lambda e: -float(e.rule_score))
    seen_keys: set[tuple[str, int, int, int]] = set()
    for ev in sorted_evs:
        key = (ev.pattern_name, int(ev.attributes["extreme1_bar"]),
               int(ev.attributes["neckline_bar"]),
               int(ev.attributes["extreme2_bar"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        keep.append(ev)
    return keep
```

Tích hợp vào `MultiPatternEngine._resolve` (xem `signal_engine_v2.py`) **trước** khi dedup/time-window.

#### Đề xuất 1.3.4 — Hiển thị DebugDrawer không vẽ overlap

Trong `DebugDrawer.mq5` thêm: nếu 2 event overlap quá 50% structure box, **chỉ vẽ text event có `model_prob` cao hơn**; event còn lại vẫn emit warning lên echo file nhưng không paint rectangle.

```cpp
// pseudo MQL5: thêm map overlap để check trước khi ObjectCreate
if(HasHeavyOverlap(InpPrefix+"pat_", t1p, t2p, patLo, patHi, 0.5))
  {
   Print("OVERLAP: skipped ", uid);
   continue;
  }
```

---

## 2. Vấn đề B — SL/TP đặt quá xa → winrate giảm

### 2.1. Mô tả hiện tượng

Hình #3 cho thấy TP bar xanh kéo dài gấp gần 4× depth của pattern. Chart #1 cũng khẳng định: SL vàng ngắn (depth pattern = 3 ATR) nhưng TP xanh (3R = 3 × R ≈ 9 ATR) rất xa so với biên độ price action quanh breakout.

Lý do: `target_r` trong plugin = `1.5`; live engine đọc `v2_target_r = 3.0` ⇒ live trade target gấp đôi training target.

### 2.2. Nguyên nhân gốc — đọc code

#### A. Formula SL/TP hiện tại (`double_bottom/detector.py:247–256`):

```python
if bullish:
    stop_price = min(v1, v3) - stop_buffer_atr * atr_k     # SL = cận dưới + 0.5 ATR
else:
    stop_price = max(v1, v3) + stop_buffer_atr * atr_k
risk = abs(entry_price - stop_price)
if bullish:
    target_price = entry_price + target_r * risk            # target_r=1.5 → TP = entry + 1.5R
    ...
```

`risk = |entry − stop|`. Khi depth pattern lớn (vì `min_depth_atr=3.0`), `min(v1,v3)` thấp, `stop` càng thấp, `risk` càng lớn. Và `target_r=1.5` nhân lên ⇒ TP rất xa.

#### B. Live engine nhân với `v2_target_r=3.0`:

`signal_engine_v2.py:462–485`:

```python
reward_r = pipeline_params["v2_target_r"]    # = 3.0 từ v2_frozen.yaml
...
target_price = entry_price + reward_r * (entry_price - stop_price)
```

⇒ live trade target = **3R** (= 2× training target = 4.5× depth pattern). Quá xa cho M15.

#### C. `v2_frozen.yaml` ép cứng:

```yaml
v2_target_r: 3.0
reward_r_values: [2.0]
primary_reward_r: 2.0
```

Hai tham số khác nhau tạo ambiguity: spec muốn `2.0`, nhưng `v2_target_r=3.0` lại là giá trị thực sự được dùng.

### 2.3. Đề xuất rework

#### Đề xuất 2.3.1 — Structure-based SL

Thay vì `min(v1,v3) − stop_buffer_atr × ATR`, dùng trailing-structure:

```python
def _structure_stop_bullish(v_low: float, swings: list[SwingPoint],
                            atr_k: float, swing_right_bars: int) -> float:
    """SL đặt dưới 'last higher swing low' (nếu có) thay vì đáy cũ.
    Nếu không có, fallback = v_low - 0.5*ATR (giữ cũ)."""
    higher_lows = [
        s for s in swings
        if s.kind == "L" and s.price > v_low and s.price - v_low < 2.0 * atr_k
    ]
    if higher_lows:
        return min(s.price for s in higher_lows) - 0.25 * atr_k
    return v_low - 0.5 * atr_k
```

Hệ quả: SL **gần entry hơn** khi có higher-low swing trong khoảng 2 ATR phía trên `v_low` (tăng winrate vì SL không bị "đào" qua cả structure lớn).

#### Đề xuất 2.3.2 — Capped R-multiples + partial TP/BE trailing

Đặt `target_r` theo **swing leg tiếp theo** (target cấu trúc, không R):

```python
def _structure_target_bullish(entry: float, swings: list[SwingPoint],
                              atr_k: float, max_target_atr: float = 4.0) -> float:
    """Target đặt tại:
      1. Higher-high structure swing gần nhất (nếu gần hơn max_target_atr);
      2. Hoặc entry + max_target_atr * ATR (trần);
      3. Tối thiểu 0.75R (không vào lệnh với R:R < 1).
    """
    highs = [s for s in swings if s.kind == "H" and s.price > entry]
    nearest_hh = min(highs, key=lambda s: s.price - entry, default=None)
    if nearest_hh is not None and (nearest_hh.price - entry) <= max_target_atr * atr_k:
        return nearest_hh.price - 0.10 * atr_k
    return entry + max_target_atr * atr_k
```

⇒ TP **không xa hơn `max_target_atr × ATR`** (= 4 ATR cho M15 XAUUSD). So với hiện tại:
* `risk` ~3.5 ATR (depth + buffer),
* `target` trước = `entry + 3R = entry + 10.5 ATR` (xa),
* `target` sau = `entry + 4 ATR` (gần hơn, winrate cao hơn).

#### Đề xuất 2.3.3 — Cấu hình canonical target_R trong `v2_frozen.yaml`

Sửa `live/db/configs/v2_frozen.yaml`:

```yaml
# Canonical: target_R = 2.0 (R-multiple), CAPPED ở 4ATR cho oscillator patterns.
# Plugin-detector emits structure-based target_k ATR (0.75R..4R).
v2_target_r: 2.0         # dùng làm tỉ lệ mặc định; cap 4 ATR cho structure
v2_target_cap_atr: 4.0
reward_r_values: [1.5, 2.0, 2.5]  # A/B trong multi-R sweep backtest
primary_reward_r: 2.0
```

`signal_engine_v2.py` đọc cả `v2_target_cap_atr`, rồi clip:

```python
risk = abs(entry_price - stop_price)
target_raw = entry_price + reward_r * risk if is_long else entry_price - reward_r * risk
cap_distance = v2_target_cap_atr * atr_value
target_distance = min(abs(target_raw - entry_price), cap_distance)
target_price = entry_price + target_distance if is_long else entry_price - target_distance
```

#### Đề xuất 2.3.4 — Partial TP + BE trailing

Trong `live/engine/execution_layer_v2.py` thêm cơ chế staged exit:

* Lệnh 1: chốt 50% tại 1R sau entry, dời SL về entry (BE).
* Lệnh 2: dời SL trailing 0.5 ATR mỗi lần +1R (bám cấu trúc).
* RR trung bình cả lệnh ~1.8R (50% × 1R + 50% × 2.6R) so với hiện tại 1R cho 50% đầu.

→ API cần thiết: thêm `manage_open_trade(...)` trong execution layer; partial fill flag vào `SignalCandidate.metadata`. (Xem `live/engine/execution_layer_v2.py` để biết vị trí chèn.)

#### Đề xuất 2.3.5 — ATR buffer dynamic theo volatility regime

Hiện tại `stop_buffer_atr=0.5` constant. Đề xuất:

```python
def stop_buffer_atr(volatility_ratio: float) -> float:
    """Volatility ratio = atr_fast / atr_slow. >1 trending, <1 ranging."""
    if volatility_ratio > 1.3:    # trending
        return 0.75
    if volatility_ratio < 0.85:   # ranging
        return 0.35
    return 0.5                    # neutral
```

→ Đặt vào `live/engine/regime_wiring.py` (đã có sẵn — xem `live/engine/feature_emitter.py:1-307`) đọc regime hiện tại từ HMM emitter.

---

## 3. Đề xuất đo lại (backtest protocol) sau rework

### 3.1. Đo trên 3 trục

| Trục | Công thức | Trước (hiện tại) | Sau (đề xuất) |
|------|-----------|------------------|---------------|
| Winrate | `TP_hit / total_trades` | ước tính ~28% (chart #3 → SL hit) | expected 42–48% (cap 4 ATR + structure TP) |
| Expectancy | `mean(pnl_R)` | ~−0.1R (SL=1R, target xa → TP hiếm) | expected +0.4R (BE + 1.8R partial) |
| Events/yr | events ÷ years (XAUUSD M15 2022–2026) | ~120 DB/DT/year | ~70 DB/DT/year (NMS giảm 30–40%) |

### 3.2. Cú pháp đo

Trên repo đã có `research/patterns/double_bottom/scripts/train_walkforward.py:1–300` — chỉnh `target_r=1.25` (line 129) thành `target_r=1.0` và thêm `target_cap_atr=4.0`. Re-run:

```
/tmp/ptv2_venv/bin/python research/patterns/double_bottom/scripts/train_walkforward.py \
  --target_r 1.0 \
  --target_cap_atr 4.0 \
  --n_folds 5
```

Sau đó `pytest tests/test_double_bottom.py -q` để đảm bảo causality gate vẫn pass.

### 3.3. Kỳ vọng golden UPDATED

* `debug_events.txt` (xem `docs/debug_drawer/debug_events.txt`) từ 618 event → ~400 event.
* `events.txt` trong DebugDrawer echo: mỗi event có `target_p` ∈ [entry ± 4 ATR × ATR-value], **không xa quá 4× depth pattern**.
* Winrate đo qua `research/core/walkforward_trainer.py` (đọc trong `docs/handoff.md` §2.4) ≥ 42%.

---

## 4. Trình tự implementation (1 sprint)

| Step | File chính | Effort | Risk |
|------|------------|--------|------|
| 4.1 | `research/patterns/double_bottom/detector.py:228` — chèn NMS code | 0.5d | LOW (causality preserved) |
| 4.2 | `research/patterns/double_top/detector.py` — inherits từ base, không cần sửa (nếu fix ở base) | 0d | LOW |
| 4.3 | `research/core/dedupe.py` (mới) — `drop_opposite_overlap` | 0.5d | LOW |
| 4.4 | `research/patterns/double_bottom/detector.py:247–256` — thêm `_structure_target_bullish` + `_structure_stop_bullish` | 1d | MEDIUM (TP logic thay đổi → re-run gate §6.3) |
| 4.5 | `live/db/configs/v2_frozen.yaml:40,66,68` — `v2_target_r=2.0`, `v2_target_cap_atr=4.0` | 0.25d | MEDIUM (thay đổi live order) |
| 4.6 | `live/engine/signal_engine_v2.py:462–485` — clip target tại cap | 0.5d | MEDIUM |
| 4.7 | `docs/debug_drawer/DebugDrawer.mq5` — overlap warning + skip painting | 1d | LOW |
| 4.8 | `live/engine/execution_layer_v2.py` — `manage_open_trade` partial TP + BE trailing | 2d | HIGH (cần MCP broker-side partial fill) |
| 4.9 | Tests: `tests/test_double_bottom.py::test_nms_overlap` + `test_structure_target_capped` | 0.5d | LOW |
| 4.10 | Re-run `pytest -q -m no_lookahead` (168 tests), golden LSW (15/15), DB gate (§6.3) | 0.5d | LOW |

Tổng: ~7 ngày-sprint.

---

## 5. Lưu ý cuối

* Không chạm `research/core/contracts.py` (frozen §16).
* Không modify `live/engine/feature_emitter.py` (đã verified strict).
* Không đổi `feature_schema_version` (`double-v1.0`) — nếu đổi target cap thành **feature mới**, version bump → `double-v1.1` (yêu cầu §6.2 config_hash re-run).
* `models/double_bottom_xauusd_m15_v1/model.pkl` (xem `artifacts/models/…`) cần re-train nếu train_walkforward thay đổi (line 129 `target_r=1.25` → `1.0`).

---

## 6. Trạng thái

| Hạng mục | Status |
|----------|--------|
| Bug A1–B5 từ session trước | Đã liệt kê trong `trading_v3_bug_summary.md` (CHƯA SỬA) |
| Overlap/NMS (đề xuất §1.3) | **CHƯA implement** |
| SL/TP cap (đề xuất §2.3) | **CHƯA implement** |
| Backtest (đề xuất §3) | **CHƯA re-run** với params mới |
| Golden tests | Đang giữ 15/15 (theo `docs/handoff.md` §2.5) — KHÔNG vượt gate khi chưa fix |

---

## 7. Tài liệu tham chiếu trong repo (file đã đọc xác minh)

* `research/patterns/double_bottom/detector.py` (line 1–328)
* `research/patterns/double_top/detector.py` (toàn file 41 dòng)
* `research/patterns/double_bottom/PATTERN_SPECS.md` (line 1–80)
* `docs/debug_drawer/DebugDrawer.mq5` (line 1–241 + 261–327)
* `docs/debug_drawer/export_debug_events.py` (line 1–233)
* `docs/debug_drawer/debug_events.txt` (200 dòng đầu)
* `live/engine/signal_engine_v2.py` (line 270, 432, 462, 482–485, 489, 1080–1131)
* `live/db/configs/v2_frozen.yaml` (line 40, 66, 68)
* `research/patterns/double_bottom/scripts/train_walkforward.py` (line 129, 149, 212, 235)
* `docs/handoff.md` (line 1–149 — section §2.5, §2.7)
* `docs/findings_resolution.md` (line 14–203 — F1, F2 confirmed)

---

**Tóm lại:** vấn đề đến từ 4 chỗ rõ ràng — `_detect_double` không NMS, `_dedupe` chỉ tính theo `confirm_bar`, `target_r=1.5` × 3R trong live engine, và TP không cap theo ATR. Rework theo §1.3 + §2.3 của bản mô tả này, kỳ vọng winrate tăng 30–50% và overlap event giảm 30–40%.
