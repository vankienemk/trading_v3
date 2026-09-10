# REVERSAL PATTERN ENGINE — Đặc tả Hệ thống Chuẩn hoá Multi-Pattern

**Phiên bản:** 1.1
**Ngày:** 2026-09-07
**Trạng thái:** Supersedes v1.0 (giữ nguyên toàn bộ nguyên tắc nền, bổ sung 4 hợp đồng mới)
**Mục tiêu:** Chuẩn hoá engine Liquidity Sweep thành **Pattern Plugin Architecture**, cho phép cắm bất kỳ pattern đảo chiều nào, gắn pattern(s) + model đã train cho từng symbol qua GUI Onboarding, giữ nguyên tính liền mạch với pipeline research → train → live của dự án `trading_lid_060826`.

**Thay đổi chính so với v1.0:**
1. §4 — **Dedup / Correlation Contract**: đặc tả đầy đủ cơ chế khử trùng lặp và cộng hưởng giữa các pattern (v1.0 chỉ là 1 comment).
2. §5 — **Pattern Lifecycle & Shadow Mode**: vòng đời model/pattern `validated → shadow → live → degraded → retired`, shadow trading không tạo lệnh thật.
3. §3.4 — **Causal Rules cho pattern giá cổ điển**: quy tắc pivot right-bar, staleness window, feature `available_at ≤ known_at` runtime check.
4. §7 — **Event Lake**: mọi PatternEvent (kể cả bị reject) đều persist kèm outcome.
5. §8 — **PatternSynthesizer**: sinh OHLC có ground-truth để benchmark detector.
6. §13 — **Task breakdown chi tiết cho 8 Agents** với dependencies và Definition of Done.

---

## 1. Tóm tắt hiện trạng & Mục tiêu chuyển đổi

### 1.1. Engine hiện tại
- **Pattern duy nhất:** Liquidity Sweep (Low Sweep + Reclaim bullish / High Sweep + Reclaim bearish).
- Pipeline research: Data Audit → Events (sweep + confirmation) → Dataset (features + labels + costs + rule_score) → Train (walk-forward + calibration).
- Live: `signal_engine_v2.py` load model từ Model Registry theo `model_id` của symbol → detect sweep → tính score/prob → tạo `PendingSignal`.
- GUI Paper Trading V2: 5 tabs (Onboarding / Live Control / Performance / Account / Log), kết nối MT5 qua MCP, chỉ symbol `status=validated` + có `model_id` mới được xử lý.
- Nguyên tắc bất di bất dịch: **no look-ahead**, tách setup / confirmation / entry / label, backtest có đủ costs, kết quả tái lập được, CI gate (ruff + mypy + pytest + no_lookahead).

### 1.2. Mục tiêu v1.1
1. Chuẩn hoá Liquidity Sweep thành **một plugin** trong hệ thống multi-pattern.
2. Hỗ trợ cắm thêm mọi pattern đảo chiều (Double Top/Bottom, H&S, Wedge, Cup & Handle, …).
3. GUI Onboarding: gắn **nhiều pattern** cho symbol, chọn **model đã train** tương thích, quản lý **lifecycle state** và **shadow mode**.
4. Mỗi pattern có pipeline research riêng (hoặc shared) → train model riêng → đăng ký Model Registry với metadata `pattern_name` + `feature_schema_version` + `lifecycle_state`.
5. Live Engine load đúng detector + model theo config của symbol, chạy **Dedup/Correlation layer** trước khi tạo `PendingSignal`.
6. **Event Lake**: lưu đầy đủ attributes của mọi event (kể cả rejected) để train / retrain / mine false-negative.
7. Multi-pattern backtest + báo cáo chi tiết per-pattern + so sánh + confluence analysis.

---

## 2. Kiến trúc tổng thể (Plugin-based)

```
trading_v3/
├── live/
│   ├── engine/
│   │   ├── signal_engine_v2.py          # Orchestrator (giữ interface cũ)
│   │   ├── pattern_registry.py          # Load plugins động
│   │   ├── correlation_manager.py       # [MỚI] Dedup / Confluence (§4)
│   │   ├── lifecycle_manager.py         # [MỚI] Shadow / auto-demotion (§5)
│   │   └── ...
│   └── gui/                             # Onboarding + Pattern Performance tab (§10)
├── research/
│   ├── core/                            # Shared: data, indicators, labeling, scoring, modeling
│   │   ├── pattern_synthesizer.py       # [MỚI] Ground-truth generator (§8)
│   │   └── causal_checks.py             # [MỚI] Runtime available_at validator (§3.4)
│   ├── patterns/                        # Mỗi pattern = 1 package plugin
│   │   ├── liquidity_sweep/             # Pattern hiện tại (đã có)
│   │   ├── double_top/
│   │   ├── double_bottom/
│   │   ├── rising_wedge/
│   │   ├── falling_wedge/
│   │   ├── head_shoulders/
│   │   ├── inverse_head_shoulders/
│   │   ├── triple_top_bottom/
│   │   ├── cup_and_handle/
│   │   └── ... (dễ mở rộng)
│   └── multi_backtest/                  # Runner so sánh nhiều pattern
├── event_lake/                          # [MỚI] Append-only PatternEvent store (§7)
└── model_registry/
    └── index.yaml                       # + pattern_name, feature_schema, lifecycle_state
```

### 2.1. Core Contract (BasePatternDetector + PatternEvent)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class PatternEvent:
    # --- Identity ---
    event_id: str
    pattern_name: str                    # "liquidity_sweep", "double_bottom", ...
    pattern_version: str
    symbol: str
    timeframe: str
    direction: str                       # "bullish" | "bearish"

    # --- Timing (causal) ---
    detect_time: pd.Timestamp            # thời điểm setup được nhận diện
    confirm_time: Optional[pd.Timestamp] # thời điểm confirmation (break neckline, reclaim...)
    known_at: pd.Timestamp               # = max(detect_time, confirm_time) — thời điểm THỊ TRƯỜNG biết event
    entry_time: Optional[pd.Timestamp]

    # --- Price levels ---
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    structure_levels: Dict[str, float]   # neckline, sweep_low, shoulder_prices...

    # --- Scores ---
    rule_score: float                    # chất lượng pattern theo rule (0-1)
    model_prob: Optional[float]          # xác suất từ ML filter (sau calibration)

    # --- Lineage & audit ---
    config_hash: str                     # hash(detector_config + detector.version + indicators_version)
    feature_schema_version: str
    attributes: Dict[str, Any] = field(default_factory=dict)  # full pattern-specific attrs

    # --- Lifecycle (v1.1) ---
    lifecycle_state: str = "live"        # "shadow" | "live" — quyết định có tạo PendingSignal không
    confluence_group_id: Optional[str] = None  # gán bởi CorrelationManager (§4)
```

```python
@dataclass
class PatternFeature:
    name: str
    dtype: str
    available_at: str                    # "detect" | "confirm" | "entry" — runtime-enforced (§3.4)
    uses_future_data: bool = False       # phải luôn False; CI + runtime đều check
    description: str = ""

class BasePatternDetector(ABC):
    name: str
    version: str
    feature_schema: List[PatternFeature]

    @abstractmethod
    def detect(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[PatternEvent]: ...

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]: ...  # PHẢI chứa key "version"

    def validate_causality(self, events: List[PatternEvent]) -> None:
        """Runtime check: known_at >= confirm_time >= detect_time;
        mọi feature dùng cho scoring phải có available_at <= known_at."""
```

### 2.2. Plugin Discovery

`pattern_registry.py` quét `research/patterns/*/detector.py`, import class kế thừa `BasePatternDetector`, đăng ký vào dict `{name: detector_class}`. Plugin không import được → log warning, không crash engine (fail-open về mặt kỹ thuật, fail-closed về mặt tín hiệu: pattern lỗi = không tín hiệu).

---

## 3. Quy tắc Nhân quả (Causal Rules) — bắt buộc cho mọi pattern

### 3.1. Nguyên tắc gốc (giữ từ v1.0)
- Tách biệt hoàn toàn: **setup / confirmation / entry / label**.
- Label chỉ dùng dữ liệu SAU `entry_time` (forward window), features chỉ dùng dữ liệu TỐI ĐA tới `known_at`.
- Walk-forward với purge + embargo; kết quả tái lập được bằng seed + config hash.

### 3.2. Pivot/Swing Right-Bar Rule (MỚI — bắt buộc ghi vào PATTERN_SPECS.md của từng pattern)
Pattern giá cổ điển (Double Top, H&S, Wedge…) dựa trên swing points. Một pivot high/low tại bar `t` chỉ **tồn tại về mặt nhân quả** tại bar `t + k` (k = số right bars xác nhận). Do đó:
- Mọi detector PHẢI gán `pivot.known_at_bar = pivot_bar + right_bars`.
- Không detector nào được phép tham chiếu pivot trước khi nó "tồn tại".
- `PATTERN_SPECS.md` của từng pattern PHẢI ghi rõ: công thức swing (left/right bars), nguồn giá (high/low hay close), và min/max distance giữa các pivot.

### 3.3. Staleness Window (MỚI)
Với pattern có khoảng detect → confirm dài (H&S, Cup & Handle):
- Config bắt buộc: `max_bars_between_detect_and_confirm` (default theo pattern, ví dụ H&S M15: 60 bars).
- Event confirm sau hạn này → **discard**, vẫn log vào Event Lake với `attributes.discard_reason = "stale"`.
- Tại entry: nếu `current_price` đã đi quá `entry_price ± max_entry_drift_atr * ATR` → không tạo signal (tránh đuổi giá).

### 3.4. Runtime Causal Check (MỚI — bổ sung cho CI)
CI marker `no_lookahead` là static analysis; v1.1 thêm **runtime check** trong `causal_checks.py`:
- Khi detector trả events, engine chạy `validate_causality()`: với mỗi feature trong `feature_schema`, assert `available_at` tương ứng ≤ `known_at` của event; assert `uses_future_data == False`.
- Vi phạm → raise `CausalityViolation`, event bị drop, alert lên GUI Log tab.
- Chạy ở cả research (trên toàn bộ dataset) lẫn live (per event).

### 3.5. Timeframe-per-Assignment (làm rõ v1.0)
- `timeframe` là thuộc tính của **pattern assignment** (symbol × pattern), KHÔNG phải của symbol.
- Engine hỗ trợ nhiều TF đồng thời: mỗi assignment có polling loop/fetch riêng; dữ liệu TF khác nhau không được resample chéo trừ khi feature khai báo rõ trong schema.

---

## 4. Dedup / Correlation Contract (MỚI — mục quan trọng nhất của v1.1)

### 4.1. Vấn đề
Khi nhiều pattern active trên 1 symbol, các event thường confirm trên **cùng 1–2 nến** (ví dụ Liquidity Sweep tạo bottom thứ 2 của Double Bottom). Nếu RiskGuard không biết 2 tín hiệu là "cùng một ý tưởng", rủi ro thực tế bị nhân đôi vô hình.

### 4.2. CorrelationManager — Correlation Window
Hai event `e1`, `e2` thuộc **cùng correlation group** khi THỎA ĐỒNG THỜI:
1. Cùng `symbol` và cùng `direction`.
2. `|e1.known_at − e2.known_at| ≤ corr_time_window` (default: 5 bars của TF lớn hơn trong 2 TF).
3. `|e1.entry_price − e2.entry_price| ≤ corr_price_window_atr × ATR(14)` tại `known_at` (default: 0.5 ATR).

Grouping là transitive (union-find): nếu e1~e2 và e2~e3 thì cả 3 thuộc một group, kể cả khi e1 và e3 không trực tiếp thỏa điều kiện.

### 4.3. Chính sách xử lý group — 3 mode cấu hình per symbol

| Mode | Hành vi | Khi nào dùng |
|------|---------|--------------|
| `dedup` (default) | Chỉ giữ 1 event tốt nhất theo `priority = model_prob × rule_score` (fallback: rule_score). Các event còn lại log `discard_reason = "correlated"` vào Event Lake. | Giai đoạn đầu, chưa có đủ data confluence |
| `confluence` | Gộp thành 1 signal, `confluence_score` = f(số pattern đồng thuận, độ gần thời gian/giá); size được nhân hệ số (≤ cap §9.3). | Sau khi meta-model confluence pass gate (§6.4) |
| `independent` | Không dedup (chỉ dùng khi 2 pattern đã chứng minh độc lập thống kê, ví dụ khác TF xa nhau). Cần approval trong config review. | Hiếm — phải justify bằng correlation matrix OOS |

### 4.4. Exposure Caps (bắt buộc, không phụ thuộc mode)
- `max_total_risk_per_symbol`: tổng risk mọi position đang mở của 1 symbol ≤ cap (default 1.0× risk đơn lẻ chuẩn).
- `max_direction_cluster_risk`: tổng risk các position cùng direction có entry chênh < 1 ATR ≤ cap (default 0.75×).
- Kill-switch per-symbol hiện hữu vẫn áp dụng trên toàn bộ assignments của symbol đó.

### 4.5. Confluence Score (định nghĩa chuẩn)
```
confluence_score = w1 × (n_patterns - 1) / (n_patterns_max - 1)
                 + w2 × exp(-Δt / τ_time)
                 + w3 × exp(-Δprice_atr / τ_price)
```
với `w1 + w2 + w3 = 1` (default 0.5/0.25/0.25), `τ_time = corr_time_window`, `τ_price = corr_price_window_atr`. Score này là **feature đầu vào** cho meta-model (§6.4), không phải tiêu chí pass/fail.

---

## 5. Pattern Lifecycle & Shadow Mode (MỚI)

### 5.1. State machine

```
                 research gate pass              shadow gate pass
  trained ───────────────► validated ─────────────────► shadow ──► live
                              │                            │         │
                              │                            │         ▼
                              │                       shadow fail  degraded ──► retired
                              │                            ▲         │
                              └──────── retrain ◄──────────┴─────────┘
```

| State | Ý nghĩa | Quyền hạn |
|-------|---------|-----------|
| `trained` | Model vừa train xong, chưa qua gate | Chỉ nằm trong registry, GUI không cho gắn |
| `validated` | Pass research gates (§6.3) | Được gắn vào symbol, được bật shadow |
| `shadow` | Chạy live đầy đủ: detect → score → ghi event + hypothetical outcome vào Event Lake, **KHÔNG tạo PendingSignal** | Đọc data thị trường, ghi Event Lake; RiskGuard không thấy |
| `live` | Tạo PendingSignal bình thường | Full |
| `degraded` | Live metrics suy giảm → auto hoặc manual hạ cấp | Không tạo signal mới; position cũ vẫn quản lý |
| `retired` | Kết thúc vòng đời | Read-only, giữ để audit |

### 5.2. Shadow Mode — cầu nối research → live
- Mục tiêu: lấp khoảng trống "backtest pass ≠ live tốt" (slippage thật, latency, spread regime, data feed khác research).
- Config: `shadow_min_events` (default 30) và `shadow_max_days` (default 30) — điều kiện nào tới trước thì evaluate.
- **Shadow gate** (promote → live) yêu cầu:
  1. `n_shadow_events ≥ shadow_min_events`.
  2. Live PF (hypothetical, có trừ costs thật từ fill gần nhất) nằm trong CI của backtest OOS PF (không lệch quá 1σ).
  3. Feature distribution drift: PSI < 0.1 trên mọi feature core.
  4. Không có `CausalityViolation` nào trong suốt shadow period.

### 5.3. Auto-Demotion Triggers (LifecycleManager, chạy định kỳ)
| Trigger | Ngưỡng default | Hành động |
|---------|----------------|-----------|
| Live PF CI lower < breakeven | sau ≥ 30 trades live | live → degraded |
| PSI feature drift | PSI ≥ 0.2 trên ≥ 2 core features | flag retrain + cảnh báo GUI; ≥ 0.25 → degraded |
| Win rate sụt đột biến | p-value < 0.05 (binomial test vs backtest WR) | live → degraded |
| Consecutive losses | ≥ `max_consec_losses` (per assignment) | pause assignment, giữ nguyên state |

### 5.4. Per-Pattern Kill-Switch (bắt buộc, không còn optional như v1.0)
- LifecycleManager theo dõi rolling metrics per `(symbol, pattern_name, timeframe)`.
- Disable chỉ pattern degrade, **không kéo cả symbol xuống** — đây là lý do kiến trúc multi-pattern tồn tại.
- Kill-switch per-symbol hiện hữu vẫn là lớp cuối cùng (circuit breaker toàn cục).

### 5.5. Registry schema mở rộng (index.yaml)
```yaml
- model_id: "double_bottom_btcusd_h1_v1"
  pattern_name: "double_bottom"
  symbol: "BTCUSDm"
  timeframe: "H1"
  feature_schema_version: "db_v1.0"
  lifecycle_state: "shadow"          # MỚI
  live_metrics_ref: "event_lake/metrics/double_bottom_btcusd_h1_v1.parquet"  # MỚI
  gate_passed: true
  calibrated: true
  config_hash: "a1b2c3..."
  trained_at: "2026-09-01T00:00:00Z"
```

---

## 6. Research → Train → Registry (cập nhật)

### 6.1. Pipeline per pattern (không đổi từ v1.0)
Data Audit → Events (detect + confirm) → Dataset (features + labels + costs + rule_score) → Train (walk-forward + purge + embargo + calibration) → Gate → Registry.

### 6.2. Config Hash Convention (làm rõ)
`config_hash = sha1(canonical_json(detector_config) + detector.version + shared_indicators_version)[:12]`
- `get_default_config()` PHẢI chứa `"version"` — config cũ không được âm thầm đổi ý nghĩa; thay đổi semantics = bump version.
- Mọi row dataset, mọi PatternEvent, mọi model registry entry đều ghi `config_hash` → tái lập tuyệt đối.

### 6.3. Research Gates — bổ sung Sample-Size Gate (MỚI)
Pattern giá cổ điển tần suất thấp → dễ "pass nhờ may mắn". Gate bắt buộc TRƯỚC khi `gate_passed = true`:
1. `n_events_total ≥ 300` VÀ `n_events_oos ≥ 100` (sau purge/embargo).
2. Label balance: tỷ lệ positive ∈ [10%, 90%] (ngoài khoảng này phải dùng class weighting + báo cáo riêng).
3. Effective sample size (sau de-correlation các event chồng lấn): `ESS ≥ 60% n_events_oos`.
4. OOS PF CI lower > 1.0 VÀ PR-AUC > baseline (tỷ lệ positive) với margin ≥ 5%.
5. Kết quả ổn định trên ≥ 3 walk-forward folds liên tiếp (không có fold nào PF < 0.8).

### 6.4. Meta-Labeling & Confluence Meta-Model (MỚI — training mode chuẩn)
- **Tầng 1 (primary):** rule của pattern quyết định direction + entry/SL/TP (như hiện tại).
- **Tầng 2 (meta-model):** binary classifier học "trade hay bỏ" trên tập event do tầng 1 sinh ra, features gồm: toàn bộ PatternFeature + `confluence_score` + context (session, volatility regime, spread).
- `model_prob` trong PatternEvent chính là output calibrated của tầng 2.
- Confluence meta-model chỉ được bật (mode `confluence` ở §4.3) sau khi pass riêng một research gate: chứng minh trên OOS rằng bucket `confluence_score` cao có expectancy > bucket thấp (monotonicity test).

### 6.5. Cost Model
Giữ nguyên chuẩn v1.0: spread + commission + slippage per symbol, lưu trong dataset row, dùng xuyên suốt backtest/shadow/live-hypothetical.

---

## 7. Event Lake (MỚI — nâng cấp FeatureStore thành sản phẩm hạng nhất)

### 7.1. Nguyên tắc
**Mọi PatternEvent đều được persist** — kể cả bị reject bởi threshold, bị dedup, bị discard vì stale — kèm `attributes.discard_reason` và outcome sau đó (forward return, MFE/MAE, hypothetical PnL trừ costs).

### 7.2. Schema lưu trữ
```
event_lake/
├── events/{pattern_name}/{symbol}/{yyyy-mm}.parquet      # append-only, full PatternEvent
├── outcomes/{event_id}.parquet                            # forward outcomes, join by event_id
└── metrics/{model_id}.parquet                             # rolling metrics cho LifecycleManager
```

### 7.3. Giá trị khai thác
1. **Retrain không cần chạy lại detector**: query Event Lake → rebuild dataset.
2. **Mine false-negative**: các event `rule_score` thấp nhưng outcome tốt → nguồn cải tiến rule_score và features (đây là nguồn alpha rẻ nhất).
3. **Drift monitoring**: PSI giữa phân phối feature live vs train, feed cho LifecycleManager.
4. **Confluence analysis**: join các event cùng group → tính expectancy theo số pattern đồng thuận.

---

## 8. PatternSynthesizer (MỚI — QA ground-truth)

### 8.1. Mục đích
Unit test dạng "có event không" quá yếu. Synthesizer sinh OHLC có **cài sẵn pattern với tham số kiểm soát + nhiễu**, làm benchmark khách quan cho detector.

### 8.2. API
```python
class PatternSynthesizer:
    def generate(self, pattern_name: str, n_bars: int,
                 pattern_params: Dict[str, Any],   # depth, symmetry, duration...
                 noise_sigma: float, trend_drift: float,
                 seed: int) -> Tuple[pd.DataFrame, GroundTruth]: ...
```
`GroundTruth` chứa vị trí chính xác các pivot, neckline, breakout bar → so sánh với output detector.

### 8.3. Acceptance Benchmark (gate bắt buộc cho mọi pattern mới)
Trên 500 synthetic series (grid tham số phủ miền hoạt động dự kiến):
- **Recall ≥ 80%**: detector bắt được ≥ 80% pattern đã cài (cho phép sai lệch vị trí ≤ 2 bars).
- **False positive ≤ 5%** trên series nhiễu thuần (không cài pattern).
- Robustness: recall không sụt quá 15% khi `noise_sigma` tăng gấp đôi.

---

## 9. Live Engine Integration (cập nhật flow §6 của v1.0)

### 9.1. Flow mới
```
symbol config (assignments[])                     # symbol × pattern × TF × model_id × state
        │
        ▼
for each assignment (state == "live" | "shadow"):
    df = fetch(tf=assignment.timeframe)           # TF-per-assignment (§3.5)
    events = detector.detect(df, config)
    validate_causality(events)                    # runtime check (§3.4)
    events → attach model_prob (tầng 2, calibrated)
        │
        ▼
CorrelationManager.group(events_all_assignments)  # §4.2
        │
        ├── mode=dedup/confluence → 1 signal/group
        └── mode=independent → pass-through
        │
        ▼
if assignment.state == "shadow":
    → ghi Event Lake + hypothetical tracking, KHÔNG tạo PendingSignal
else:
    → Exposure caps check (§4.4) → PendingSignal → RiskGuard → Execution
        │
        ▼
LifecycleManager (định kỳ): rolling metrics → auto-demotion (§5.3)
```

### 9.2. Order Comment Schema (chuẩn hoá)
`{pattern_short}-v{major}-{event_id_short}` — ví dụ: `DB-v1-a3f9`, `LSW-v2-77c1`.
- Đây là **khoá reconcile duy nhất** giữa trade ↔ event ↔ model khi chạy multi-pattern.
- Bảng `pattern_short` đăng ký trong `pattern_registry.py` (2–4 ký tự, unique).

### 9.3. Risk Budget per Pattern (MỚI)
- Mỗi assignment có `risk_fraction` riêng (default 1.0 = risk chuẩn của symbol).
- Confluence signal (mode `confluence`) được nhân tối đa `confluence_size_boost` (default 1.5×), **nhưng tổng vẫn ≤ `max_total_risk_per_symbol`** (§4.4).
- RiskGuard mở rộng nhận thêm `assignment_id` để enforce per-pattern caps.

---

## 10. GUI Onboarding & Performance (cập nhật)

### 10.1. Tab Onboarding
- Multi-select pattern cho symbol; mỗi pattern → dropdown model **đã lọc theo `pattern_name` + `feature_schema_version` + `lifecycle_state ∈ {validated, shadow, live}`**.
- Toggle per assignment: `state` (shadow/live), `risk_fraction`, TF.
- Badge màu lifecycle: shadow (xanh dương), live (xanh lá), degraded (vàng), retired (xám).
- Giữ nguyên quy tắc an toàn hiện hữu: chỉ `validated` trở lên mới gắn được; đổi state từ live → auto-deactivate.

### 10.2. Tab Performance — Pattern Breakdown (MỚI)
- Bảng per `(pattern, TF)`: trades, winrate, PF, expectancy, rolling equity per pattern.
- Mục đích: operator nhìn ra ngay pattern nào đang kéo symbol xuống — điều kiện tiên quyết để vận hành multi-pattern.
- Confluence panel: expectancy theo số pattern đồng thuận (validate giả thuyết §6.4 bằng data thật).

### 10.3. Backtest Compare trong GUI (MỚI, nice-to-have)
- Nút "Compare patterns" gọi `multi_backtest` runner, hiển thị bảng so sánh + equity curves chồng nhau, thay vì bắt operator chạy CLI.

---

## 11. PATTERN_SPECS.md — Lộ trình pattern (thay đổi thứ tự so với v1.0)

Thứ tự theo tỷ lệ **effort / edge** (Double Bottom trước, H&S sau):

| Ưu tiên | Pattern | Lý do | Chia sẻ infra |
|---------|---------|-------|---------------|
| P0 | Liquidity Sweep (refactor) | Baseline, regression test | — |
| P1 | **Double Bottom** | Dễ nhất, event count cao, cùng pivot infra với Sweep | swing detector, ATR, neckline |
| P2 | **Double Top** | Mirror của Double Bottom (gần như free) | 100% infra của P1 |
| P3 | **Rising/Falling Wedge** | Trendline fitting độ khó vừa | pivot infra + linear regression channel |
| P4 | **Head & Shoulders / Inverse** | Khó nhất: nhiều tham số symmetry, staleness dài | pivot + trendline + §3.3 quan trọng nhất ở đây |
| P5 | Cup & Handle | Dài, ít event — chỉ thực sự có giá trị ở D1 | toàn bộ infra trên |
| P6 | Triple Top/Bottom | Sau khi Double đã ổn (là generalization) | infra của P1/P2 |

Mỗi `PATTERN_SPECS.md` bắt buộc có: định nghĩa hình học, swing formula (left/right bars + giá nguồn), confirmation rule, entry/SL/TP, staleness window, default config + version, danh sách features với `available_at`, expected event frequency (ước lượng từ data audit).

---

## 12. Multi-Pattern Backtest & Reporting

- `multi_backtest/runner.py`: chạy N pattern trên cùng symbol/period, áp dụng **đúng CorrelationManager + exposure caps** như live (backtest phải giống live — cùng code path).
- Report bắt buộc:
  1. Per-pattern: trades, PF, expectancy, max DD, PR-AUC, rolling PF.
  2. Portfolio: equity tổng hợp sau dedup/confluence + caps.
  3. Correlation matrix giữa các cặp pattern (% thời gian cùng group).
  4. Confluence buckets: expectancy theo số pattern đồng thuận.
  5. So sánh "multi-pattern portfolio" vs "best single pattern" — nếu portfolio không thắng best-single trên OOS, đây là tín hiệu vàng cần review trước khi deploy.

---

## 13. Task Breakdown — 8 Agents

> Quy ước: **DoD** = Definition of Done. Mọi agent phải pass CI (ruff + mypy + pytest + no_lookahead marker) trước khi merge. Branch làm việc: `feature/multi-pattern-engine`.

### Agent 1 — Core Contracts & Causal Infrastructure
**Phụ thuộc:** không (start ngay).
1. Freeze `PatternEvent` (bản v1.1 đầy đủ, gồm `lifecycle_state`, `confluence_group_id`) + `PatternFeature` + `BasePatternDetector` vào `research/core/contracts.py`.
2. Implement `causal_checks.py`: `validate_causality()` runtime (§3.4) + `CausalityViolation`.
3. Implement `config_hash` convention (§6.2) + unit test canonical JSON.
4. Viết CI test template `no_lookahead` mà mọi pattern test phải kế thừa.
**DoD:** contracts import được từ mọi nơi; 100% mypy strict trên core; test cố tình vi phạm causality phải raise.

### Agent 2 — Refactor Liquidity Sweep → Plugin (Regression Baseline)
**Phụ thuộc:** Agent 1 (contracts).
1. Di chuyển detector sweep hiện tại vào `research/patterns/liquidity_sweep/` implement `BasePatternDetector`.
2. Không đổi logic: **bit-identical events** so với engine cũ trên cùng dataset (golden test).
3. `PATTERN_SPECS.md` cho liquidity_sweep (chuẩn §11).
4. Model Registry: thêm fields mới, migrate entry hiện có (`lifecycle_state="live"` cho model đang chạy).
**DoD:** golden test pass (event-by-event equal); regression suite hiện hữu xanh; không đổi interface `signal_engine_v2` ra ngoài.

### Agent 3 — Double Bottom / Double Top Plugin
**Phụ thuộc:** Agent 1; Agent 8 (synthesizer) chạy song song.
1. Shared `swing_detector` trong core (left/right bars configurable, pivot `known_at_bar` đúng §3.2).
2. Implement `double_bottom` rồi `double_top` (mirror), kèm `PATTERN_SPECS.md` đầy đủ.
3. Dataset builder + labeling (forward MFE/MAE, triple-barrier hoặc fixed-horizon — ghi rõ trong spec).
4. Pass Synthesizer benchmark (§8.3) + research gates (§6.3) trên ≥ 2 symbols.
**DoD:** Synthesizer benchmark report đính kèm PR; gate report có ESS + label balance; zero `CausalityViolation` trên toàn dataset.

### Agent 4 — Wedge + Head & Shoulders Plugins
**Phụ thuộc:** Agent 3 (tái dùng pivot + trendline infra).
1. Trendline fitting module (core): linear regression trên pivots + tolerance ATR.
2. `rising_wedge` / `falling_wedge` (P3) → `head_shoulders` / `inverse_head_shoulders` (P4).
3. Staleness window (§3.3) bắt buộc có test riêng cho H&S.
4. Synthesizer benchmark + research gates như Agent 3.
**DoD:** tương tự Agent 3; H&S phải có test chứng minh không dùng pivot trước khi tồn tại (§3.2).

### Agent 5 — Event Lake & Feature Store
**Phụ thuộc:** Agent 1 (schema PatternEvent).
1. Implement append-only writer + reader theo layout §7.2 (parquet, partition theo pattern/symbol/month).
2. Outcome tracker: job định kỳ tính forward outcomes + hypothetical PnL cho mọi event (live, shadow, rejected).
3. Drift monitor: PSI per feature (live vs train) → expose cho LifecycleManager.
4. Query API: `rebuild_dataset(pattern, symbol, from, to)` phục vụ retrain không cần chạy lại detector.
**DoD:** round-trip test (write → read → identical); outcome join đúng 100% event_id; PSI job chạy được trên dữ liệu giả lập.

### Agent 6 — Live Engine: Correlation, Lifecycle, GUI
**Phụ thuộc:** Agent 1, 2, 5.
1. `correlation_manager.py`: grouping (§4.2, union-find), 3 mode (§4.3), exposure caps (§4.4) tích hợp RiskGuard.
2. `lifecycle_manager.py`: state machine §5.1, shadow gate §5.2, auto-demotion §5.3.
3. Update `signal_engine_v2.py` theo flow §9.1 (giữ interface cũ ra ngoài).
4. Order comment schema §9.2 + migration notes cho execution layer.
5. GUI: Onboarding multi-pattern (§10.1) + Performance breakdown (§10.2).
**DoD:** integration test end-to-end trên paper account: 2 pattern cùng bắn 1 nến → chỉ 1 lệnh (dedup) hoặc 1 lệnh size boost (confluence); shadow assignment không tạo lệnh; kill-switch per-pattern hoạt động.

### Agent 7 — Bias Auditor & QA
**Phụ thuộc:** chạy song song từ đầu, review cuối mỗi PR.
1. Review MỌI detector PR: kiểm tra §3.1–3.5 bằng checklist (pivot known_at, feature available_at, staleness).
2. Adversarial tests: cố tình inject look-ahead (shift -1, dùng close của confirm bar trong detect features...) → CI phải bắt được.
3. Audit labeling: không leakage giữa forward window và features; purge/embargo đúng.
4. Audit backtest: costs đầy đủ, code path backtest ≡ live (§12).
**DoD:** signed-off checklist trên từng PR; adversarial suite nằm trong CI cố định.

### Agent 8 — PatternSynthesizer & Multi-Backtest Runner
**Phụ thuộc:** Agent 1.
1. `pattern_synthesizer.py` (§8): generator cho sweep/double/wedge/H&S + GroundTruth.
2. Acceptance benchmark harness (§8.3) chạy trong CI cho mọi pattern mới.
3. `multi_backtest/runner.py` + report generator đủ 5 mục §12 (dùng chung CorrelationManager với live).
4. Cung cấp synthetic fixtures cho Agent 3/4/7.
**DoD:** benchmark chạy < 10 phút trong CI; report mẫu trên BTCUSDm + XAUUSDm cho ≥ 3 pattern.

### Dependency Graph
```
A1 ──► A2 ──► A6 ◄── A5 ◄── A1
A1 ──► A8 ──► A3 ──► A4
A7: review tất cả (liên tục)
```
Tuần 1: A1 + A8 + A7(setup). Tuần 2: A2 + A5 + A3 start. Tuần 3–4: A3/A4 + A6. Tuần 5: integration + shadow đầu tiên trên paper.

---

## 14. Rủi ro & Biện pháp giảm thiểu (cập nhật từ v1.0)

| Rủi ro | Giảm thiểu |
|--------|------------|
| Look-ahead qua pivot chưa tồn tại | §3.2 bắt buộc + runtime `validate_causality` + adversarial CI (Agent 7) |
| Rủi ro nhân đôi vô hình khi multi-pattern cùng bắn | CorrelationManager §4 + exposure caps bắt buộc §4.4 |
| Model "pass gate nhờ may mắn" (event ít) | Sample-size gate §6.3 (n_oos ≥ 100, ESS ≥ 60%) |
| Backtest pass nhưng live chết | Shadow mode §5.2 bắt buộc trước khi live |
| Feature schema drift train/live | Lock schema version trong model_id + runtime check + PSI monitor |
| 1 pattern degrade kéo cả symbol | Per-pattern kill-switch §5.4 |
| Phá vỡ Liquidity Sweep hiện tại | Golden test bit-identical (Agent 2) trước mọi merge |
| Confluence mode tăng size sai lúc | Chỉ bật sau khi meta-model pass monotonicity gate §6.4 |
| GUI phức tạp | Multi-select + dropdown đã lọc; badge lifecycle; không thêm tab mới ngoài Performance breakdown |

---

## 15. Acceptance Criteria tổng (Definition of Done toàn dự án)

1. Golden test Liquidity Sweep bit-identical sau refactor.
2. ≥ 3 pattern mới (Double Bottom, Double Top, 1 trong {Wedge, H&S}) pass Synthesizer benchmark + research gates.
3. End-to-end trên paper: 2 pattern cùng confirm 1 nến → đúng hành vi theo mode cấu hình; shadow không tạo lệnh; per-pattern kill-switch tự kích hoạt khi inject losses giả.
4. Multi-backtest report đủ 5 mục §12 trên ≥ 2 symbols.
5. Event Lake chứa 100% events (kể cả rejected) với outcome đầy đủ sau forward window.
6. CI xanh: ruff + mypy + pytest + no_lookahead + adversarial suite.
7. GUI: gắn multi-pattern cho symbol, đổi lifecycle state, xem breakdown per-pattern — không cần đụng CLI.

---

## 16. Kết luận & Next Action

v1.1 giữ nguyên triết lý v1.0 (plugin hoá, causal-first, tái lập được) và bịt 4 lỗ hổng lớn nhất: **correlation vô hình**, **thiếu cầu nối research→live (shadow)**, **look-ahead ngầm của pattern giá cổ điển**, và **gate thiếu sample-size**. Đồng thời nâng multi-pattern từ "thêm tín hiệu" thành "edge mới" qua Confluence Score + meta-labeling + Event Lake mining.

**Hành động ngay:**
1. Captain tạo branch `feature/multi-pattern-engine`.
2. Freeze contracts v1.1 (Agent 1) — mọi thay đổi sau đó phải qua PR riêng có Agent 7 sign-off.
3. Khởi động song song Agent 1 + Agent 8 + Agent 7 (setup).
4. Đăng ký `pattern_short` cho Liquidity Sweep (`LSW`) ngay trong PR đầu tiên của Agent 2.

File này là nguồn sự thật duy nhất (single source of truth) cho toàn bộ team agents, thay thế v1.0.

---

*Tài liệu v1.1 kế thừa kiến trúc `trading_lid_060826`, spec v1.0 (2026-09-07), và bổ sung các hợp đồng: Dedup/Correlation (§4), Lifecycle & Shadow (§5), Causal Rules mở rộng (§3), Event Lake (§7), PatternSynthesizer (§8), cùng task breakdown 8 agents (§13).*
