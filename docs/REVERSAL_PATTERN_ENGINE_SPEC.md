# REVERSAL PATTERN ENGINE — Đặc tả Hệ thống Chuẩn hoá Multi-Pattern

**Phiên bản:** 1.0  
**Ngày:** 2026-09-07  
**Mục tiêu:** Chuẩn hoá engine hiện tại (Liquidity Sweep) thành **Pattern Plugin Architecture**, cho phép cắm bất kỳ pattern đảo chiều nào, gắn pattern(s) + model đã train cho từng symbol qua GUI Onboarding, giữ nguyên tính liền mạch với pipeline research → train → live của dự án `trading_lid_060826`.

---

## 1. Tóm tắt hiện trạng & Mục tiêu chuyển đổi

### 1.1. Engine hiện tại
- **Pattern duy nhất:** Liquidity Sweep (Low Sweep + Reclaim bullish / High Sweep + Reclaim bearish).
- Pipeline research: Data Audit → Events (sweep + confirmation) → Dataset (features + labels + costs + rule_score) → Train (walk-forward + calibration).
- Live: `signal_engine_v2.py` load model từ Model Registry theo `model_id` của symbol → detect sweep → tính score/prob → tạo `PendingSignal`.
- Nguyên tắc bất di bất dịch: **no look-ahead**, tách setup / confirmation / entry / label, backtest có đủ costs, kết quả tái lập được, CI gate (ruff + mypy + pytest + no_lookahead).

### 1.2. Mục tiêu mới
1. Chuẩn hoá Liquidity Sweep thành **một plugin** trong hệ thống multi-pattern.
2. Hỗ trợ cắm thêm mọi pattern đảo chiều (Head & Shoulders, Double Top/Bottom, Triple, Wedge, Cup & Handle, …).
3. GUI Onboarding cho phép:
   - Gắn **một hoặc nhiều pattern** cho symbol.
   - Chọn **model đã được train** (model_id) tương thích với pattern đó.
4. Mỗi pattern có pipeline research riêng (hoặc shared) → train model riêng → đăng ký vào Model Registry với metadata `pattern_name` + `feature_schema_version`.
5. Live Engine load đúng detector + model theo config của symbol.
6. Lưu đầy đủ attributes của mọi event để train / retrain ML filter phía sau.
7. Multi-pattern backtest + báo cáo chi tiết per-pattern + so sánh.

---

## 2. Kiến trúc tổng thể (Plugin-based)

```
trading_live/
├── live/
│   ├── engine/
│   │   ├── signal_engine_v2.py          # Orchestrator (giữ interface cũ)
│   │   ├── pattern_registry.py          # Load plugins động
│   │   └── ...
│   └── gui/                             # Onboarding cập nhật
├── research/
│   ├── core/                            # Shared: data, indicators, labeling, scoring, modeling
│   ├── patterns/                        # Mỗi pattern = 1 package plugin
│   │   ├── liquidity_sweep/             # Pattern hiện tại (đã có)
│   │   ├── head_shoulders/
│   │   ├── inverse_head_shoulders/
│   │   ├── double_top/
│   │   ├── double_bottom/
│   │   ├── triple_top_bottom/
│   │   ├── rising_wedge/
│   │   ├── falling_wedge/
│   │   ├── cup_and_handle/
│   │   └── ... (dễ mở rộng)
│   └── multi_backtest/                  # Runner so sánh nhiều pattern
└── model_registry/
    └── index.yaml                       # Thêm field pattern_name, feature_schema
```

### 2.1. Core Contract (BasePatternDetector)

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
    detect_time: pd.Timestamp            # thời điểm pattern được nhận diện (setup)
    confirm_time: Optional[pd.Timestamp] # confirmation (nếu có)
    entry_time: Optional[pd.Timestamp]
    known_at: pd.Timestamp               # max(detect, confirm) — thời điểm an toàn để trade

    # --- Prices ---
    entry_price: float
    stop_loss: float
    take_profit: float                   # measured move mặc định
    pattern_height: float                # dùng tính target

    # --- Quality ---
    rule_score: float                    # 0-100
    confidence_raw: float                # 0-1

    # --- Rich attributes (để train ML) ---
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Ví dụ chung: atr, volume_ratio, rsi, htf_trend, neckline_price,
    # left_shoulder_price, head_price, right_shoulder_price, age_bars,
    # touch_count, wick_ratio, penetration_atr, reclaim_atr, ...

    # --- Metadata ---
    config_hash: str
    data_version: str


class BasePatternDetector(ABC):
    """Mọi pattern phải implement interface này."""
    name: str                            # unique key
    version: str
    supported_directions: List[str]      # ["bullish"], ["bearish"], or both
    default_timeframes: List[str]        # ["M15", "H1", "H4", "D1"]

    @abstractmethod
    def detect(self, df: pd.DataFrame, config: dict) -> List[PatternEvent]:
        """
        Causal detection only.
        df phải đã được sort theo timestamp tăng dần.
        Không được dùng dữ liệu sau bar hiện tại.
        """
        ...

    @abstractmethod
    def get_feature_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Trả về schema của attributes để lock feature list.
        Format:
        {
          "feature_name": {
            "dtype": "float64",
            "description": "...",
            "available_at": "detect_time" | "confirm_time",
            "uses_future_data": False
          }
        }
        """
        ...

    def score(self, event: PatternEvent, context: dict) -> float:
        """Rule-based score 0-100. Có thể override."""
        return 50.0

    def get_default_config(self) -> dict:
        """Config mặc định của pattern."""
        return {}
```

### 2.2. Pattern Registry & Engine

```python
class PatternRegistry:
    def register(self, detector: BasePatternDetector): ...
    def get(self, name: str) -> BasePatternDetector: ...
    def list_available(self) -> List[str]: ...

class ReversalPatternEngine:
    def __init__(self, config: dict, registry: PatternRegistry): ...
    def run(self, df: pd.DataFrame, pattern_names: List[str]) -> List[PatternEvent]: ...
    def generate_report(self, events: List[PatternEvent]) -> dict: ...
```

Liquidity Sweep hiện tại được refactor thành `LiquiditySweepDetector(BasePatternDetector)`.

---

## 3. Đặc điểm chi tiết từng loại Pattern (Reversal)

Tất cả pattern dưới đây đều phải tuân thủ:
- Causal only (no look-ahead).
- Tách detect_time / confirm_time / entry_time.
- Measured move target = pattern height (hoặc 1× height).
- Stop loss nằm ngoài pattern (beyond extreme).
- Attributes đầy đủ để train ML.

### 3.1. Liquidity Sweep (pattern hiện có – baseline)
- **Loại:** Reversal / Continuation tuỳ context.
- **Bullish:** Price quét dưới liquidity low (rolling / swing / equal low) → reclaim về trên level.
- **Bearish:** Quét trên liquidity high → reclaim xuống dưới.
- **Key attributes:** level_type, level_price, penetration_atr, wick_ratio, reclaim_atr, level_age_bars, touch_count, confirmation_type.
- **Confirm:** Reclaim + (optional) close beyond level hoặc engulfing.
- **Đặc thù:** Phụ thuộc level detection (rolling/swing/equal) – đã có sẵn.

### 3.2. Head & Shoulders (Bearish Reversal)
- **Cấu trúc:** Left Shoulder → Head (cao hơn) → Right Shoulder (gần bằng Left).
- **Neckline:** Đường nối 2 đáy giữa các vai.
- **Confirm:** Close dưới neckline + volume tăng.
- **Target:** Neckline − (Head − Neckline).
- **Stop:** Trên Right Shoulder (hoặc Head).
- **Key attributes:** left_shoulder_price/time, head_price/time, right_shoulder_price/time, neckline_price, shoulder_symmetry_ratio, volume_decline_ratio (L→H→R), pattern_height, age_bars.
- **Yêu cầu tối thiểu:** 3 peaks rõ ràng, ít nhất 2–3 bars giữa các peak.

### 3.3. Inverse Head & Shoulders (Bullish Reversal)
- **Cấu trúc:** Left Shoulder → Head (thấp hơn) → Right Shoulder.
- **Neckline:** Nối 2 đỉnh giữa các vai.
- **Confirm:** Close trên neckline + volume tăng.
- **Target:** Neckline + (Neckline − Head).
- **Stop:** Dưới Right Shoulder.
- **Key attributes:** tương tự H&S nhưng inverted.

### 3.4. Double Top (Bearish)
- **Cấu trúc:** 2 đỉnh gần bằng nhau (tolerance ATR hoặc %).
- **Neckline / Valley:** Đáy giữa 2 đỉnh.
- **Confirm:** Close dưới valley.
- **Target:** Valley − (Top − Valley).
- **Stop:** Trên high của 2 tops.
- **Key attributes:** top1_price/time, top2_price/time, valley_price, top_similarity_ratio, time_between_tops, volume_at_tops.

### 3.5. Double Bottom (Bullish)
- **Cấu trúc:** 2 đáy gần bằng nhau.
- **Confirm:** Close trên peak giữa 2 đáy.
- **Target:** Peak + (Peak − Bottom).
- **Stop:** Dưới low của 2 bottoms.
- **Key attributes:** tương tự Double Top inverted.
- **Ghi chú:** Thường có tỷ lệ thành công cao hơn Double Top trên crypto/forex.

### 3.6. Triple Top / Triple Bottom
- **Cấu trúc:** 3 lần chạm kháng cự/hỗ trợ thất bại.
- **Confirm:** Phá neckline sau lần chạm thứ 3.
- **Ưu điểm:** Mạnh hơn Double vì thể hiện sự từ chối rõ ràng hơn.
- **Key attributes:** 3 extremes + neckline + touch spacing.

### 3.7. Rising Wedge (Bearish Reversal / Continuation)
- **Cấu trúc:** 2 đường xu hướng dốc lên hội tụ (higher highs + higher lows nhưng slope của highs thấp hơn).
- **Confirm:** Break xuống dưới lower trendline.
- **Target:** Chiều cao wedge tại điểm bắt đầu.
- **Key attributes:** upper_slope, lower_slope, convergence_ratio, bars_in_wedge, volume_trend (thường giảm).

### 3.8. Falling Wedge (Bullish Reversal / Continuation)
- **Cấu trúc:** 2 đường dốc xuống hội tụ.
- **Confirm:** Break lên trên upper trendline.
- **Key attributes:** tương tự Rising Wedge.

### 3.9. Cup and Handle (Bullish Continuation / Reversal nhẹ)
- **Cấu trúc:** U-shape (cup) → small pullback (handle) → breakout trên rim.
- **Confirm:** Close trên rim của cup + volume.
- **Target:** Rim + depth of cup.
- **Yêu cầu:** Cup đủ dài (khuyến nghị ≥ 30 bars trên Daily), handle nông và volume thấp.
- **Key attributes:** cup_left, cup_bottom, cup_right (rim), handle_low, handle_bars, cup_depth_atr.

### 3.10. Các pattern bổ sung (ưu tiên thấp hơn, dễ thêm sau)
- Rounding Bottom / Top
- Diamond Top / Bottom
- Broadening Formation
- Island Reversal
- Candlestick-based (Engulfing, Morning/Evening Star, Hammer tại S/R) – có thể là sub-detector.

---

## 4. Pipeline Research & Train Model (giữ liền mạch với hiện tại)

### 4.1. Luồng chuẩn (áp dụng cho mọi pattern)

```
1. Data Audit          → parquet chuẩn hoá + quality report
2. Detect Events       → PatternEvent list (causal)
3. Build Dataset       → features (từ attributes + shared indicators) 
                        + labels (triple barrier / MFE-MAE / R:R)
                        + costs (spread/slippage/commission/stop buffer)
                        + rule_score
4. Train               → time-series split + embargo + purge
                        → walk-forward
                        → model (logistic / lightgbm / …)
                        → probability calibration (isotonic)
                        → eval metrics (PR-AUC, PF, expectancy, …)
5. Artifacts           → model.pkl + calibrator + feature_schema.json 
                        + metrics.json + config snapshot
6. Register            → Model Registry (index.yaml) với metadata:
                          pattern_name, symbol, timeframe, 
                          feature_schema_version, train_period, 
                          oos_metrics, gate_passed
```

### 4.2. Yêu cầu bắt buộc từ pipeline hiện tại (không được phá vỡ)

| Nguyên tắc | Chi tiết |
|------------|----------|
| No look-ahead | Mọi feature chỉ dùng data ≤ known_at |
| Tách setup/confirm/label | Label chỉ tính từ sau entry_time |
| Costs đầy đủ | Gross → Net result_r |
| Reproducible | Config + data_version + git_commit + seed |
| Walk-forward + embargo | Tránh leakage qua overlapping events |
| Gate | CI lower bound Profit Factor > 1.0 (hoặc ngưỡng symbol-specific) mới được “validated” |
| Feature schema lock | `get_feature_schema()` phải match với lúc train |

### 4.3. Model Registry mở rộng

```yaml
# model_registry/index.yaml
models:
  - model_id: "xauusd_lsweep_v2_202608"
    pattern_name: "liquidity_sweep"
    symbol: "XAUUSD"
    timeframe: "M15"
    feature_schema_version: "1.2.0"
    path: "artifacts/models/XAUUSD/liquidity_sweep/..."
    oos_pf_ci_lower: 1.092
    gate_passed: true
    created_at: "2026-08-27"

  - model_id: "btcusd_double_bottom_h1_v1"
    pattern_name: "double_bottom"
    symbol: "BTCUSD"
    timeframe: "H1"
    ...
```

### 4.4. Multi-pattern testing

- CLI: `python -m research.multi_backtest --patterns liquidity_sweep,double_bottom,head_shoulders --symbol BTCUSD --tf H1`
- Output:
  - Per-pattern metrics table
  - Comparison chart (PF, winrate, expectancy, max DD)
  - Event charts grouped by pattern + outcome
  - Recommendation: pattern nào pass gate cho symbol nào

---

## 5. GUI Onboarding (cập nhật)

### 5.1. Thay đổi Tab Symbol Onboarding

Khi **Add / Edit Symbol**:

1. **Dropdown Symbol** (từ MCP market watch) – giữ nguyên.
2. **Multi-select Patterns** (mới):
   - List tất cả pattern có trong PatternRegistry.
   - Có thể chọn 1 hoặc nhiều (ví dụ: `liquidity_sweep` + `double_bottom`).
3. **Model Assignment** (mới – theo pattern):
   - Với mỗi pattern đã chọn → dropdown Model (chỉ hiện các model có `pattern_name` khớp + `gate_passed=true` + cùng symbol/timeframe ưu tiên).
   - Cho phép “Browse local” nếu cần.
4. **Status**: validated / candidate / rejected (giữ nguyên logic).
5. **Timeframe** (nếu multi-TF): chọn TF chính cho symbol.

**Persist:**
```json
{
  "symbol": "BTCUSDm",
  "status": "validated",
  "patterns": [
    {
      "pattern_name": "double_bottom",
      "model_id": "btcusd_double_bottom_h1_v1",
      "enabled": true
    },
    {
      "pattern_name": "liquidity_sweep",
      "model_id": "btcusd_lsweep_m15_v2",
      "enabled": true
    }
  ],
  "active": true
}
```

### 5.2. Live Control

- Pending Signals hiển thị thêm cột **Pattern**.
- Có thể filter theo pattern.
- Khi gửi order: comment chứa `pattern_name` + `event_id`.

### 5.3. Quy tắc an toàn
- Chỉ symbol `validated` + có ít nhất 1 pattern + model hợp lệ mới được Activate.
- Nếu model của pattern bị xoá / gate fail → tự động disable pattern đó.
- Kill-switch vẫn hoạt động per-symbol (hoặc per-pattern nếu cần mở rộng).

---

## 6. Live Signal Engine (thay đổi tối thiểu)

```python
# signal_engine_v2.py (pseudo)
def on_new_candle(symbol, df):
    symbol_cfg = state.get_symbol(symbol)
    if not symbol_cfg.active:
        return

    all_events = []
    for p in symbol_cfg.patterns:
        if not p.enabled:
            continue
        detector = registry.get(p.pattern_name)
        events = detector.detect(df, config=p.config)
        model = model_registry.load(p.model_id)
        for e in events:
            e.rule_score = detector.score(e, context)
            e.model_prob = model.predict_proba(e.attributes)
            if e.model_prob >= threshold and e.rule_score >= min_score:
                all_events.append(e)

    # Dedup / priority nếu nhiều pattern cùng lúc
    pending = convert_to_pending_signals(all_events)
    state.add_pending(pending)
```

Interface `PendingSignal` giữ nguyên để không phá GUI / RiskGuard / Execution.

---

## 7. Nhiệm vụ & Yêu cầu cho Team Agents

### 7.1. Thành phần Team (tối đa 8 agent)

| # | Agent | Vai trò chính |
|---|-------|---------------|
| 0 | **Captain / Integrator** | Schema lock, architecture, merge, CI gate, decision freeze |
| 1 | **Core Platform** | BasePatternDetector, PatternRegistry, ReversalPatternEngine, FeatureStore, shared indicators |
| 2 | **Liquidity Sweep Refactor** | Chuyển code hiện tại thành plugin tuân thủ BasePatternDetector |
| 3 | **Classic Reversal Detectors** | Head & Shoulders, Inverse H&S, Double Top/Bottom, Triple |
| 4 | **Wedge & Cup Detectors** | Rising/Falling Wedge, Cup & Handle |
| 5 | **Labeling & Scoring Shared** | Triple barrier, MFE/MAE, costs, rule_score generic, dataset pipeline |
| 6 | **GUI & Live Integration** | Cập nhật Onboarding, Signal Engine, Model Registry, PendingSignal |
| 7 | **QA / Bias Auditor** | No-lookahead tests, schema validation, integration tests, gate enforcement |

### 7.2. Task Breakdown chi tiết

#### Agent 0 — Captain / Integrator
- [ ] Định nghĩa & lock `docs/SCHEMAS.md` (PatternEvent + feature schema chung).
- [ ] Định nghĩa `docs/INTERFACES.md` (BasePatternDetector contract).
- [ ] Tạo skeleton repo `research/core/` + `research/patterns/`.
- [ ] CI gate mở rộng (phải chạy được với nhiều pattern).
- [ ] Quy tắc merge: không merge nếu schema thay đổi mà chưa update tất cả detector.
- [ ] Final integration test: chạy multi-pattern trên XAUUSD + BTCUSD sample.

#### Agent 1 — Core Platform
- [ ] Implement `BasePatternDetector` + `PatternEvent` dataclass.
- [ ] `PatternRegistry` (register / get / list, support entry-point hoặc folder scan).
- [ ] `ReversalPatternEngine` (run nhiều pattern, collect events, basic report).
- [ ] Shared causal indicators (ATR, swing, volume ratio, trend).
- [ ] FeatureStore (lưu events + attributes → parquet).
- [ ] Unit tests cho registry & engine.

#### Agent 2 — Liquidity Sweep Refactor
- [ ] Refactor toàn bộ code hiện tại (`events/`, `liquidity/`, …) thành `LiquiditySweepDetector`.
- [ ] Đảm bảo output đúng `PatternEvent` schema.
- [ ] Giữ nguyên behaviour (không thay đổi logic detection).
- [ ] Regression test: so sánh events cũ vs mới trên cùng data (phải gần như identical).
- [ ] Cập nhật config baseline.yaml cho pattern này.

#### Agent 3 — Classic Reversal Detectors
- [ ] `HeadShouldersDetector` + `InverseHeadShouldersDetector`.
- [ ] `DoubleTopDetector` + `DoubleBottomDetector`.
- [ ] `TripleTopBottomDetector`.
- [ ] Mỗi detector có `get_default_config()`, `get_feature_schema()`, unit test với synthetic data.
- [ ] Tài liệu ngắn trong `PATTERN_SPECS.md` mô tả rule detection causal.

#### Agent 4 — Wedge & Cup Detectors
- [ ] `RisingWedgeDetector` + `FallingWedgeDetector`.
- [ ] `CupAndHandleDetector`.
- [ ] Xử lý đặc thù: trendline fitting causal, minimum bars, volume behaviour.
- [ ] Unit test + synthetic examples.

#### Agent 5 — Labeling & Scoring Shared
- [ ] Generic labeling module nhận `List[PatternEvent]` → thêm entry/stop/target/MFE/MAE/outcome/costs.
- [ ] Rule score framework (có thể per-pattern weights).
- [ ] Dataset builder tương thích với modeling hiện tại.
- [ ] Walk-forward + calibration pipeline generic (nhận pattern_name).
- [ ] Đảm bảo `same_bar_policy`, embargo, purge vẫn hoạt động.

#### Agent 6 — GUI & Live Integration
- [ ] Cập nhật Symbol Onboarding UI: multi-select pattern + model per pattern.
- [ ] Persist cấu hình mới (JSON/YAML).
- [ ] Sửa `signal_engine_v2.py` để load nhiều detector + model.
- [ ] Cập nhật Model Registry schema + loader.
- [ ] Pending Signals hiển thị pattern_name.
- [ ] Backward compatible: symbol cũ chỉ có liquidity_sweep vẫn chạy bình thường.

#### Agent 7 — QA / Bias Auditor
- [ ] Bộ test `no_lookahead` cho mọi detector mới.
- [ ] Schema validation tests.
- [ ] Integration test end-to-end (data → events → dataset → train → inference).
- [ ] Kiểm tra regression Liquidity Sweep.
- [ ] Review feature schema của từng pattern (uses_future_data phải False).
- [ ] Báo cáo audit cuối cùng trước khi freeze.

### 7.3. Thứ tự thực hiện khuyến nghị
1. Agent 0 + Agent 1 (core contract) → freeze schema.
2. Agent 2 (refactor Liquidity Sweep) → chứng minh architecture hoạt động.
3. Agent 5 (labeling shared) song song.
4. Agent 3 + Agent 4 (thêm pattern).
5. Agent 6 (GUI + live).
6. Agent 7 audit toàn bộ → Captain merge & freeze.

### 7.4. Definition of Done (toàn hệ thống)
- [ ] Liquidity Sweep chạy qua plugin architecture và cho kết quả tương đương bản cũ.
- [ ] Ít nhất 4 pattern đảo chiều khác đã implement + unit test + feature schema.
- [ ] Có thể train model riêng cho từng pattern và đăng ký vào Model Registry.
- [ ] GUI Onboarding gắn được nhiều pattern + model tương ứng.
- [ ] Live engine phát signal từ nhiều pattern trên cùng symbol.
- [ ] Multi-backtest CLI hoạt động và xuất báo cáo so sánh.
- [ ] Toàn bộ test (unit + no_lookahead + integration) pass.
- [ ] Tài liệu SCHEMAS.md + INTERFACES.md + PATTERN_SPECS.md đầy đủ.

---

## 8. Rủi ro & Biện pháp giảm thiểu

| Rủi ro | Giảm thiểu |
|--------|------------|
| Look-ahead khi thêm pattern mới | Bắt buộc Agent 7 review + CI marker `no_lookahead` |
| Feature schema drift giữa train & live | Lock schema version trong model_id + runtime check |
| Quá nhiều false signal khi multi-pattern | Dedup theo time window + priority score + model_prob threshold |
| Phá vỡ Liquidity Sweep hiện tại | Regression test bắt buộc trước mọi merge |
| GUI phức tạp | Giữ UX đơn giản: multi-select + dropdown model rõ ràng |

---

## 9. Kết luận & Next Action

Hệ thống mới biến **Liquidity Sweep** thành một công dân bình đẳng trong một **Pattern Plugin Ecosystem**. Mọi pattern đảo chiều đều đi qua cùng một pipeline research → train → registry → live, đảm bảo tính liền mạch, logic cao và khả năng mở rộng dài hạn của dự án.

**Hành động ngay:**
1. Captain tạo branch `feature/multi-pattern-engine`.
2. Freeze `PatternEvent` + `BasePatternDetector` schema.
3. Bắt đầu Agent 1 + Agent 2 song song.

File này là nguồn sự thật duy nhất (single source of truth) cho toàn bộ team agents.

---

*Tài liệu này được tổng hợp từ kiến trúc hiện tại của `trading_lid_060826`, `huong_dan.md` của liquidity-sweep, và yêu cầu chuẩn hoá multi-pattern + GUI onboarding + ML continuity.*
