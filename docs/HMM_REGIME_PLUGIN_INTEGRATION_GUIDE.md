# HMM Regime Plugin — Hướng dẫn tích hợp chặt chẽ

**Phiên bản:** 1.0  
**Ngày:** 2026-09-08  
**Mục tiêu:** Tích hợp Hidden Markov Model (HMM) vào hệ thống Multi-Pattern Plugin Engine như một **plugin độc lập**, tuân thủ tuyệt đối nguyên tắc no-lookahead, và cho phép **plug / unplug** feature mà không phá vỡ model đã train.

---

## 1. Vai trò của HMM trong kiến trúc

HMM đóng vai trò **Environment / Regime Engine**, không phải pattern detector.

- **Không** cạnh tranh với các `BasePatternDetector`.
- **Không** phụ thuộc vào pattern cụ thể.
- Chỉ quan sát dữ liệu thị trường tổng quát (returns, volatility, volume…) và trả về trạng thái chế độ thị trường.

Hai cách sử dụng chính (có thể dùng đồng thời):

| Cách dùng              | Mô tả                                                                 | Ảnh hưởng đến model đã train |
|------------------------|-----------------------------------------------------------------------|------------------------------|
| **Feature Emitter**    | Inject `hmm_state`, `hmm_state_prob_*` vào `PatternEvent.attributes` | Có thể plug/unplug an toàn  |
| **Hard Gate / Filter** | Chặn hoặc cho phép pattern chạy theo regime                          | Không ảnh hưởng model       |

---

## 2. Nguyên tắc bắt buộc (No Look-ahead)

Mọi implementation phải tuân thủ:

1. **Causal only**  
   Tại thời điểm bar `t` đóng cửa, HMM chỉ được sử dụng dữ liệu ≤ `t`.

2. **Không fit trên toàn bộ series rồi map ngược**  
   Không được chạy HMM trên full history rồi gán state cho quá khứ.

3. **State chỉ được biết sau khi bar đóng**  
   `hmm_known_at = bar_close_time`.

4. **Độ trễ phải được khai báo rõ**  
   Nếu dùng smoothing / posterior decoding có lag, phải ghi rõ số bar lag và test ảnh hưởng.

5. **Reproducible**  
   Phải lưu: config HMM, seed, data_version, model_version, transition matrix (nếu cần).

6. **Feature schema versioning**  
   Mọi feature HMM phải có `feature_schema_version` riêng hoặc nằm trong schema chung có version.

---

## 3. Thiết kế Plugin

### 3.1. Interface đề xuất

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

@dataclass
class RegimeState:
    timestamp: pd.Timestamp          # bar close time (known_at)
    state: int                       # 0, 1, 2, ...
    state_name: str                  # "trending", "sideways", "high_vol"
    state_prob: Dict[str, float]     # {"trending": 0.82, "sideways": 0.15, ...}
    confidence: float                # max probability
    lag_bars: int                    # độ trễ đã áp dụng
    model_version: str
    config_hash: str


class BaseRegimePlugin(ABC):
    """Plugin HMM / Regime Filter."""
    
    name: str = "hmm_regime"
    version: str = "1.0.0"
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, config: dict) -> None:
        """Train HMM trên dữ liệu lịch sử (chỉ dùng cho research)."""
        ...
    
    @abstractmethod
    def predict(self, df: pd.DataFrame, config: dict) -> List[RegimeState]:
        """
        Causal inference.
        df phải đã sort theo timestamp tăng dần.
        Chỉ trả về state cho các bar đã đóng.
        """
        ...
    
    @abstractmethod
    def get_feature_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Schema của các feature sẽ inject.
        Phải khai báo uses_future_data = False.
        """
        ...
    
    def get_default_config(self) -> dict:
        return {
            "n_states": 3,
            "features": ["log_return", "atr_norm", "volume_zscore"],
            "lag_bars": 0,
            "min_confidence": 0.55,
        }
```

### 3.2. Feature Schema chuẩn (khuyến nghị)

```json
{
  "hmm_state": {
    "dtype": "int64",
    "description": "Most likely regime state (0=trending, 1=sideways, 2=high_vol)",
    "available_at": "bar_close",
    "uses_future_data": false
  },
  "hmm_prob_trending": {
    "dtype": "float64",
    "description": "Posterior probability of trending regime",
    "available_at": "bar_close",
    "uses_future_data": false
  },
  "hmm_prob_sideways": {
    "dtype": "float64",
    "description": "Posterior probability of sideways regime",
    "available_at": "bar_close",
    "uses_future_data": false
  },
  "hmm_prob_high_vol": {
    "dtype": "float64",
    "description": "Posterior probability of high volatility regime",
    "available_at": "bar_close",
    "uses_future_data": false
  },
  "hmm_confidence": {
    "dtype": "float64",
    "description": "Max state probability",
    "available_at": "bar_close",
    "uses_future_data": false
  }
}
```

---

## 4. Cơ chế Plug / Unplug an toàn (quan trọng nhất)

Mục tiêu: **Có thể bật/tắt feature HMM mà không làm hỏng model đã train**.

### 4.1. Nguyên tắc thiết kế Feature List

- Mọi model đã train phải lưu **exact feature list** + `feature_schema_version` trong artifact.
- Khi inference, engine chỉ đưa đúng các feature mà model đó được train.
- Feature HMM được đánh dấu là **optional**.

### 4.2. Cách triển khai

#### A. Trong training

```python
# config example
features:
  core:
    - atr
    - volume_ratio
    - rsi
    - ...
  optional:
    hmm:
      enabled: true          # có thể set false
      features:
        - hmm_state
        - hmm_prob_trending
        - hmm_prob_sideways
        - hmm_prob_high_vol
        - hmm_confidence
```

- Khi `hmm.enabled = false` → không đưa các cột HMM vào dataset.
- Model được train và lưu kèm `feature_list` thực tế đã dùng.

#### B. Trong inference / live

```python
def build_feature_vector(event: PatternEvent, model_meta: dict) -> np.ndarray:
    required_features = model_meta["feature_list"]   # list chính xác lúc train
    
    vector = []
    for feat in required_features:
        if feat.startswith("hmm_"):
            # Nếu model cũ không có HMM → sẽ không bao giờ vào đây
            value = event.attributes.get(feat, np.nan)
        else:
            value = event.attributes.get(feat, np.nan)
        vector.append(value)
    
    return np.array(vector)
```

#### C. Quy tắc vàng

- **Model cũ (train trước khi có HMM)** → `feature_list` không chứa `hmm_*` → inference vẫn chạy bình thường.
- **Model mới (train có HMM)** → `feature_list` có `hmm_*` → bắt buộc phải có HMM plugin đang active và tính được feature.
- Không bao giờ “tự ý” thêm hoặc bớt feature so với lúc train.

### 4.3. Model Registry metadata bắt buộc

```yaml
model_id: "btcusd_double_bottom_h1_v3"
pattern_name: "double_bottom"
feature_schema_version: "1.3.0"
feature_list:
  - atr
  - volume_ratio
  - rsi_14
  - hmm_state
  - hmm_prob_trending
  - hmm_prob_sideways
  - hmm_confidence
optional_plugins:
  - name: "hmm_regime"
    version: "1.0.0"
    required: true          # true nếu model này phụ thuộc HMM
```

---

## 5. Luồng tích hợp vào Pipeline

```
1. Data Audit (giữ nguyên)
2. HMM Plugin (chạy riêng, causal)
   → sinh RegimeState cho mọi bar
   → lưu vào FeatureStore (parquet)
3. Pattern Detectors chạy
4. Feature Pipeline
   - Nếu config.hmm.enabled = true → merge hmm features vào event.attributes
   - Nếu false → bỏ qua
5. Labeling (không đổi)
6. Train
   - Chỉ dùng feature list theo config hiện tại
   - Lưu feature_list vào artifact
7. Live
   - Load model → đọc feature_list của model đó
   - Chỉ tính và inject đúng những feature cần thiết
```

---

## 6. Cấu hình mẫu

```yaml
# configs/plugins/hmm_regime.yaml
plugin:
  name: hmm_regime
  version: "1.0.0"
  enabled: true

hmm:
  n_states: 3
  state_names: ["trending", "sideways", "high_vol"]
  input_features:
    - log_return_1
    - atr_14_norm
    - volume_zscore_20
  lag_bars: 0
  min_confidence: 0.50
  covariance_type: "full"
  n_iter: 100
  random_state: 42

# Trong pattern config
features:
  optional_plugins:
    - hmm_regime          # chỉ cần liệt kê tên plugin
```

---

## 7. Checklist No-Lookahead riêng cho HMM

- [ ] `predict()` chỉ dùng dữ liệu ≤ bar hiện tại
- [ ] Không chạy HMM trên full history rồi gán state ngược
- [ ] `RegimeState.timestamp` = thời điểm bar đóng cửa
- [ ] Có test incremental (chạy từng bar một vs chạy full) → kết quả giống nhau
- [ ] Có test synthetic future-data injection → phải bị chặn
- [ ] Lag (nếu có) được khai báo và test ảnh hưởng đến performance
- [ ] Feature schema khai báo `uses_future_data: false`
- [ ] Model artifact lưu đúng `feature_list` thực tế đã dùng

---

## 8. Hard Gate (tùy chọn)

Ngoài việc inject feature, có thể dùng HMM như cổng lọc:

```yaml
regime_filter:
  enabled: true
  rules:
    - pattern: "double_bottom"
      allowed_states: ["trending", "sideways"]
      min_confidence: 0.60
    - pattern: "head_shoulders"
      allowed_states: ["trending"]
      min_confidence: 0.70
```

Logic:

```python
def is_allowed(event: PatternEvent, regime: RegimeState, rules: dict) -> bool:
    rule = rules.get(event.pattern_name)
    if not rule:
        return True
    if regime.state_name not in rule["allowed_states"]:
        return False
    if regime.confidence < rule["min_confidence"]:
        return False
    return True
```

Hard Gate **không** ảnh hưởng đến model đã train vì nó nằm ngoài feature vector.

---

## 9. Thứ tự triển khai khuyến nghị

1. Implement `BaseRegimePlugin` + một HMM concrete (3 states).
2. Viết unit test no-lookahead + incremental test.
3. Chạy offline trên data lịch sử, lưu RegimeState.
4. Thêm config `optional_plugins`.
5. Sửa Feature Pipeline để support plug/unplug.
6. Train một model mới **có** HMM và một model **không** có HMM → so sánh OOS.
7. Cập nhật Model Registry schema.
8. Cập nhật Live Engine để đọc `feature_list` từ model metadata.
9. Agent 7 audit toàn bộ.

---

## 10. Tóm tắt quy tắc vàng

> - HMM là plugin độc lập.  
> - Feature HMM là **optional**.  
> - Model chỉ nhìn đúng feature list lúc nó được train.  
> - Plug/unplug không được làm thay đổi hành vi của model cũ.  
> - Mọi thứ phải causal và reproducible.

---

*Tài liệu này là phần bổ sung cho `REVERSAL_PATTERN_ENGINE_SPEC.md` và `LEAKAGE_CHECKLIST_MULTI_PATTERN.md`.*
