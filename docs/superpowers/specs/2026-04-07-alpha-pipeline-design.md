# Alpha Pipeline 設計規格

## 目標

將 `VectorizedBacktester` 的 signal → weight 邏輯重構為三階段 `AlphaPipeline`：

1. **Normalize** — 將原始 alpha 映射到已知範圍
2. **Allocate** — 將標準化信號轉為滿足特定不變量的權重
3. **Constrain + Renormalize** — 套用約束後恢復不變量

這是 0.4.0 的 breaking change，不提供向後相容。

---

## 動機

- **使用者體驗**：現行 API 要求使用者自行在外部做 `cs_rank()` 等前處理，且 `neutralization` 參數的語義不夠直觀
- **擴展性**：新增 allocation 策略（如 TopN）需要在 `_calculate_weights()` 裡加 if-else，職責混雜

---

## 架構

### 資料流

```
raw signal (任意範圍)
  │
  ▼  Normalizer.normalize()
normalized signal (已知定義域)
  │
  ▼  WeightAllocator.allocate()
target weights (滿足不變量)
  │
  ▼  WeightConstraint.apply() × N
constrained weights (可能破壞不變量)
  │
  ▼  WeightAllocator.renormalize()  ← 只在有 constraints 時執行
final weights
  │
  ▼  VectorizedBacktester
backtest results
```

### 檔案結構

```
backtest/
├── normalizers.py      # Normalizer ABC + 4 implementations
├── allocators.py       # WeightAllocator ABC + 3 implementations
├── pipeline.py         # AlphaPipeline
├── constraints.py      # 不變
├── vectorized.py       # 修改：改接 pipeline
├── utils.py            # 修改：刪除 polars 版 neutralize/renormalize
└── ...其餘不變
```

---

## 元件設計

### Normalizer (`backtest/normalizers.py`)

```python
class Normalizer(ABC):
    @abstractmethod
    def normalize(self, df: pl.DataFrame, signal_col: str, group_col: str) -> pl.DataFrame:
        """原地替換 signal_col 為標準化後的值。"""
        ...
```

| Class | 輸出範圍 | 邏輯 | 邊界處理 |
|-------|---------|------|---------|
| `RawNormalizer` | 不變 | pass-through | 無 |
| `RankNormalizer` | [0, 1] | cross-sectional `rank / count` per group | 無 |
| `ZScoreNormalizer` | ≈[-3, 3] | `(x - mean) / std` per group | std=0 → `null` |
| `MinMaxNormalizer` | [0, 1] | `(x - min) / (max - min)` per group | range=0 → `null` |

設計決策：
- `normalize()` 直接覆寫 `signal_col`，不產生新欄位，讓下游 Allocator 不需要知道上游用了哪個 Normalizer
- 邊界情況輸出 `null` 而非 0，由 Allocator 統一處理

### WeightAllocator (`backtest/allocators.py`)

```python
class WeightAllocator(ABC):
    @abstractmethod
    def allocate(self, df: pl.DataFrame, signal_col: str, group_col: str) -> pl.DataFrame:
        """新增 'weight' 欄位，滿足該 allocator 的不變量。signal 為 null 的行得到 0.0。"""
        ...

    @abstractmethod
    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        """constraints 之後恢復不變量。"""
        ...
```

| Class | 不變量 | allocate 邏輯 | renormalize 邏輯 |
|-------|--------|--------------|-----------------|
| `MarketNeutralAllocator` | Σw=0, Σ\|w\|=1 | demean → scale by abs sum | demean → scale by abs sum |
| `LongOnlyAllocator` | Σw=1, w≥0 | 只取正值信號 → scale to sum=1 | clip negatives → scale to sum=1 |
| `TopNAllocator(n, long_short=False)` | equal-weight | rank descending，top N 得 +1/N；`long_short=True` 時 bottom N 得 -1/N | 重新將非零權重恢復 equal-weight |

設計決策：
- `LongOnlyAllocator` 只取正值信號，不做 min-shift。負信號代表「不看好」，不應持倉
- `allocate()` 對 signal null 的行輸出 `weight = 0.0`

### AlphaPipeline (`backtest/pipeline.py`)

```python
class AlphaPipeline:
    def __init__(
        self,
        normalizer: Normalizer = RawNormalizer(),
        allocator: WeightAllocator = MarketNeutralAllocator(),
        constraints: list[WeightConstraint] | None = None,
    ):
        self.normalizer = normalizer
        self.allocator = allocator
        self.constraints = constraints or []

    def transform(
        self, df: pl.DataFrame, signal_col: str, group_col: str = "end_time"
    ) -> pl.DataFrame:
        # Stage 1: Normalize
        df = self.normalizer.normalize(df, signal_col, group_col)
        # Stage 2: Allocate
        df = self.allocator.allocate(df, signal_col, group_col)
        # Stage 3: Constrain + renormalize
        for constraint in self.constraints:
            df = constraint.apply(df)
        if self.constraints:
            df = self.allocator.renormalize(df, group_col)
        return df
```

預設行為 `AlphaPipeline()` = `RawNormalizer() + MarketNeutralAllocator()`，等同現行 `neutralization="market"` 的語義。

---

## VectorizedBacktester 修改

### 移除的參數

- `neutralization: Literal["market", "none"]` — 刪除
- `constraints: list | None` — 刪除（移入 Pipeline）

### 新增的參數

- `pipeline: AlphaPipeline | None = None` — 預設 `AlphaPipeline()`

### `_calculate_weights()` 簡化

修改前的 `_calculate_weights()` 包含 neutralization 分支、constraints 迴圈、renormalize 呼叫。修改後簡化為：

1. Mask 處理（保留在 Backtester，不進 Pipeline）
2. 呼叫 `self.pipeline.transform(df, signal_col, group_col)`
3. 清理 mask 臨時欄位

---

## utils.py 變更

| 函數 | 處置 | 原因 |
|------|------|------|
| `neutralize_weights_polars()` | 刪除 | 邏輯移入 `MarketNeutralAllocator.allocate()` |
| `renormalize_weights()` | 刪除 | 邏輯拆入各 Allocator 的 `renormalize()` |
| `neutralize_weights()` (pandas) | 保留 | 可能被 Portfolio 等其他模組使用 |
| `normalize_weights()` (pandas) | 保留 | 同上 |

---

## constraints.py 變更

無。`WeightConstraint` ABC 和所有現有 constraints（`MaxPositionConstraint`, `LongOnlyConstraint`, `MaxGrossExposureConstraint`, `MarketNeutralConstraint`）完全不動。

---

## __init__.py 更新

新增 exports：
- `Normalizer`, `RawNormalizer`, `RankNormalizer`, `ZScoreNormalizer`, `MinMaxNormalizer`
- `WeightAllocator`, `MarketNeutralAllocator`, `LongOnlyAllocator`, `TopNAllocator`
- `AlphaPipeline`

移除 exports：
- `neutralize_weights_polars`（如果有）
- `renormalize_weights`（如果有）

---

## 不在範圍內

- **Mask/Universe 篩選**：留在 `_calculate_weights()` 中，Pipeline 之前執行。未來可考慮移入 Pipeline 作為 Stage 0
- **動態風控**（turnover limit、drawdown stop、volatility targeting）：不放入 Pipeline。Pipeline 只負責靜態截面轉換。未來在 Pipeline 和 Backtester 之間插入新層
- **Factor 耦合**：Pipeline 在 DataFrame 層級獨立實作 rank/zscore，不依賴 Factor 類別

---

## 使用範例

```python
# 預設（等同現行 neutralization="market"）
bt = VectorizedBacktester(prices, signal)

# Long-only + rank normalization + position cap
pipeline = AlphaPipeline(
    normalizer=RankNormalizer(),
    allocator=LongOnlyAllocator(),
    constraints=[MaxPositionConstraint(0.1)],
)
bt = VectorizedBacktester(prices, signal, pipeline=pipeline)

# Market-neutral + z-score
pipeline = AlphaPipeline(
    normalizer=ZScoreNormalizer(),
    allocator=MarketNeutralAllocator(),
)
bt = VectorizedBacktester(prices, signal, pipeline=pipeline)

# Top 10 equal-weight long-short
pipeline = AlphaPipeline(
    normalizer=RankNormalizer(),
    allocator=TopNAllocator(n=10, long_short=True),
)
bt = VectorizedBacktester(prices, signal, pipeline=pipeline)

# 原始信號直接 market-neutral
pipeline = AlphaPipeline(
    normalizer=RawNormalizer(),
    allocator=MarketNeutralAllocator(),
)
bt = VectorizedBacktester(prices, signal, pipeline=pipeline)
```
