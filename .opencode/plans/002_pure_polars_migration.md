# Pure Polars Migration Plan

## Context

### Original Request
將因子分析引擎從 Pandas 完全遷移到 Polars，以解決：
- 記憶體不足問題
- 大量資料處理時 Pandas 運行緩慢

### 現有狀態
由於 `git restore` 操作，以下檔案被還原到舊版本：

| 檔案 | 狀態 |
|-----|------|
| `src/factorium/factors/base.py` | 混合狀態（有 `_to_polars()` 但仍用 pd.DataFrame） |
| `src/factorium/factors/mixins/ts_ops.py` | 部分遷移（部分 ops 已委派到 engine） |
| `src/factorium/factors/mixins/cs_ops.py` | 部分遷移 |
| `src/factorium/factors/mixins/math_ops.py` | 未遷移 |

以下檔案已存在且完整（未追蹤）：
- `src/factorium/factors/engine/__init__.py` - 匯出 PolarsEngine 為預設
- `src/factorium/factors/engine/protocol.py` - ComputeEngine Protocol (~30 methods)
- `src/factorium/factors/engine/polars.py` - Polars 實作 (558 行)
- `src/factorium/factors/engine/pandas.py` - Pandas 實作 (550 行)
- `tests/factors/test_engine_consistency.py` - 雙後端一致性測試

### User Decisions
- **API**: 完全 Polars API（`factor.data` 返回 `pl.DataFrame`）
- **計算策略**: LazyFrame + 延遲執行
- **遷移範圍**: 全部遷移（包括 base.py 中的基礎算術運算）
- **開發方法**: TDD（Test-Driven Development）

---

## Work Objectives

### Core Objective
將 Factor 類別從 Pandas 完全遷移到 Polars：
1. 內部儲存使用 `pl.LazyFrame`
2. 運算鏈式構建 LazyFrame，延遲執行
3. `factor.data` 返回 `pl.DataFrame`（breaking change）
4. 提供 `factor.to_pandas()` 方法以維持下游相容性

### Concrete Deliverables
| 檔案 | 變更類型 | 說明 |
|-----|---------|------|
| `base.py` | 重構 | `_lf: pl.LazyFrame`，修改所有屬性和方法 |
| `mixins/ts_ops.py` | 重構 | 所有方法改為返回 LazyFrame 鏈 |
| `mixins/cs_ops.py` | 重構 | 所有方法改為返回 LazyFrame 鏈 |
| `mixins/math_ops.py` | 重構 | 所有方法改為返回 LazyFrame 鏈 |
| `analyzer.py` | 適配 | 使用 `factor.to_pandas()` |
| `backtest/backtester.py` | 適配 | 使用 `factor.to_pandas()` |
| `tests/` | 更新 | 更新測試以符合新 API |

### Definition of Done
- [x] `uv run pytest` - 所有測試通過 (478/478)
- [x] `Factor._lf` 類型為 `pl.LazyFrame`
- [x] `factor.data` 返回 `pl.DataFrame`
- [x] `factor.to_pandas()` 返回 `pd.DataFrame`
- [x] 運算鏈不會立即執行（驗證 LazyFrame 延遲特性）
- [x] 數值精度與 Pandas 版本一致 (rtol=1e-9, atol=1e-12)
- [x] 已清理所有 Pandas 殘留程式碼 (Task 2.3 完成)
- [x] 已移除未使用的 helper 方法 (Task 1.1 完成)
- [x] **純化完成**: mixins 中僅 `ts_quantile` 保留 NumPy (因 scipy PPF 依賴)

### Must NOT Have (Guardrails)
- **MUST NOT** 修改 `data/` 模組（資料下載/載入）
- **MUST NOT** 修改 `bar.py`、`aggbar.py`
- **MUST NOT** 引入新的外部依賴
- **MUST NOT** 刪除 PandasEngine（保留作為後備和驗證）
- **MUST NOT** 犧牲數值精度換取效能

---

## Architecture Design

### 新的 Factor 類別結構

```python
import polars as pl
import pandas as pd
from typing import Union, Optional
from abc import ABC

class BaseFactor(ABC):
    _lf: pl.LazyFrame  # 內部儲存（LazyFrame）
    _name: str
    
    def __init__(
        self, 
        data: Union["AggBar", pd.DataFrame, pl.DataFrame, pl.LazyFrame, Path], 
        name: Optional[str] = None
    ):
        self._name = name or "factor"
        self._lf = self._to_lazy(data)
    
    def _to_lazy(self, data) -> pl.LazyFrame:
        """Convert any input to LazyFrame with standard schema."""
        if isinstance(data, pl.LazyFrame):
            lf = data
        elif isinstance(data, pl.DataFrame):
            lf = data.lazy()
        elif isinstance(data, pd.DataFrame):
            lf = pl.from_pandas(data).lazy()
        elif isinstance(data, Path):
            if data.suffix == ".parquet":
                lf = pl.scan_parquet(data)
            elif data.suffix == ".csv":
                lf = pl.scan_csv(data)
            else:
                raise ValueError(f"Unsupported file format: {data.suffix}")
        elif hasattr(data, "to_df"):  # AggBar
            lf = pl.from_pandas(data.to_df()).lazy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Ensure standard schema: start_time, end_time, symbol, factor
        return self._normalize_schema(lf)
    
    @property
    def data(self) -> pl.DataFrame:
        """Collect and return as Polars DataFrame."""
        return self._lf.collect()
    
    def to_pandas(self) -> pd.DataFrame:
        """For backward compatibility with analyzer/backtest."""
        return self._lf.collect().to_pandas()
    
    @property
    def lazy(self) -> pl.LazyFrame:
        """Access underlying LazyFrame for advanced users."""
        return self._lf
    
    @property
    def name(self) -> str:
        return self._name
```

### 運算子實作模式（TDD）

#### Time-Series Operations
```python
def ts_mean(self, window: int) -> Self:
    """Rolling mean over symbol groups with strict NaN handling."""
    self._validate_window(window)
    new_lf = self._lf.with_columns(
        pl.col("factor")
          .rolling_mean(window_size=window, min_samples=window)
          .over("symbol")
          .alias("factor")
    )
    return self.__class__(new_lf, f"ts_mean({self.name},{window})")
```

#### Cross-Sectional Operations
```python
def cs_rank(self) -> Self:
    """Cross-sectional rank with strict NaN propagation."""
    new_lf = self._lf.with_columns(
        pl.when(pl.col("factor").is_null().any().over("end_time"))
          .then(None)
          .otherwise(
              pl.col("factor").rank(method="min").over("end_time")
              / pl.col("factor").count().over("end_time")
          )
          .alias("factor")
    )
    return self.__class__(new_lf, f"cs_rank({self.name})")
```

#### Binary Operations
```python
def _binary_op(self, other: Union["BaseFactor", float], op_expr, op_name: str) -> Self:
    if isinstance(other, self.__class__):
        # Join on time/symbol, then apply operation
        joined = self._lf.join(
            other._lf.select([
                "start_time", "end_time", "symbol", 
                pl.col("factor").alias("other")
            ]),
            on=["start_time", "end_time", "symbol"],
            how="inner"
        )
        new_lf = joined.with_columns(
            op_expr(pl.col("factor"), pl.col("other")).alias("factor")
        ).drop("other")
        return self.__class__(new_lf, f"({self.name}{op_name}{other.name})")
    else:
        new_lf = self._lf.with_columns(
            op_expr(pl.col("factor"), pl.lit(other)).alias("factor")
        )
        return self.__class__(new_lf, f"({self.name}{op_name}{other})")
```

### 數值精度標準

金融計算需要高精度，採用以下容差標準：

| 運算類型 | rtol | atol | 說明 |
|---------|------|------|------|
| 基本算術 | 1e-14 | 1e-14 | 加減乘除 |
| Rolling stats | 1e-10 | 1e-12 | ts_mean, ts_std 等 |
| Ranking | 1e-9 | 1e-12 | cs_rank, ts_rank |
| 複雜統計 | 1e-9 | 1e-12 | skewness, kurtosis |
| 迴歸 | 1e-8 | 1e-10 | ts_beta, cs_neutralize |

---

## TDD Development Process

每個 Task 遵循 **RED-GREEN-REFACTOR** 循環：

1. **RED**: 先寫失敗的測試
   - 測試預期行為
   - 測試邊界條件（NaN、空資料、單一值）
   - 測試數值精度（與 Pandas 對比）

2. **GREEN**: 實作最小可行程式碼使測試通過

3. **REFACTOR**: 優化程式碼，保持測試通過

### 測試檔案結構

```
tests/
├── factors/
│   ├── test_base_polars.py      # Task 1.1, 1.2 測試
│   ├── test_ts_ops_polars.py    # Task 2.1 測試
│   ├── test_cs_ops_polars.py    # Task 2.2 測試
│   ├── test_math_ops_polars.py  # Task 2.3 測試
│   └── test_engine_consistency.py  # 已存在，保留
├── test_analyzer_compat.py      # Task 3.1 測試
└── backtest/
    └── test_backtester_compat.py  # Task 3.2 測試
```

---

## Task Flow

```
Phase 1: 基礎架構
    Task 1.1 (base.py 重構) ─── TDD
         ↓
    Task 1.2 (binary ops 遷移) ─── TDD
         ↓
Phase 2: 運算子遷移（可平行，使用 subagent）
    ┌─────────────┬─────────────┬─────────────┐
    │  Task 2.1   │  Task 2.2   │  Task 2.3   │
    │  ts_ops.py  │  cs_ops.py  │ math_ops.py │
    │    TDD      │    TDD      │    TDD      │
    └─────────────┴─────────────┴─────────────┘
         ↓
Phase 3: 下游適配（可平行，使用 subagent）
    ┌─────────────┬─────────────┐
    │  Task 3.1   │  Task 3.2   │
    │ analyzer.py │backtester.py│
    └─────────────┴─────────────┘
         ↓
Phase 4: 整合測試與驗證
    Task 4.1 (整合測試)
         ↓
    Task 4.2 (效能驗證)
```

---

## TODOs

### Phase 1: 基礎架構

- [ ] 1.1 重構 base.py - 核心類別遷移到 LazyFrame

  **TDD - RED Phase**:
  先在 `tests/factors/test_base_polars.py` 撰寫以下測試：
  ```python
  class TestBaseFactorPolars:
      def test_init_from_pandas_dataframe(self):
          """Factor can be initialized from pd.DataFrame."""
          
      def test_init_from_polars_dataframe(self):
          """Factor can be initialized from pl.DataFrame."""
          
      def test_init_from_polars_lazyframe(self):
          """Factor can be initialized from pl.LazyFrame."""
          
      def test_init_from_parquet_path(self):
          """Factor can be initialized from Path to parquet."""
          
      def test_data_returns_polars_dataframe(self):
          """factor.data returns pl.DataFrame."""
          
      def test_to_pandas_returns_pandas_dataframe(self):
          """factor.to_pandas() returns pd.DataFrame."""
          
      def test_lazy_returns_lazyframe(self):
          """factor.lazy returns pl.LazyFrame."""
          
      def test_lazy_evaluation_not_collected(self):
          """Operations don't trigger collection until .data is accessed."""
          
      def test_schema_normalization(self):
          """Input with non-standard column names is normalized."""
  ```

  **TDD - GREEN Phase**:
  - 將 `_data: pd.DataFrame` 改為 `_lf: pl.LazyFrame`
  - 實作 `_to_lazy()` 方法處理各種輸入類型
  - 實作 `_normalize_schema()` 確保標準欄位
  - 新增 `data` 屬性返回 `pl.DataFrame`
  - 新增 `to_pandas()` 方法
  - 新增 `lazy` 屬性返回 `pl.LazyFrame`
  - 移除 `_to_polars()` 和 `_from_polars()` helper
  - 移除 `_apply_rolling()` helper

  **Must NOT do**:
  - 不要在此 Task 修改 `_binary_op`（Task 1.2 處理）
  - 不要在此 Task 修改 `_cs_op`（Task 2.2 處理）
  - 不要修改任何 mixin

  **Parallelizable**: NO（後續任務依賴）

  **References**:
  - `src/factorium/factors/base.py:1-219` - 現有實作
  - `src/factorium/factors/engine/polars.py` - from_pandas/to_pandas 參考

  **Acceptance Criteria**:
  - [ ] 所有 `test_base_polars.py` 測試通過
  - [ ] `isinstance(factor.data, pl.DataFrame)` → True
  - [ ] `isinstance(factor.to_pandas(), pd.DataFrame)` → True
  - [ ] `isinstance(factor.lazy, pl.LazyFrame)` → True

  **Commit**: `refactor(factor): migrate base.py to Polars LazyFrame`

---

- [ ] 1.2 遷移 binary operations - 基礎算術運算遷移

  **TDD - RED Phase**:
  在 `tests/factors/test_base_polars.py` 新增測試：
  ```python
  class TestBinaryOperationsPolars:
      def test_add_two_factors(self):
          """factor1 + factor2 returns correct result."""
          
      def test_add_factor_and_scalar(self):
          """factor + 1.0 returns correct result."""
          
      def test_sub_two_factors(self):
          """factor1 - factor2 returns correct result."""
          
      def test_mul_two_factors(self):
          """factor1 * factor2 returns correct result."""
          
      def test_div_two_factors_safe(self):
          """factor1 / factor2 handles division by zero."""
          
      def test_comparison_gt(self):
          """factor1 > factor2 returns 0/1 integers."""
          
      def test_binary_op_preserves_lazy(self):
          """Binary operations don't trigger collection."""
          
      def test_binary_op_numerical_precision(self):
          """Results match Pandas with rtol=1e-14."""
  ```

  **TDD - GREEN Phase**:
  - 重寫 `_binary_op` 使用 Polars join 和表達式
  - 重寫 `_comparison_op` 使用 Polars
  - 更新 `__add__`, `__sub__`, `__mul__`, `__truediv__`
  - 更新 `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`
  - 更新 `__lt__`, `__le__`, `__gt__`, `__ge__`, `__eq__`, `__ne__`
  - 更新 `__neg__`

  **Must NOT do**:
  - 不要使用 `pd.merge`（改用 `pl.LazyFrame.join`）

  **Parallelizable**: NO（Phase 2 依賴）

  **References**:
  - `src/factorium/factors/base.py:90-216` - 現有 _binary_op 實作
  - `src/factorium/factors/engine/polars.py:maximum()` - join 模式參考

  **Acceptance Criteria**:
  - [ ] 所有 `TestBinaryOperationsPolars` 測試通過
  - [ ] 運算結果與 Pandas 版本一致（rtol=1e-14）
  - [ ] 運算不會立即 collect（保持 lazy）

  **Commit**: `refactor(factor): migrate binary operations to Polars`

---

### Phase 2: 運算子遷移（可使用 subagent 平行開發）

- [ ] 2.1 遷移 ts_ops.py - 時間序列運算子

  **運算子清單** (25 個):
  | 運算子 | 現狀 | Polars 實作方式 |
  |-------|------|----------------|
  | `ts_sum` | 已委派 engine | `rolling_sum().over("symbol")` |
  | `ts_mean` | 已委派 engine | `rolling_mean().over("symbol")` |
  | `ts_std` | 已委派 engine | `rolling_std().over("symbol")` |
  | `ts_min` | 已委派 engine | `rolling_min().over("symbol")` |
  | `ts_max` | 已委派 engine | `rolling_max().over("symbol")` |
  | `ts_shift` | 已委派 engine | `shift().over("symbol")` |
  | `ts_delta` | 已委派 engine | `diff().over("symbol")` |
  | `ts_median` | Pandas | `rolling_median().over("symbol")` |
  | `ts_product` | Pandas | `map_batches` with cumulative product |
  | `ts_rank` | Pandas | Custom rolling rank |
  | `ts_argmin` | Pandas | `rolling_map` with argmin |
  | `ts_argmax` | Pandas | `rolling_map` with argmax |
  | `ts_scale` | 組合 | 依賴 ts_min, ts_max |
  | `ts_zscore` | 組合 | 依賴 ts_mean, ts_std |
  | `ts_quantile` | 組合 | 依賴 ts_rank + scipy.stats |
  | `ts_kurtosis` | Pandas | `map_batches` with kurtosis formula |
  | `ts_skewness` | 組合 | 依賴 ts_mean, ts_sum, pow |
  | `ts_step` | Pandas | `int_range().over("symbol")` |
  | `ts_cv` | Pandas | ts_std / ts_mean |
  | `ts_jumpiness` | 組合 | 依賴 ts_delta, ts_sum, ts_max, ts_min |
  | `ts_vr` | 組合 | Variance ratio |
  | `ts_beta` | 組合 | ts_cov / ts_var |
  | `ts_alpha` | 組合 | 依賴 ts_beta, ts_mean |
  | `ts_resid` | 組合 | 依賴 ts_alpha, ts_beta |
  | `ts_corr` | Pandas | Join + rolling_corr |
  | `ts_cov` | Pandas | Join + rolling_cov |
  | `ts_autocorr` | 組合 | 依賴 ts_shift, ts_corr |
  | `ts_reversal_count` | Pandas | `map_batches` |

  **TDD - RED Phase**:
  在 `tests/factors/test_ts_ops_polars.py` 為每個運算子撰寫測試：
  ```python
  class TestTsOpsPolars:
      @pytest.fixture
      def sample_factor(self):
          """Create sample factor with known values for testing."""
          
      def test_ts_mean_basic(self, sample_factor):
          """ts_mean computes correct rolling mean."""
          
      def test_ts_mean_nan_propagation(self, sample_factor):
          """ts_mean returns NaN when window contains NaN."""
          
      def test_ts_mean_numerical_precision(self, sample_factor):
          """ts_mean matches Pandas with rtol=1e-10."""
          
      # ... 為每個運算子撰寫類似測試
  ```

  **TDD - GREEN Phase**:
  依序實作每個運算子，確保測試通過。

  **Must NOT do**:
  - 不要使用 `self._data.copy()`
  - 不要使用 `pd.Series.rolling()`
  - 不要使用 `groupby("symbol")`（改用 `.over("symbol")`）

  **Parallelizable**: YES（與 2.2, 2.3 平行）

  **References**:
  - `src/factorium/factors/mixins/ts_ops.py:1-444` - 現有實作
  - `src/factorium/factors/engine/polars.py` - Polars rolling 實作參考
  - Polars docs: `rolling_*`, `shift`, `diff`, `rank` expressions

  **Acceptance Criteria**:
  - [ ] 所有 `test_ts_ops_polars.py` 測試通過
  - [ ] `grep "pd\." ts_ops.py` → 無結果（除 import 和 type hints）
  - [ ] `grep "np\." ts_ops.py` → 無結果（除必要的 nan 常數）
  - [ ] 數值精度符合標準

  **Commit**: `refactor(factor): migrate ts_ops.py to pure Polars`

---

- [ ] 2.2 遷移 cs_ops.py - 橫截面運算子

  **運算子清單** (7 個):
  | 運算子 | 現狀 | Polars 實作方式 |
  |-------|------|----------------|
  | `cs_rank` | 已委派 engine | `rank().over("end_time")` + NaN check |
  | `cs_zscore` | 已委派 engine | `(x - mean) / std` over end_time |
  | `cs_demean` | 已委派 engine | `x - mean().over("end_time")` |
  | `cs_winsorize` | Pandas | `clip()` with quantile over end_time |
  | `cs_neutralize` | Pandas | `map_batches` with lstsq |
  | `mean` | Pandas | `mean().over("end_time")` broadcast |
  | `median` | Pandas | `median().over("end_time")` broadcast |

  **TDD - RED Phase**:
  在 `tests/factors/test_cs_ops_polars.py` 撰寫測試。

  **Must NOT do**:
  - 不要使用 `groupby("end_time").transform()`
  - 不要使用 `self._cs_op()` helper

  **Parallelizable**: YES（與 2.1, 2.3 平行）

  **References**:
  - `src/factorium/factors/mixins/cs_ops.py:1-125` - 現有實作
  - `src/factorium/factors/engine/polars.py:cs_rank()` - Polars 實作參考

  **Acceptance Criteria**:
  - [ ] 所有 `test_cs_ops_polars.py` 測試通過
  - [ ] 嚴格 NaN 傳播語義保持不變
  - [ ] 數值精度符合標準

  **Commit**: `refactor(factor): migrate cs_ops.py to pure Polars`

---

- [ ] 2.3 遷移 math_ops.py - 數學運算子

  **運算子清單** (15 個):
  | 運算子 | Polars 實作方式 |
  |-------|----------------|
  | `abs` | `pl.col("factor").abs()` |
  | `sign` | `pl.col("factor").sign()` |
  | `inverse` | `pl.when(x != 0).then(1/x).otherwise(None)` |
  | `log` | `pl.col("factor").log(base)` |
  | `ln` | `pl.col("factor").log()` |
  | `sqrt` | `pl.col("factor").sqrt()` |
  | `signed_log1p` | `sign * log1p(abs)` |
  | `signed_pow` | `sign * pow(abs, exp)` |
  | `pow` | `pl.col("factor").pow(exp)` |
  | `where` | `pl.when().then().otherwise()` |
  | `max` | Join + `pl.max_horizontal` or scalar compare |
  | `min` | Join + `pl.min_horizontal` or scalar compare |
  | `add/sub/mul/div` | Alias to __add__ etc. |
  | `reverse` | Alias to __neg__ |

  **TDD - RED Phase**:
  在 `tests/factors/test_math_ops_polars.py` 撰寫測試。

  **Must NOT do**:
  - 不要使用 `np.where`, `np.sign`, `np.log` 等
  - 不要使用 `pd.merge`

  **Parallelizable**: YES（與 2.1, 2.2 平行）

  **References**:
  - `src/factorium/factors/mixins/math_ops.py:1-162` - 現有實作
  - `src/factorium/factors/engine/polars.py:where()` - Polars 實作參考

  **Acceptance Criteria**:
  - [ ] 所有 `test_math_ops_polars.py` 測試通過
  - [ ] `grep "np\." math_ops.py` → 無結果
  - [ ] 數值精度符合標準

  **Commit**: `refactor(factor): migrate math_ops.py to pure Polars`

---

### Phase 3: 下游適配（可使用 subagent 平行開發）

- [ ] 3.1 適配 analyzer.py - 因子分析器

  **What to do**:
  - 將所有 `self.factor.data` 改為 `self.factor.to_pandas()`
  - 保持 analyzer 內部邏輯不變（仍使用 Pandas）

  **關鍵修改點**:
  - `prepare_data()` - `self.factor.data.copy()` → `self.factor.to_pandas()`
  - `calculate_ic()` - 依賴 Pandas pivot
  - `calculate_quantile_returns()` - 依賴 Pandas groupby

  **TDD - RED Phase**:
  確保現有 `tests/factors/test_analyzer.py` 測試通過。

  **Must NOT do**:
  - 不要改變 analyzer 的公開 API
  - 不要改變輸出格式

  **Parallelizable**: YES（與 3.2 平行）

  **References**:
  - `src/factorium/factors/analyzer.py` - 現有實作

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/factors/test_analyzer.py -v` → PASS
  - [ ] `FactorAnalyzer` 公開 API 不變

  **Commit**: `fix(analyzer): adapt to Polars Factor API`

---

- [ ] 3.2 適配 backtester.py - 回測引擎

  **What to do**:
  - 將所有 `self.signal.data` 改為 `self.signal.to_pandas()`
  - 將所有 `self.prices.data` 改為 `self.prices.to_pandas()`

  **關鍵修改點**:
  - `_validate_inputs()` - 檢查資料類型
  - `_get_common_timestamps()` - 時間戳對齊
  - `_prepare_data_access()` - groupby 邏輯

  **TDD - RED Phase**:
  確保現有 `tests/backtest/test_backtester.py` 測試通過。

  **Must NOT do**:
  - 不要改變回測邏輯
  - 不要改變輸出格式

  **Parallelizable**: YES（與 3.1 平行）

  **References**:
  - `src/factorium/backtest/backtester.py` - 現有實作

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/backtest/test_backtester.py -v` → PASS
  - [ ] `Backtester` 公開 API 不變

  **Commit**: `fix(backtest): adapt to Polars Factor API`

---

### Phase 4: 整合測試與驗證

- [ ] 4.1 整合測試 - 全面測試套件驗證

  **What to do**:
  - 執行完整測試套件
  - 修復任何殘留問題
  - 更新 `tests/test_expression_parser.py` 如有需要

  **Acceptance Criteria**:
  - [ ] `uv run pytest` → 所有測試 PASS
  - [ ] 測試覆蓋率不下降

  **Commit**: `test: final integration test fixes for Polars migration`

---

- [ ] 4.2 效能驗證 - Benchmark

  **What to do**:
  建立效能基準測試：
  ```python
  # scripts/benchmark_polars.py
  import time
  import tracemalloc
  from factorium import Factor
  
  def benchmark():
      # 生成大量測試資料
      df = generate_large_factor_data(n_symbols=1000, n_periods=10000)
      factor = Factor(df)
      
      # 測試鏈式運算
      tracemalloc.start()
      start = time.perf_counter()
      
      result = (
          factor
          .ts_mean(20)
          .cs_rank()
          .ts_zscore(10)
          .ts_rank(5)
      )
      
      # Lazy phase - 應該很快
      lazy_time = time.perf_counter() - start
      
      # Collection phase
      start = time.perf_counter()
      _ = result.data
      collect_time = time.perf_counter() - start
      
      current, peak = tracemalloc.get_traced_memory()
      tracemalloc.stop()
      
      print(f"Lazy build time: {lazy_time:.4f}s")
      print(f"Collect time: {collect_time:.4f}s")
      print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
  ```

  **Acceptance Criteria**:
  - [ ] 鏈式運算建構時間 < 0.1s（lazy 特性）
  - [ ] 記憶體使用量比 Pandas 版本降低（大資料集）

  **Commit**: NO（純驗證，不需提交）

---

## Risk Assessment

### High Risk
| 風險 | 影響 | 緩解措施 |
|-----|------|---------|
| 數值精度差異 | 金融計算錯誤 | TDD 強制驗證精度，保留 PandasEngine 對照 |
| Breaking API | 下游程式碼需修改 | 提供 `to_pandas()` 過渡方法 |
| 複雜運算難以 Polars 化 | 部分功能無法遷移 | 允許使用 `map_batches` fallback |

### Medium Risk
| 風險 | 影響 | 緩解措施 |
|-----|------|---------|
| LazyFrame 限制 | 某些操作不支援 lazy | 在必要時使用 `.collect()` |
| 測試覆蓋不足 | 遺漏邊緣案例 | TDD 開發，擴展測試案例 |

---

## Commit Strategy

| Phase | Task | Commit Message | Verification |
|-------|------|----------------|--------------|
| 1 | 1.1 | `refactor(factor): migrate base.py to Polars LazyFrame` | test_base_polars.py |
| 1 | 1.2 | `refactor(factor): migrate binary operations to Polars` | test_base_polars.py |
| 2 | 2.1 | `refactor(factor): migrate ts_ops.py to pure Polars` | test_ts_ops_polars.py |
| 2 | 2.2 | `refactor(factor): migrate cs_ops.py to pure Polars` | test_cs_ops_polars.py |
| 2 | 2.3 | `refactor(factor): migrate math_ops.py to pure Polars` | test_math_ops_polars.py |
| 3 | 3.1 | `fix(analyzer): adapt to Polars Factor API` | test_analyzer.py |
| 3 | 3.2 | `fix(backtest): adapt to Polars Factor API` | test_backtester.py |
| 4 | 4.1 | `test: final integration test fixes for Polars migration` | pytest |

---

## Subagent Usage Guide

Phase 2 和 Phase 3 的任務可以使用 subagent 平行開發：

### Phase 2 平行執行
```
Subagent A: Task 2.1 (ts_ops.py)
Subagent B: Task 2.2 (cs_ops.py)
Subagent C: Task 2.3 (math_ops.py)
```

### Phase 3 平行執行
```
Subagent A: Task 3.1 (analyzer.py)
Subagent B: Task 3.2 (backtester.py)
```

每個 subagent 應：
1. 遵循 TDD 流程（RED-GREEN-REFACTOR）
2. 確保數值精度測試通過
3. 在完成前執行相關測試驗證
4. 不要修改非指定檔案

---

## Success Criteria

### Final Verification Commands
```bash
# 全部測試通過
uv run pytest

# 驗證無 Pandas 殘留（在 mixins 中）
grep -r "pd\." src/factorium/factors/mixins/ | grep -v "pd.DataFrame" | grep -v "type" | grep -v "import"

# 驗證 LazyFrame 類型
python -c "
from factorium import Factor
import polars as pl
import pandas as pd

# Test initialization
df = pd.DataFrame({
    'start_time': pd.date_range('2020-01-01', periods=10),
    'end_time': pd.date_range('2020-01-01', periods=10),
    'symbol': ['A'] * 10,
    'factor': range(10)
})
f = Factor(df)

assert isinstance(f.lazy, pl.LazyFrame), 'lazy should be LazyFrame'
assert isinstance(f.data, pl.DataFrame), 'data should be pl.DataFrame'
assert isinstance(f.to_pandas(), pd.DataFrame), 'to_pandas should return pd.DataFrame'
print('All type checks passed!')
"

# 效能測試
uv run python scripts/benchmark_polars.py
```

### Final Checklist
- [x] `Factor._lf` 類型為 `pl.LazyFrame`
- [x] `factor.data` 返回 `pl.DataFrame`
- [x] `factor.to_pandas()` 可用於下游相容
- [x] 所有 mixin 方法使用純 Polars 表達式 (除 `ts_quantile` 需 scipy)
- [x] 所有測試通過 (478/478)
- [x] 數值精度符合金融計算標準
- [x] TDD 流程完整執行

---

## 後續擴展: polars-ds 插件

### 評估結果

`polars-ds` (Polars for Data Science) 是一個值得投資的 Polars 擴展插件。

**基本資訊**
- **GitHub**: [abstractqqq/polars_ds_extension](https://github.com/abstractqqq/polars_ds_extension)
- **版本**: v0.10.4+ (活躍維護)
- **授權**: MIT License
- **安裝**: `pip install polars-ds`

**與 Factorium 契合的功能**
| 功能 | 說明 |
|------|------|
| Rolling Linear Regression | 滑動窗口線性回歸 (含 Lasso/Ridge) |
| 統計檢定 | t-test, Chi², KS-test |
| 相關性分析 | Pearson, Spearman, Kendall, **Xi Correlation** |
| **PSI** | Population Stability Index (因子失效檢測) |
| 時間序列特徵 | 去趨勢、卷積、近似熵、Lempel-Ziv 複雜度 |

**技術優勢**
- 完全相容 Polars 1.37+
- Rust 實作，利用 Polars 並行能力
- 輕量依賴

**注意事項**
- 不支援 Polars Streaming 模式
- Beta 狀態，API 可能微調

### 未來整合計畫

1. **Phase 1**: 引入 `polars-ds` 作為可選依賴
2. **Phase 2**: 利用其統計檢定功能增強因子分析
3. **Phase 3**: 探索使用 `statrs` Rust 庫實現純 Polars PPF
