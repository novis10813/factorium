# Factorium Code Review

> 審查日期：2026-04-09
> 審查範圍：`src/factorium/` 全部模組（factors, backtest, data, aggbar, universe, research, storage）

---

## 總覽

| 嚴重程度 | 數量 |
|----------|------|
| Critical | 4 |
| High | 10 |
| Medium | 15 |
| Low | 11 |
| **合計** | **40** |

---

## Critical（嚴重）

### C-1. `engine.py` — NaN 偵測只用 `is_null()`，遺漏 float NaN

- **檔案**: `src/factorium/factors/engine.py`
- **位置**: L123, L140, L154, L167, L186, L199（所有 `cs_*` 方法）
- **影響函數**: `cs_rank`, `cs_zscore`, `cs_demean`, `cs_winsorize`, `cs_mean`, `cs_median`

```python
# engine.py 目前的寫法
has_nan = pl.col(value_col).is_null().any().over(time_col)

# cs_ops.py mixin 正確的寫法
has_nan = (pl.col("factor").is_null() | pl.col("factor").is_nan()).any().over("end_time")
```

**問題**：Polars 中 `is_null()` **不會** 偵測到 `float('nan')`（NaN 和 null 是不同概念）。從 Pandas/NumPy 轉換來的資料常包含 float NaN，會導致嚴格的 NaN 傳遞機制完全失效。engine 會用不完整的資料計算結果，產生錯誤的因子數值且無任何警告。

---

### C-2. `operators.py` — `div(float, Factor)` 繞過 safe_div

- **檔案**: `src/factorium/factors/operators.py`
- **位置**: L338–341
- **影響函數**: `div`

```python
elif isinstance(factor2, FactorClass):
    # float / Factor = (1/Factor) * float = Factor.pow(-1) * float
    return cast("Factor", factor2.pow(-1) * factor1)
```

**問題**：`pow(-1)` 在 `math_ops.py` 中使用原始 `pl.col("factor").pow(pl.lit(-1))`，不做 EPSILON 檢查。而 `BaseFactor.__rtruediv__` 正確地使用了 `pl.when(pl.col("factor").abs() <= EPSILON).then(pl.lit(None))` 保護。

parser 解析 `5 / close` 這類表達式時會走到 `operators.div(5.0, close_factor)` 這條路徑，當分母接近零時產生 `inf` 而非 `null`，違反專案的 `safe_div` 規範。

**修復建議**：改用 `factor1 / factor2`（觸發 `__rtruediv__`），或在 `pow` 中加入 EPSILON 檢查。

---

### C-3. `research/session.py` — `or` 誤判 falsy 有效值

- **檔案**: `src/factorium/research/session.py`
- **位置**: L302–303
- **影響函數**: `backtest`

```python
initial_capital=initial_capital or self.default_initial_capital,
transaction_cost=transaction_cost or self.default_transaction_cost,
```

**問題**：使用者傳入 `transaction_cost=0.0`（模擬零手續費回測），Python 中 `0.0` 為 falsy，`or` 會回傳預設值 `0.0003`，使用者的意圖被靜默覆蓋。同理，`initial_capital=0` 也會被覆蓋。

**修復建議**：

```python
initial_capital=self.default_initial_capital if initial_capital is None else initial_capital,
transaction_cost=self.default_transaction_cost if transaction_cost is None else transaction_cost,
```

---

### C-4. `data/utils.py` — 時區混用（naive vs aware datetime）

- **檔案**: `src/factorium/data/utils.py`
- **位置**: L42–43, L52（naive），L56（aware）
- **影響函數**: `calculate_date_range`

```python
# L42-43: timezone-naive
start = datetime.strptime(start_date, "%Y-%m-%d")
end_inclusive = datetime.strptime(end_date, "%Y-%m-%d")

# L56: timezone-aware (UTC)
today_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
```

**問題**：`strptime` 回傳 timezone-naive datetime，但後續分支使用 `datetime.now(timezone.utc)` 回傳 timezone-aware。在非 UTC 系統上呼叫 `.timestamp()` 時，naive datetime 會被解讀為本地時間。例如：在 UTC+8 伺服器上，`"2024-01-01"` 會變成 `2023-12-31T16:00:00 UTC`，導致 DuckDB 查詢的時間範圍偏移。

**修復建議**：所有 `datetime.strptime(...)` 後加上 `.replace(tzinfo=timezone.utc)`。

---

## High（高）

### H-1. `composite.py` — `to_factor()` 用 `fill_null(0)` 而非傳遞 NaN

- **檔案**: `src/factorium/factors/composite.py`
- **位置**: L145
- **影響函數**: `to_factor`

```python
result = result.with_columns(
    (pl.col("weighted") + pl.col(f"weighted_{i}").fill_null(0)).alias("weighted")
)
```

**問題**：某因子缺失的 (time, symbol) 對被靜默填為 0，而非傳遞 null。合成因子看似有效，實際上是用更少的因子計算出的結果，違反專案的嚴格 NaN 傳遞原則。

---

### H-2. `composite.py` — left join 丟棄非首因子的獨有 rows

- **檔案**: `src/factorium/factors/composite.py`
- **位置**: L140–144
- **影響函數**: `to_factor`

```python
result = result.join(
    factor_df.select(["start_time", "end_time", "symbol", f"weighted_{i}"]),
    on=["start_time", "end_time", "symbol"],
    how="left",
)
```

**問題**：合成因子的 universe 永遠由第一個因子（`factors[0]`）決定。任何存在於 `factors[1]`、`factors[2]` 等但不在 `factors[0]` 中的 (time, symbol) 對會被靜默丟棄。`CompositeFactor([A, B])` 和 `CompositeFactor([B, A])` 可能產生不同結果。

**修復建議**：改用 `how="outer"` 或先取所有因子的 union key 再逐一 join。

---

### H-3. `composite.py` — `from_zscore()` 用 `fill_nan(0.0)` 掩蓋未定義 z-score

- **檔案**: `src/factorium/factors/composite.py`
- **位置**: L113–114
- **影響函數**: `from_zscore`

```python
((pl.col("factor") - pl.col("cs_mean")) / pl.col("cs_std"))
    .fill_nan(0.0)  # Handle division by zero
    .alias("factor")
```

**問題**：所有 symbol 值相同時 std=0，z-score 為 0/0=NaN 或 x/0=inf。填為 0.0 讓常數值因子獲得一個人為的「中性」z-score，引入偏差並違反嚴格 NaN 傳遞原則。

---

### H-4. `cs_ops.py` — `cs_zscore` 缺少 EPSILON 和 infinity 檢查

- **檔案**: `src/factorium/factors/mixins/cs_ops.py`
- **位置**: L29–36
- **影響函數**: `cs_zscore`

```python
std_expr = pl.col("factor").std(ddof=1).over("end_time")
z_expr = (pl.col("factor") - mean_expr) / std_expr
result_lf = self._lf.with_columns(
    pl.when(nan_mask).then(pl.lit(None)).otherwise(z_expr).alias("factor")
)
```

**問題**：對比 `ts_zscore` 有三層防護（NaN mask + `std.abs() <= EPSILON` → null + `is_finite()` → null），`cs_zscore` 完全沒有 EPSILON 或 infinity 檢查。當某時間截面所有 symbol 值近乎相同，std 接近零時會產生極大值或 `inf`。

---

### H-5. `ts_ops.py` — `ts_scale` 和 `ts_zscore` 缺少 sort

- **檔案**: `src/factorium/factors/mixins/ts_ops.py`
- **位置**: L236–256（`ts_scale`），L258–277（`ts_zscore`）

**問題**：其他所有 `ts_*` 方法（如 `ts_mean`、`ts_std`、`ts_corr` 等）在 rolling 操作前都有 `.sort(["symbol", "end_time"])`，但 `ts_scale` 和 `ts_zscore` 沒有。Polars rolling operations 依賴物理行順序，若 LazyFrame 未按時間排序（例如經過 join 或 filter 後），rolling window 會涵蓋錯誤的 row，結果靜默錯誤。

**修復建議**：在 rolling 操作前加入 `.sort(["symbol", "end_time"])`。

---

### H-6. `backtest/metrics.py` — Sortino ratio 計算錯誤

- **檔案**: `src/factorium/backtest/metrics.py`
- **位置**: L60–63
- **影響函數**: `calculate_metrics`

```python
downside_returns = returns[returns < 0]
if len(downside_returns) > 0:
    downside_std = float(downside_returns.std() * np.sqrt(periods_per_year))
```

**問題**：標準 downside deviation 公式為：

$$\sigma_{down} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \min(r_i - target, 0)^2}$$

程式碼用的是：

$$\sigma_{code} = \sqrt{\frac{1}{n_{neg}-1} \sum_{r_i < 0} (r_i - \bar{r}_{neg})^2}$$

兩個錯誤：(1) 以負報酬的均值為中心，而非 target return (0)；(2) 除以 `n_neg - 1`（Pandas `.std()` 的 ddof=1）而非 `N`（總期數）。對於 60% 正報酬的策略，downside deviation 會被系統性地高估，Sortino ratio 被低估。

---

### H-7. `backtest/vectorized.py` — `shift().over()` 在 sort 之前執行

- **檔案**: `src/factorium/backtest/vectorized.py`
- **位置**: L165–176
- **影響函數**: `_prepare_data`

```python
# shift 在 sort 之前
combined = combined.with_columns([
    pl.col("signal").shift(1).over("symbol").alias("prev_signal")
]).drop("signal")

combined = combined.with_columns([
    (pl.col("price") / pl.col("price").shift(1).over("symbol") - 1.0).alias("asset_return")
])

# sort 在 shift 之後
combined = combined.sort(["end_time", "symbol"])
```

**問題**：`shift(1).over("symbol")` 依賴物理行順序。若輸入的 raw `pl.DataFrame` 未預先按時間排序，「前一期 signal」可能實際上是未來的 signal，引入**前瞻偏差（look-ahead bias）**；`asset_return` 可能從非連續的價格計算，產生無意義的報酬率。

**修復建議**：把 `sort` 移到 `shift` 之前。

---

### H-8. `parser.py` — 留有 debug print

- **檔案**: `src/factorium/factors/parser.py`
- **位置**: L142–144
- **影響函數**: `_evaluate`

```python
print(
    f"DEBUG: binary_op {op}, left: {getattr(left, 'name', left)}, right: {getattr(right, 'name', right)}"
)
```

**問題**：每次解析二元運算（+、-、*、/）都會印到 stdout。在生產環境中處理大量因子表達式時，會造成效能退化和輸出洪流。

**修復建議**：刪除或改為 `logger.debug()`。

---

### H-9. `universe/rules.py` — `MinListingAge` 在無 `listing_date` 時排除所有 symbol

- **檔案**: `src/factorium/universe/rules.py`
- **位置**: L119
- **影響函數**: `MinListingAge.apply`

**問題**：若 metadata 中沒有任何 symbol 有 `listing_date` 欄位，`listing_map` 為空，方法回傳 `pl.lit(False)`，整個 universe 被清空，且無任何警告。對於 spot market 或自定義資料源特別容易觸發。

**修復建議**：當 `listing_map` 為空時發出警告或提供 fallback。

---

### H-10. `data/loader.py` — Off-by-one：下載多抓一天

- **檔案**: `src/factorium/data/loader.py`
- **位置**: L689, L646–647
- **影響函數**: `_download_all_symbols`, `_download_missing_files`

**問題**：`end_dt` 已經是 exclusive end（如 Jan 8 = 資料到 Jan 7）。格式化為字串 `"2024-01-08"` 傳入 `download_data(end_date="2024-01-08")`。在 `download_data` 內部，`calculate_date_range` 將 `end_date` 視為 inclusive 再加 +1 天 → Jan 9。結果多下載一天。

**修復建議**：傳入前轉回 inclusive（`end_dt - timedelta(days=1)`），或新增接受 exclusive end 的內部 API。

---

## Medium（中）

### M-1. `metrics.py` — Sharpe ratio 混用 geometric return 與 arithmetic volatility

- **檔案**: `src/factorium/backtest/metrics.py`
- **位置**: L55–58

**問題**：`annual_return` 是複合年化報酬率（CAGR，geometric），`annual_volatility` 是算術年化標準差。教科書 Sharpe ratio 應統一使用 arithmetic 或 geometric。混用會在高波動策略中高估 Sharpe ratio。

---

### M-2. `engine.py` `cs_zscore` — 除以 std 無 EPSILON 保護

- **檔案**: `src/factorium/factors/engine.py`
- **位置**: L141–143

**問題**：與 H-4 類似，engine 版本的 `cs_zscore` 也缺少 EPSILON 防護。std 為零時產生 inf/NaN。

---

### M-3. `math_ops.py` `sqrt` — `factor == 0` 被排除

- **檔案**: `src/factorium/factors/mixins/math_ops.py`
- **位置**: L54–58

```python
pl.when(pl.col("factor") > 0).then(pl.col("factor").sqrt()).otherwise(None)
```

**問題**：`sqrt(0) = 0` 是合法結果，但條件 `> 0` 排除了 `factor == 0`，回傳 null。後續的嚴格 NaN 操作會進一步丟棄這些 row。

**修復建議**：改為 `pl.col("factor") >= 0`。

---

### M-4. `base.py` `_normalize_schema_lazy` — 按位置重命名的陷阱

- **檔案**: `src/factorium/factors/base.py`
- **位置**: L58–66

**問題**：4 欄 DataFrame 若無 `"factor"` 欄，會依位置 `[0, 1, 2, 3]` → `[start_time, end_time, symbol, factor]` 重命名。若使用者提供的欄順序不同（如 `[symbol, end_time, start_time, value]`），語義會靜默錯亂。

---

### M-5. `evaluation.py` `calculate_turnover` — 忽略 `factor_data` 參數

- **檔案**: `src/factorium/factors/evaluation.py`
- **位置**: L68–77

**問題**：方法接受 `factor_data: pd.DataFrame` 但完全不使用，永遠從 `self.factor` 重新計算。任何由 `_prepare_data` 做的篩選或對齊都被忽略。

---

### M-6. `evaluation.py` / `analyzer.py` — Turnover 定義相反

- **檔案**: `src/factorium/factors/evaluation.py` L76, `src/factorium/factors/analyzer.py` L477

**問題**：
- `evaluation.py` 回傳 rank autocorrelation（高值 = 低 turnover）
- `analyzer.py` 回傳 `1 - rank_autocorrelation`（高值 = 高 turnover）

從 `FactorEvaluator` 遷移到 `FactorAnalyzer` 的使用者會得到語義相反的結果。

---

### M-7. `analyzer.py` `prepare_data` — `drop_nulls()` 過度丟棄

- **檔案**: `src/factorium/factors/analyzer.py`
- **位置**: L385

```python
self._clean_data = df_lf.collect().drop_nulls()
```

**問題**：`drop_nulls()` 無指定 `subset`，會丟棄任何欄位為 null 的 row。嚴格 NaN 傳遞產生的合法 null 因子值也會被刪除，導致分析偏向數據完整的時期/標的。

**修復建議**：`drop_nulls(subset=["forward_return"])` 或適當的欄位子集。

---

### M-8. `parser.py` — Multi-element list fallback 靜默丟棄 tokens

- **檔案**: `src/factorium/factors/parser.py`
- **位置**: L245–247

```python
# If we have a list that didn't match anything above, it might be raw tokens
return self._evaluate(node_list[0], context)
```

**問題**：多元素 list 不匹配任何已知模式時，只評估第一個元素，其餘靜默丟棄。畸形的表達式不會報錯，而是產生部分結果。

---

### M-9. `ts_ops.py` — `ts_cv` / `ts_jumpiness` / `ts_vr` 用 `+ 1e-10` 模式

- **檔案**: `src/factorium/factors/mixins/ts_ops.py`
- **位置**: L562, L587, L662

```python
cv_expr = std_expr / (mean_expr.abs() + 1e-10)
result = total_jump / (range_val + 1e-10)
result = var_k / (k * var_1 + 1e-10)
```

**問題**：(1) 使用硬編碼 `1e-10` 而非集中定義的 `EPSILON` 常數；(2) `+ epsilon` 模式對所有結果引入微小偏差，不同於專案標準的 `abs(denom) <= EPSILON → null` 模式。

---

### M-10. `ts_ops.py` `ts_sum` — 輸出行順序不一致

- **檔案**: `src/factorium/factors/mixins/ts_ops.py`
- **位置**: L20–29

**問題**：`ts_sum` 使用 `__row_idx__` 恢復原始行順序，但所有其他 `ts_*` 方法輸出按 `["end_time", "symbol"]` 排序。在 pipeline 中混用可能導致物理行順序不同，影響後續 rolling 操作。

---

### M-11. `backtest/backtester.py` — 交易成本可能把 cash 推成負值

- **檔案**: `src/factorium/backtest/backtester.py`
- **位置**: L211–235

**問題**：目標持倉根據 total_value 計算（使用 100% 組合），但每筆交易額外扣除手續費。在高 turnover 或 `full_rebalance=True` 的情況下，`Σ|trade_value| + Σ|cost| > total_value`，cash 可能變為負數，隱性引入槓桿。

---

### M-12. `backtest/portfolio.py` — 缺少價格的持倉被靜默估值為 0

- **檔案**: `src/factorium/backtest/portfolio.py`
- **位置**: L65–70

```python
for symbol, qty in self.positions.items():
    if symbol in prices.index:
        market_value += qty * prices[symbol]
```

**問題**：若持有的 symbol 從 price feed 消失（下架、資料缺口），持倉被靜默估值為 $0，造成人為的突發回撤。position 仍存在但不計入 `total_value`，影響後續目標持倉計算。

---

### M-13. `backtest/allocators.py` — `TopNAllocator` 在 `n > count/2` 時不平衡

- **檔案**: `src/factorium/backtest/allocators.py`
- **位置**: L113–133

**問題**：long-short 模式下，當 `n > count/2` 時 top-N 和 bottom-N 集合重疊。`when/then` chain 優先分配 long_w，導致多頭多於空頭，組合不再 dollar-neutral。無任何驗證或警告。

---

### M-14. `data/aggregator.py` — `group_by` 不保證行內順序

- **檔案**: `src/factorium/data/aggregator.py`
- **位置**: L876–891（`_resample_klines`）

**問題**：`.first()` 和 `.last()` 在 `group_by().agg()` 中依賴 group 內的行順序。Polars 的 `group_by` 不保證保留輸入順序。即使前面有 `.sort()`，在某些 Polars 版本或後續更新中可能行為改變。

**修復建議**：使用 `pl.col("open").sort_by("start_dt").first()` 確保順序。

---

### M-15. `data/cache.py` — `force_download=True` 後 cache 未失效

- **檔案**: `src/factorium/data/cache.py`
- **位置**: 無特定行號

**問題**：在 `loader.py` 中 `force_download=True` 重新下載資料後，time bar 的迴圈仍先檢查 cache。若 cache 中有舊聚合結果，會被直接回傳而非重新聚合。使用者期望 `force_download=True` 拿到最新資料，但實際上可能拿到舊 cache。

---

## Low（低）

### L-1. `base.py` L4–7 — `Self` import fallback 兩邊相同

```python
try:
    from typing import Self
except ImportError:
    from typing import Self  # 應改為 from typing_extensions import Self
```

在 Python < 3.11 上會直接 crash。

---

### L-2. `evaluation.py` L1–3 — 重複 `import pandas as pd`

```python
import pandas as pd
import numpy as np
import pandas as pd  # 重複
```

---

### L-3. `evaluation.py` L109 — type annotation 與 default 不符

```python
def run_full_report(self, periods: List[int] = (1, 5, 10), ...):
```

標注為 `List[int]` 但預設值為 `tuple`。

---

### L-4. `operators.py` — 覆蓋 Python builtins

模組級函數 `abs`、`log`、`sqrt`、`pow`、`max`、`min` 覆蓋了 Python 內建函數。`from factorium.factors.operators import *` 會失去內建函數存取。

---

### L-5. `math_ops.py` `where` — NaN 條件被視為 truthy

```python
pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
```

Polars 中 `is_not_null()` 對 float NaN 回傳 `True`，`NaN != 0` 也為 `True`（IEEE 754）。NaN 條件值會被視為 truthy。

---

### L-6. `backtest/utils.py` L123 — 不可達的 dead code

```python
positive_signals = valid_signals[valid_signals > 0]
if len(positive_signals) == 0:
    return pd.Series(dtype=float)
total = positive_signals.sum()
if total == 0:  # 永遠不會為 0，因為 positive_signals 只有 > 0 的值
    return pd.Series(0.0, index=positive_signals.index)
```

---

### L-7. `backtest/utils.py` `safe_divide` — numpy 路徑多餘 RuntimeWarning

```python
result = np.where(
    np.isnan(b) | (np.abs(b) <= EPSILON),
    default,
    a / np.where(np.abs(b) <= EPSILON, 1.0, b),
)
```

NaN 值不被 `np.abs(b) <= EPSILON` 捕捉，中間計算 `a / NaN` 會觸發 RuntimeWarning。最終結果正確（被外層 `np.where` 覆蓋），但警告可能困擾使用者。

---

### L-8. `aggbar.py` `__repr__` — 空 AggBar 時 crash

空 AggBar 的 `self.timestamps` 回傳空 `DatetimeIndex`，`.min()` 為 `NaT`，`NaT.strftime(...)` 會拋出 `ValueError`。

---

### L-9. `aggbar.py` `slice` — 11–12 位整數 timestamp 落入錯誤分支

`convert_timestamp` 處理 10 位（秒級）和 13+ 位（毫秒級），但 11–12 位的整數（如 `17126000000`）落入 else 分支，被原樣處理，產生錯誤結果。

---

### L-10. `data/loader.py` — klines 與 trade bars 的 `end_time` 語義不一致

- Trade bars（aggregator.py）：`end_time = start_time + interval_ms`（**exclusive** end）
- Klines：`end_time = start_time + interval - 1`（**inclusive** end，Binance 原始格式）

下游程式碼用 `end_time` 做 join/alignment 時，來自不同來源的 AggBar 行為會不同。

---

### L-11. `storage/s3.py` — DuckDB SQL 字串插值有 SQL injection 風險

```python
con.execute(f"SET s3_access_key_id='{access_key}'")
con.execute(f"SELECT * FROM read_parquet('{uri}')")
```

`access_key`、`secret_key`（來自環境變數）或 `uri`（來自使用者路徑）若包含單引號 `'`，會導致 SQL injection 或語法錯誤。

---

## 建議修復優先順序

### 第一優先（立即修復）

| 編號 | 問題 | 預估工作量 |
|------|------|-----------|
| C-1 | engine.py NaN 偵測加入 `is_nan()` | 小 |
| C-2 | operators.py `div` 改用 `__rtruediv__` | 小 |
| C-3 | session.py `or` 改為 `is None` | 極小 |
| C-4 | utils.py datetime 統一加 tzinfo=UTC | 小 |
| H-8 | parser.py 刪除 debug print | 極小 |

### 第二優先（因子計算核心）

| 編號 | 問題 | 預估工作量 |
|------|------|-----------|
| H-1 | composite.py fill_null(0) → 傳遞 null | 小 |
| H-3 | composite.py fill_nan(0.0) → 傳遞 null | 小 |
| H-4 | cs_ops.py cs_zscore 加入 EPSILON 防護 | 小 |
| H-5 | ts_ops.py ts_scale/ts_zscore 加入 sort | 小 |
| M-3 | math_ops.py sqrt 改為 `>= 0` | 極小 |

### 第三優先（回測正確性）

| 編號 | 問題 | 預估工作量 |
|------|------|-----------|
| H-6 | metrics.py Sortino ratio 公式修正 | 中 |
| H-7 | vectorized.py 把 sort 移到 shift 前 | 小 |
| H-10 | loader.py off-by-one 修正 | 小 |
| M-1 | metrics.py Sharpe ratio arithmetic 一致性 | 中 |
| M-15 | cache.py force_download 清除 cache | 小 |
