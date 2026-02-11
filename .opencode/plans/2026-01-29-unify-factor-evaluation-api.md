# 統一因子評估 API 設計

**日期：** 2026-01-29  
**狀態：** 設計完成，待實作  
**相關 Issue：** #3

---

## 背景與目標

### 現狀問題

目前專案中存在兩套重複的因子評估系統：

| 類別 | 技術棧 | 回傳類型 | 量化指標 | 分桶策略 |
|------|--------|----------|----------|----------|
| `FactorAnalyzer` | Polars | `FactorAnalysisResult` | IC, ICIR, t-stat, Quantile Returns | Per-day cross-sectional |
| `FactorEvaluator` | Pandas | `Dict[str, Any]` | IC, ICIR, Turnover, Layer Returns | Global quantiles |

- `Factor.eval()` 目前委託給 `FactorEvaluator`（Pandas 路徑）
- `ResearchSession.analyze()` 使用 `FactorAnalyzer`（Polars 路徑）
- 兩者功能重疊但實作細節不一致，造成維護負擔

### 設計目標

1. **統一評估路徑：** `FactorAnalyzer` + `FactorAnalysisResult` 為唯一評估入口
2. **Polars-first：** 使用 Polars 作為核心運算引擎，提升效能
3. **完整指標：** 整合 IC, ICIR, t-stat, Turnover, Quantile Returns, Cumulative Returns
4. **實驗追蹤：** 支援自動輸出實驗結果到指定目錄
5. **明確分層：** Evaluation 層（因子預測能力）vs Backtest 層（投資組合表現）

---

## 架構設計

### 兩層架構

```
┌─────────────────────────────────────────────────────────┐
│                    Evaluation 層                         │
│  FactorAnalyzer.analyze() → FactorAnalysisResult        │
│  指標：IC, ICIR, t-stat, Turnover,                       │
│        Quantile Returns, Cumulative Returns              │
│  目的：評估因子的預測能力                                  │
├─────────────────────────────────────────────────────────┤
│                    Backtest 層                           │
│  VectorizedBacktester → BacktestResult                   │
│  指標：Sharpe, Sortino, MaxDD, Calmar, Returns...        │
│  目的：模擬實際投資組合表現                                │
└─────────────────────────────────────────────────────────┘
```

### API 入口點（統一後）

- `Factor.eval(prices)` → 委託給 `FactorAnalyzer.analyze()` → 回傳 `FactorAnalysisResult`
- 移除 `FactorEvaluator` 和 `evaluation.py`

---

## 詳細設計

### 1. FactorAnalysisResult 擴展

```python
@dataclass
class FactorAnalysisResult:
    """因子評估結果（完整版）"""
    
    # 基本資訊
    factor_name: str
    periods: int | list[int]      # MVP 僅支援 int，為 Issue #4 預留 list
    quantiles: int
    
    # IC 相關
    ic_series: pd.DataFrame       # index=start_time, columns=period_n
    ic_summary: dict[str, float]  # keys: mean_ic, ic_std, ic_ir, t_stat
    
    # Turnover（新增）
    turnover_series: pd.Series    # index=start_time, value=1-rank_autocorr
    turnover_mean: float
    
    # Quantile 相關
    quantile_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame | None = None
    
    # 便利方法
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        ...
    
    def __repr__(self) -> str:
        """顯示關鍵摘要（mean_ic, ic_ir, turnover_mean）"""
        ...
    
    def plot(self) -> None:
        """一鍵繪製所有圖表（使用現有 plotting_analyzer）"""
        ...
    
    def save(self, output_dir: str) -> None:
        """
        保存實驗結果到指定目錄
        
        產生結構：
        {output_dir}/
        └── YYYYMMDD_HHMMSS_{factor_name}/
            ├── config.json
            ├── ic_series.csv
            ├── ic_summary.csv
            ├── turnover.csv
            ├── quantile_returns.csv
            ├── cumulative_returns.csv
            └── plots/
                ├── ic_distribution.png
                ├── ic_timeseries.png
                ├── quantile_returns.png
                └── cumulative_returns.png
        """
        ...
```

### 2. FactorAnalyzer 擴展

**新增方法：**

```python
class FactorAnalyzer:
    def calculate_turnover(self) -> pd.Series:
        """
        計算因子的 turnover（使用 rank 自相關）
        
        方法：
        1. 對每個 start_time，計算 cross-sectional rank
        2. 計算與前一天 rank 的相關係數（rank autocorrelation）
        3. turnover = 1 - rank_autocorr
        
        實作細節：
        - 使用 Polars（從 FactorEvaluator.calculate_turnover 移植）
        - 輸入：self._clean_data（需包含 factor 欄位）
        - 輸出：pd.Series (index=start_time, value=turnover)
        
        Returns:
            pd.Series: Turnover 時間序列
        """
        # 實作使用 Polars pipeline：
        # 1. group_by("start_time") + rank("factor")
        # 2. 計算 rank 與 lag(rank) 的相關係數
        # 3. 轉換為 pandas Series
        ...
```

**更新 analyze() 方法：**

```python
def analyze(
    self, 
    price_col: str = "close", 
    periods: int = 1
) -> FactorAnalysisResult:
    """
    執行完整因子評估
    
    流程：
    1. prepare_data()
    2. calculate_ic()
    3. calculate_ic_summary()
    4. calculate_turnover()        # 新增
    5. calculate_quantile_returns()
    6. calculate_cumulative_returns()
    7. 組裝 FactorAnalysisResult
    """
    self.prepare_data(periods=[periods], price_col=price_col)
    
    ic_series = self.calculate_ic(method="rank")
    ic_summary = self.calculate_ic_summary(method="rank")
    
    # 新增：計算 turnover
    turnover_series = self.calculate_turnover()
    turnover_mean = turnover_series.mean()
    
    quantile_returns = self.calculate_quantile_returns(
        quantiles=self.quantiles, period=periods
    )
    cumulative_returns = self.calculate_cumulative_returns(
        quantiles=self.quantiles, period=periods, long_short=True
    )
    
    return FactorAnalysisResult(
        factor_name=self.factor.name,
        periods=periods,
        quantiles=self.quantiles,
        ic_series=ic_series,
        ic_summary=ic_summary,
        turnover_series=turnover_series,  # 新增
        turnover_mean=turnover_mean,      # 新增
        quantile_returns=quantile_returns,
        cumulative_returns=cumulative_returns,
    )
```

### 3. Factor.eval() 重構

**重構前（現況）：**
```python
def eval(self, prices: "Factor", periods: List[int] = [1, 5, 10], 
         quantiles: int = 5, save_path: Optional[str] = None, 
         **kwargs) -> Dict[str, Any]:
    from .evaluation import FactorEvaluator
    evaluator = FactorEvaluator(self, prices)
    return evaluator.run_full_report(
        periods=periods, quantiles=quantiles, 
        save_path=save_path, **kwargs
    )
```

**重構後（新設計）：**
```python
def eval(
    self, 
    prices: "Factor | AggBar",
    periods: int = 1,              # MVP 僅支援單一窗口
    quantiles: int = 5,
    output_dir: str | None = None,
    price_col: str = "close",
    **kwargs
) -> FactorAnalysisResult:
    """
    評估因子的預測能力（Evaluation 層）
    
    Args:
        prices: 價格數據（Factor 或 AggBar）
        periods: 預測時間窗口（未來可擴展為 list[int]）
        quantiles: 分層數量（預設 5）
        output_dir: 實驗輸出目錄（如指定會自動創建資料夾）
        price_col: 價格欄位名稱（預設 "close"）
    
    Returns:
        FactorAnalysisResult: 包含所有評估指標
    
    Example:
        >>> momentum = ts_returns(close, 20)
        >>> result = momentum.eval(prices, output_dir="./experiments")
        >>> print(result.ic_summary)
        {'mean_ic': 0.05, 'ic_ir': 1.2, 't_stat': 3.5, ...}
    """
    from .analyzer import FactorAnalyzer
    
    analyzer = FactorAnalyzer(
        factor=self, 
        prices=prices, 
        quantiles=quantiles
    )
    
    result = analyzer.analyze(price_col=price_col, periods=periods)
    
    # 如果指定輸出目錄，保存結果
    if output_dir:
        result.save(output_dir)
    
    return result
```

### 4. 實驗輸出結構

**資料夾命名：** `YYYYMMDD_HHMMSS_{factor_name}/`

**完整結構：**
```
experiments/
└── 20240115_103045_momentum_20/
    ├── config.json               # 實驗設定與 metadata
    ├── ic_series.csv             # IC 時間序列
    ├── ic_summary.csv            # IC 統計摘要
    ├── turnover.csv              # Turnover 時間序列
    ├── quantile_returns.csv      # 分層報酬
    ├── cumulative_returns.csv    # 累積報酬
    └── plots/
        ├── ic_distribution.png
        ├── ic_timeseries.png
        ├── quantile_returns.png
        └── cumulative_returns.png
```

**config.json 範例：**
```json
{
  "factor_name": "momentum_20",
  "expression": "ts_returns(close, 20)",
  "created_at": "2024-01-15T10:30:45.123456",
  "periods": 1,
  "quantiles": 5,
  "price_col": "close",
  "data_range": {
    "start": "2023-01-01",
    "end": "2024-01-15",
    "n_symbols": 50,
    "n_observations": 12500
  }
}
```

**時間戳實作：**
```python
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{timestamp}_{self.factor_name}"
output_path = Path(output_dir) / folder_name
output_path.mkdir(parents=True, exist_ok=True)
```

---

## 移除的內容

### 檔案
- `src/factorium/factors/evaluation.py`（整個檔案）

### 類別
- `FactorEvaluator`

### 理由
- 功能完全由 `FactorAnalyzer` 取代
- 避免雙軌維護
- Polars 實作更高效

---

## MVP 範圍與未來擴展

### MVP（本次實作）

**功能範圍：**
- ✅ 統一為 `FactorAnalyzer` 路徑
- ✅ 整合 IC, ICIR, t-stat, Turnover, Quantile Returns, Cumulative Returns
- ✅ 僅支援單一 `periods: int`（不支援多時間窗口）
- ✅ 僅評估 return 預測能力
- ✅ Per-day cross-sectional quantiles
- ✅ 輸出格式：CSV + PNG
- ✅ 實驗資料夾自動命名（秒級時間戳）

**技術細節：**
- Turnover 使用 rank 自相關方法
- 所有數據使用 Polars 計算，最後轉為 Pandas（為相容現有繪圖工具）

### 未來擴展（Issue #4, #5）

**預留設計空間：**
- 🔄 多時間窗口支援（`periods: int | list[int]`）
- 🔄 IC 衰減曲線分析
- 🔄 不同預測目標（volatility, drawdown 等）
- 🔄 互動式報告（HTML/WandB/TensorBoard）
- 🔄 更多輸出格式（Parquet, JSON）

**設計考量：**
- `FactorAnalysisResult.periods` 已預留 `int | list[int]` 類型
- `config.json` 結構可輕鬆擴展新欄位
- `save()` 方法可透過參數控制輸出格式

---

## 實作步驟

### Phase 1: 擴展 FactorAnalyzer

1. **新增 `calculate_turnover()` 方法**
   - 從 `FactorEvaluator.calculate_turnover()` 移植邏輯
   - 改寫為 Polars 實作
   - 確保與 Pandas 版本數值一致（撰寫單元測試驗證）

2. **更新 `analyze()` 方法**
   - 整合 turnover 計算
   - 組裝新的 `FactorAnalysisResult`

3. **單元測試**
   - `test_calculate_turnover()`: 驗證計算正確性
   - `test_analyze_with_turnover()`: 驗證完整流程

### Phase 2: 擴展 FactorAnalysisResult

1. **新增欄位**
   - `turnover_series: pd.Series`
   - `turnover_mean: float`

2. **新增 `save()` 方法**
   - 實作資料夾創建（含時間戳）
   - 輸出 CSV 檔案
   - 輸出圖表（複用 `plotting_analyzer`）
   - 生成 `config.json`

3. **更新 `__repr__()`**
   - 加入 turnover_mean 顯示

4. **單元測試**
   - `test_save_creates_correct_structure()`: 驗證輸出結構
   - `test_save_csv_content()`: 驗證 CSV 內容
   - `test_config_json()`: 驗證 config.json 格式

### Phase 3: 重構 Factor.eval()

1. **更新方法簽名**
   - 移除 `save_path`，改為 `output_dir`
   - 簡化參數（MVP 僅支援 `periods: int`）

2. **委託給 FactorAnalyzer**
   - 移除 `FactorEvaluator` 引用
   - 調用 `FactorAnalyzer.analyze()`

3. **整合測試**
   - `test_eval_returns_analysis_result()`: 驗證回傳類型
   - `test_eval_with_output_dir()`: 驗證輸出功能
   - 與現有回測測試整合（確保不破壞下游功能）

### Phase 4: 移除舊代碼

1. **刪除檔案**
   - `src/factorium/factors/evaluation.py`

2. **更新 imports**
   - 檢查所有引用 `FactorEvaluator` 的地方
   - 移除相關 import

3. **清理測試**
   - 移除 `test_evaluation.py`（如果存在）

4. **回歸測試**
   - 執行完整測試套件
   - 確保沒有破壞現有功能

### Phase 5: 更新文檔

1. **用戶指南**
   - 更新 `docs/user-guide/factor.md`
   - 新增實驗輸出範例

2. **API 文檔**
   - 更新 `Factor.eval()` 說明
   - 新增 `FactorAnalysisResult.save()` 範例

3. **CHANGELOG**
   - 記錄 breaking changes
   - 說明遷移路徑

---

## 測試策略

### 單元測試

1. **Turnover 計算正確性**
   - 與 Pandas 版本對比（數值容差 < 1e-10）
   - 邊界條件：空數據、單一日期、缺失值

2. **實驗輸出**
   - 資料夾命名格式
   - CSV 檔案完整性
   - config.json 結構

3. **向後相容性**
   - `Factor.eval()` 簽名變更（型別檢查）

### 整合測試

1. **端到端流程**
   - 創建 factor → eval() → 驗證輸出
   - 使用真實市場數據

2. **與 Backtest 層整合**
   - 確保 Evaluation 結果可傳遞給 Backtester

### 效能測試

1. **大規模數據**
   - 1000+ symbols, 5+ years data
   - Polars vs Pandas 效能對比

---

## 風險與注意事項

### Breaking Changes

- `Factor.eval()` 回傳類型從 `Dict[str, Any]` 改為 `FactorAnalysisResult`
- 參數名稱變更：`save_path` → `output_dir`
- `periods` 從 `List[int]` 改為 `int`（MVP）

**影響範圍：**
- 直接調用 `Factor.eval()` 的代碼需要更新
- 預期影響有限（主要為內部測試與範例）

### 數值一致性

- Turnover 移植時需確保 Polars 與 Pandas 實作數值完全一致
- 使用單元測試嚴格驗證（容差 < 1e-10）

### 檔案系統

- 高頻實驗可能產生大量資料夾
- 建議未來加入：
  - 自動清理舊實驗
  - 實驗 metadata 索引（SQLite）

---

## 成功標準

1. ✅ 所有測試通過（含新增的 turnover 測試）
2. ✅ `Factor.eval()` 可正常輸出實驗結果
3. ✅ Turnover 數值與 Pandas 版本一致
4. ✅ `FactorEvaluator` 相關代碼完全移除
5. ✅ 文檔更新完成
6. ✅ 效能測試通過（Polars 實作不慢於 Pandas）

---

## 附錄

### 參考檔案

- `src/factorium/factors/analyzer.py`（主要修改）
- `src/factorium/factors/core.py`（Factor.eval）
- `src/factorium/factors/evaluation.py`（待移除）
- `tests/factors/test_analyzer.py`（測試）

### 相關 Issues

- #3: 統一因子評估 API（本次）
- #4: 多時間窗口 IC 衰減（下一步）
- #5: 信號到曝險映射（Backtest 層）
