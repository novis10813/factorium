# 統一因子評估 API 實作計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 統一因子評估 API，以 FactorAnalyzer 為核心，整合 Turnover 指標，並支援實驗輸出功能

**Architecture:** 擴展 FactorAnalyzer 類新增 turnover 計算（使用 Polars），擴展 FactorAnalysisResult 新增 save() 方法支援實驗輸出，重構 Factor.eval() 委託給 FactorAnalyzer，最後移除舊的 FactorEvaluator

**Tech Stack:** Polars (數據處理), Pandas (結果格式), Matplotlib (繪圖)

---

## Task 1: 新增 FactorAnalyzer.calculate_turnover() 方法

**Files:**
- Modify: `src/factorium/factors/analyzer.py:59-345`
- Test: `tests/factors/test_analyzer.py`

**目標:** 將 FactorEvaluator 的 turnover 計算移植到 FactorAnalyzer，使用 Polars 實作

**Step 1: 撰寫 turnover 計算的測試**

在 `tests/factors/test_analyzer.py` 新增測試：

```python
def test_calculate_turnover(sample_factor_and_prices):
    """Test turnover calculation using rank autocorrelation."""
    factor, prices = sample_factor_and_prices
    analyzer = FactorAnalyzer(factor, prices, quantiles=5)
    
    # Prepare data first
    analyzer.prepare_data(periods=[1])
    
    # Calculate turnover
    turnover_series = analyzer.calculate_turnover()
    
    # Assertions
    assert isinstance(turnover_series, pd.Series)
    assert turnover_series.index.name == "start_time"
    assert len(turnover_series) > 0
    # Turnover should be between 0 and 1 (or slightly outside due to correlation)
    assert turnover_series.min() >= -0.1  # Allow small negative due to correlation
    assert turnover_series.max() <= 1.1
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/factors/test_analyzer.py::test_calculate_turnover -v
```

預期輸出: `FAILED` - `AttributeError: 'FactorAnalyzer' object has no attribute 'calculate_turnover'`

**Step 3: 實作 calculate_turnover() 方法**

在 `src/factorium/factors/analyzer.py` 的 `FactorAnalyzer` 類中新增方法（約在第 232 行之後，`calculate_ic_summary` 之後）：

```python
    def calculate_turnover(self) -> pd.Series:
        """
        Calculate factor turnover using rank autocorrelation.
        
        Method:
        1. For each start_time, calculate cross-sectional rank of factor values
        2. Compute correlation between today's rank and yesterday's rank
        3. turnover = 1 - rank_autocorrelation
        
        Returns:
            pd.Series: Turnover time series indexed by start_time
        """
        if not hasattr(self, "_clean_data"):
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Use Polars to calculate rank per start_time
        df = self._clean_data.with_columns(
            pl.col("factor")
            .rank(method="average")
            .over("start_time")
            .alias("factor_rank")
        )
        
        # Normalize rank to percentile (0-1)
        df = df.with_columns(
            ((pl.col("factor_rank") - 1) / (pl.len().over("start_time") - 1))
            .alias("factor_rank_pct")
        )
        
        # Convert to pandas for correlation calculation
        df_pd = df.select(["start_time", "symbol", "factor_rank_pct"]).to_pandas()
        
        # Pivot to wide format (start_time x symbol)
        pivot = df_pd.pivot(index="start_time", columns="symbol", values="factor_rank_pct")
        
        # Calculate rank autocorrelation (correlation with previous day)
        rank_autocorr = pivot.corrwith(pivot.shift(1), axis=1)
        
        # Turnover = 1 - autocorrelation
        turnover = 1 - rank_autocorr
        
        return turnover
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_analyzer.py::test_calculate_turnover -v
```

預期輸出: `PASSED`

**Step 5: 提交變更**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): add calculate_turnover method using Polars"
```

---

## Task 2: 擴展 FactorAnalysisResult 新增 turnover 欄位

**Files:**
- Modify: `src/factorium/factors/analyzer.py:14-57`
- Test: `tests/factors/test_analyzer.py`

**目標:** 在 FactorAnalysisResult dataclass 新增 turnover_series 和 turnover_mean 欄位

**Step 1: 撰寫測試驗證新欄位存在**

在 `tests/factors/test_analyzer.py` 新增測試：

```python
def test_analysis_result_has_turnover_fields(sample_factor_and_prices):
    """Test that FactorAnalysisResult includes turnover fields."""
    factor, prices = sample_factor_and_prices
    analyzer = FactorAnalyzer(factor, prices, quantiles=5)
    
    result = analyzer.analyze(periods=1)
    
    # Check turnover fields exist
    assert hasattr(result, "turnover_series")
    assert hasattr(result, "turnover_mean")
    
    # Check types
    assert isinstance(result.turnover_series, pd.Series)
    assert isinstance(result.turnover_mean, (float, np.floating))
    
    # Check values are reasonable
    assert not np.isnan(result.turnover_mean)
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/factors/test_analyzer.py::test_analysis_result_has_turnover_fields -v
```

預期輸出: `FAILED` - `AttributeError: 'FactorAnalysisResult' object has no attribute 'turnover_series'`

**Step 3: 更新 FactorAnalysisResult dataclass**

修改 `src/factorium/factors/analyzer.py` 中的 `FactorAnalysisResult`（第 14-57 行）：

```python
@dataclass
class FactorAnalysisResult:
    """
    Structured result from factor analysis.

    Attributes:
        factor_name: Name of the analyzed factor
        periods: Analysis periods (forward return horizons)
        quantiles: Number of quantiles used
        ic_series: Information Coefficient time series
        ic_summary: Summary statistics of IC (mean, std, ir, t-stat)
        turnover_series: Turnover time series (1 - rank autocorrelation)
        turnover_mean: Average turnover across all periods
        quantile_returns: Mean returns by quantile
        cumulative_returns: Cumulative returns by quantile (if available)
    """

    factor_name: str
    periods: int
    quantiles: int
    ic_series: pd.DataFrame
    ic_summary: Dict[str, float]
    turnover_series: pd.Series
    turnover_mean: float
    quantile_returns: pd.DataFrame
    cumulative_returns: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "factor_name": self.factor_name,
            "periods": self.periods,
            "quantiles": self.quantiles,
            "ic_series": self.ic_series,
            "ic_summary": self.ic_summary,
            "turnover_series": self.turnover_series,
            "turnover_mean": self.turnover_mean,
            "quantile_returns": self.quantile_returns,
            "cumulative_returns": self.cumulative_returns,
        }

    def __repr__(self) -> str:
        ic = self.ic_summary
        return f"""FactorAnalysisResult: {self.factor_name}
  Periods: {self.periods}, Quantiles: {self.quantiles}
  Mean IC: {ic.get("mean_ic", 0):.4f}
  IC Std: {ic.get("ic_std", 0):.4f}
  IC IR: {ic.get("ic_ir", 0):.4f}
  Turnover: {self.turnover_mean:.4f}
"""
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_analyzer.py::test_analysis_result_has_turnover_fields -v
```

預期輸出: `FAILED` - 因為 `analyze()` 還沒有傳入 turnover 數據

**Step 5: 提交變更**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): add turnover fields to FactorAnalysisResult"
```

---

## Task 3: 更新 FactorAnalyzer.analyze() 整合 turnover

**Files:**
- Modify: `src/factorium/factors/analyzer.py:77-117`
- Test: `tests/factors/test_analyzer.py`

**目標:** 在 analyze() 方法中計算 turnover 並傳入 FactorAnalysisResult

**Step 1: 測試已存在（Task 2 的測試）**

使用 Task 2 的測試 `test_analysis_result_has_turnover_fields`

**Step 2: 確認測試仍然失敗**

```bash
uv run pytest tests/factors/test_analyzer.py::test_analysis_result_has_turnover_fields -v
```

預期輸出: `FAILED` - `TypeError: __init__() missing required positional argument: 'turnover_series'`

**Step 3: 更新 analyze() 方法**

修改 `src/factorium/factors/analyzer.py` 中的 `analyze()` 方法（約第 77-117 行）：

```python
    def analyze(self, price_col: str = "close", periods: int = 1) -> FactorAnalysisResult:
        """
        Run full factor analysis.

        Returns:
            FactorAnalysisResult with IC series, summary, turnover, and quantile returns
        """
        # Prepare data
        self.prepare_data(price_col=price_col, periods=[periods])

        # Calculate IC
        ic_series = self.calculate_ic()
        ic_summary_df = self.calculate_ic_summary()

        # Convert IC summary to dict for single period as expected by FactorAnalysisResult
        col = f"period_{periods}"
        ic_summary = {
            "mean_ic": ic_summary_df.loc["mean", col] if col in ic_summary_df.columns else 0.0,
            "ic_std": ic_summary_df.loc["std", col] if col in ic_summary_df.columns else 0.0,
            "ic_ir": ic_summary_df.loc["ic_ir", col] if col in ic_summary_df.columns else 0.0,
            "t-stat": ic_summary_df.loc["t-stat", col] if col in ic_summary_df.columns else 0.0,
        }

        # Calculate turnover
        turnover_series = self.calculate_turnover()
        turnover_mean = float(turnover_series.mean())

        # Calculate quantile returns
        quantile_returns = self.calculate_quantile_returns(quantiles=self.quantiles, period=periods)

        # Calculate cumulative returns (optional)
        try:
            cumulative_returns = self.calculate_cumulative_returns(quantiles=self.quantiles, period=periods)
        except Exception:
            cumulative_returns = None

        return FactorAnalysisResult(
            factor_name=self.factor.name,
            periods=periods,
            quantiles=self.quantiles,
            ic_series=ic_series,
            ic_summary=ic_summary,
            turnover_series=turnover_series,
            turnover_mean=turnover_mean,
            quantile_returns=quantile_returns,
            cumulative_returns=cumulative_returns,
        )
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_analyzer.py::test_analysis_result_has_turnover_fields -v
```

預期輸出: `PASSED`

**Step 5: 執行完整測試套件確認沒有破壞現有功能**

```bash
uv run pytest tests/factors/test_analyzer.py -v
```

預期輸出: 所有測試 `PASSED`

**Step 6: 提交變更**

```bash
git add src/factorium/factors/analyzer.py
git commit -m "feat(analyzer): integrate turnover calculation in analyze() method"
```

---

## Task 4: 實作 FactorAnalysisResult.save() 方法

**Files:**
- Modify: `src/factorium/factors/analyzer.py:14-57`
- Test: `tests/factors/test_analyzer.py`

**目標:** 實作 save() 方法將實驗結果輸出到指定目錄

**Step 1: 撰寫 save() 方法的測試**

在 `tests/factors/test_analyzer.py` 新增測試：

```python
import tempfile
from pathlib import Path
import json

def test_save_creates_correct_structure(sample_factor_and_prices):
    """Test that save() creates correct directory structure."""
    factor, prices = sample_factor_and_prices
    analyzer = FactorAnalyzer(factor, prices, quantiles=5)
    result = analyzer.analyze(periods=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result.save(tmpdir)
        
        # Find the created experiment folder
        exp_dirs = list(Path(tmpdir).glob("*_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]
        
        # Check folder name format: YYYYMMDD_HHMMSS_factorname
        folder_name = exp_dir.name
        parts = folder_name.split("_")
        assert len(parts) >= 3
        assert parts[0].isdigit() and len(parts[0]) == 8  # YYYYMMDD
        assert parts[1].isdigit() and len(parts[1]) == 6  # HHMMSS
        
        # Check CSV files exist
        assert (exp_dir / "ic_series.csv").exists()
        assert (exp_dir / "ic_summary.csv").exists()
        assert (exp_dir / "turnover.csv").exists()
        assert (exp_dir / "quantile_returns.csv").exists()
        assert (exp_dir / "cumulative_returns.csv").exists()
        
        # Check config.json exists and has correct structure
        config_path = exp_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["factor_name"] == result.factor_name
        assert config["periods"] == result.periods
        assert config["quantiles"] == result.quantiles
        assert "created_at" in config
        
        # Check plots directory exists
        plots_dir = exp_dir / "plots"
        assert plots_dir.exists()
        assert plots_dir.is_dir()
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/factors/test_analyzer.py::test_save_creates_correct_structure -v
```

預期輸出: `FAILED` - `AttributeError: 'FactorAnalysisResult' object has no attribute 'save'`

**Step 3: 實作 save() 方法**

在 `src/factorium/factors/analyzer.py` 的 `FactorAnalysisResult` 類中新增 save() 方法（約在第 56 行之前，`__repr__` 之後）：

```python
    def save(self, output_dir: str) -> None:
        """
        Save analysis results to directory with timestamp.
        
        Creates structure:
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
        
        Args:
            output_dir: Base directory for experiment outputs
        """
        from pathlib import Path
        from datetime import datetime
        import json
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{self.factor_name}"
        exp_path = Path(output_dir) / folder_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Create plots subdirectory
        plots_path = exp_path / "plots"
        plots_path.mkdir(exist_ok=True)
        
        # Save CSV files
        self.ic_series.to_csv(exp_path / "ic_series.csv")
        
        # Convert ic_summary dict to DataFrame for CSV
        ic_summary_df = pd.DataFrame([self.ic_summary])
        ic_summary_df.to_csv(exp_path / "ic_summary.csv", index=False)
        
        self.turnover_series.to_csv(exp_path / "turnover.csv", header=True)
        self.quantile_returns.to_csv(exp_path / "quantile_returns.csv")
        
        if self.cumulative_returns is not None:
            self.cumulative_returns.to_csv(exp_path / "cumulative_returns.csv")
        
        # Save config.json
        config = {
            "factor_name": self.factor_name,
            "periods": self.periods,
            "quantiles": self.quantiles,
            "created_at": datetime.now().isoformat(),
            "data_range": {
                "start": str(self.ic_series.index.min()),
                "end": str(self.ic_series.index.max()),
                "n_observations": len(self.ic_series),
            }
        }
        
        with open(exp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Results saved to {exp_path}")
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_analyzer.py::test_save_creates_correct_structure -v
```

預期輸出: `PASSED`

**Step 5: 提交變更**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): implement FactorAnalysisResult.save() method"
```

---

## Task 5: 新增 save() 繪圖功能

**Files:**
- Modify: `src/factorium/factors/analyzer.py:14-100`
- Test: `tests/factors/test_analyzer.py`

**目標:** 在 save() 方法中加入圖表輸出功能

**Step 1: 撰寫繪圖測試**

在 `tests/factors/test_analyzer.py` 新增測試：

```python
def test_save_generates_plots(sample_factor_and_prices):
    """Test that save() generates plot files."""
    factor, prices = sample_factor_and_prices
    analyzer = FactorAnalyzer(factor, prices, quantiles=5)
    result = analyzer.analyze(periods=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result.save(tmpdir)
        
        exp_dirs = list(Path(tmpdir).glob("*_*"))
        exp_dir = exp_dirs[0]
        plots_dir = exp_dir / "plots"
        
        # Check plot files exist
        assert (plots_dir / "ic_timeseries.png").exists()
        assert (plots_dir / "ic_distribution.png").exists()
        assert (plots_dir / "quantile_returns.png").exists()
        
        # cumulative_returns plot only if data exists
        if result.cumulative_returns is not None:
            assert (plots_dir / "cumulative_returns.png").exists()
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/factors/test_analyzer.py::test_save_generates_plots -v
```

預期輸出: `FAILED` - 圖表檔案不存在

**Step 3: 更新 save() 方法加入繪圖**

在 `src/factorium/factors/analyzer.py` 的 `save()` 方法中，於儲存 config.json 之前加入：

```python
        # Save plots
        from .plotting_analyzer import FactorAnalyzerPlotter
        import matplotlib.pyplot as plt
        
        plotter = FactorAnalyzerPlotter()
        
        # IC time series plot
        try:
            fig_ic_ts = plotter.plot_ic_ts(self.ic_series)
            fig_ic_ts.savefig(plots_path / "ic_timeseries.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ic_ts)
        except Exception as e:
            logger.warning(f"Failed to generate IC timeseries plot: {e}")
        
        # IC distribution plot
        try:
            fig_ic_hist = plotter.plot_ic_hist(self.ic_series)
            fig_ic_hist.savefig(plots_path / "ic_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ic_hist)
        except Exception as e:
            logger.warning(f"Failed to generate IC distribution plot: {e}")
        
        # Quantile returns plot
        try:
            fig_qret = plotter.plot_quantile_returns(self.quantile_returns)
            fig_qret.savefig(plots_path / "quantile_returns.png", dpi=150, bbox_inches="tight")
            plt.close(fig_qret)
        except Exception as e:
            logger.warning(f"Failed to generate quantile returns plot: {e}")
        
        # Cumulative returns plot (if available)
        if self.cumulative_returns is not None:
            try:
                fig_cumret = plotter.plot_cumulative_returns(self.cumulative_returns)
                fig_cumret.savefig(plots_path / "cumulative_returns.png", dpi=150, bbox_inches="tight")
                plt.close(fig_cumret)
            except Exception as e:
                logger.warning(f"Failed to generate cumulative returns plot: {e}")
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_analyzer.py::test_save_generates_plots -v
```

預期輸出: `PASSED`

**Step 5: 提交變更**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): add plot generation to save() method"
```

---

## Task 6: 重構 Factor.eval() 使用 FactorAnalyzer

**Files:**
- Modify: `src/factorium/factors/core.py:63-95`
- Test: `tests/factors/test_core.py` (or create new test file)

**目標:** 將 Factor.eval() 從使用 FactorEvaluator 改為使用 FactorAnalyzer

**Step 1: 撰寫新的 eval() 測試**

創建或修改 `tests/factors/test_factor_eval.py`：

```python
import pytest
import tempfile
from pathlib import Path
from factorium.factors.analyzer import FactorAnalysisResult

def test_factor_eval_returns_analysis_result(sample_factor_and_prices):
    """Test that Factor.eval() returns FactorAnalysisResult."""
    factor, prices = sample_factor_and_prices
    
    result = factor.eval(prices, periods=1, quantiles=5)
    
    assert isinstance(result, FactorAnalysisResult)
    assert result.factor_name == factor.name
    assert result.periods == 1
    assert result.quantiles == 5
    assert hasattr(result, "turnover_series")
    assert hasattr(result, "turnover_mean")

def test_factor_eval_with_output_dir(sample_factor_and_prices):
    """Test that Factor.eval() creates output when output_dir is specified."""
    factor, prices = sample_factor_and_prices
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = factor.eval(prices, periods=1, output_dir=tmpdir)
        
        # Check that experiment folder was created
        exp_dirs = list(Path(tmpdir).glob("*_*"))
        assert len(exp_dirs) == 1
        
        # Check config.json exists
        config_path = exp_dirs[0] / "config.json"
        assert config_path.exists()
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/factors/test_factor_eval.py -v
```

預期輸出: `FAILED` - 因為 eval() 還在使用 FactorEvaluator，回傳 Dict 而非 FactorAnalysisResult

**Step 3: 重構 Factor.eval() 方法**

修改 `src/factorium/factors/core.py` 中的 `eval()` 方法（約第 63-95 行）：

```python
    def eval(
        self,
        prices: Union["Factor", "AggBar"],
        periods: int = 1,
        quantiles: int = 5,
        output_dir: Optional[str] = None,
        price_col: str = "close",
        **kwargs,
    ) -> "FactorAnalysisResult":
        """
        Evaluate factor's predictive power (Evaluation Layer).

        Args:
            prices: Price data (Factor or AggBar)
            periods: Prediction horizon (currently only supports single int)
            quantiles: Number of quantiles for layer analysis (default 5)
            output_dir: Experiment output directory (creates timestamped folder if specified)
            price_col: Price column name (default "close")

        Returns:
            FactorAnalysisResult: Complete evaluation metrics including IC, ICIR, t-stat,
                                  turnover, quantile returns, and cumulative returns

        Example:
            >>> momentum = ts_returns(close, 20)
            >>> result = momentum.eval(prices, output_dir="./experiments")
            >>> print(result.ic_summary)
            {'mean_ic': 0.05, 'ic_ir': 1.2, 't_stat': 3.5, ...}
        """
        from .analyzer import FactorAnalyzer

        analyzer = FactorAnalyzer(factor=self, prices=prices, quantiles=quantiles)

        result = analyzer.analyze(price_col=price_col, periods=periods)

        # Save results if output directory specified
        if output_dir:
            result.save(output_dir)

        return result
```

**Step 4: 更新 import 和型別提示**

在 `src/factorium/factors/core.py` 頂部確保有正確的 import（約第 4 行附近）：

```python
from typing import Union, Optional, List, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..aggbar import AggBar
    from .analyzer import FactorAnalysisResult  # Add this line
```

**Step 5: 執行測試確認通過**

```bash
uv run pytest tests/factors/test_factor_eval.py -v
```

預期輸出: `PASSED`

**Step 6: 提交變更**

```bash
git add src/factorium/factors/core.py tests/factors/test_factor_eval.py
git commit -m "refactor(core): reimplement Factor.eval() using FactorAnalyzer"
```

---

## Task 7: 移除 FactorEvaluator 和 evaluation.py

**Files:**
- Delete: `src/factorium/factors/evaluation.py`
- Test: Run full test suite to ensure no breakage

**目標:** 移除舊的 FactorEvaluator 實作

**Step 1: 搜尋所有引用 FactorEvaluator 的地方**

```bash
uv run rg "FactorEvaluator" --type py
uv run rg "from.*evaluation import" --type py
```

預期輸出: 應該只在 `src/factorium/factors/evaluation.py` 中找到定義，沒有其他引用

**Step 2: 移除 evaluation.py**

```bash
rm src/factorium/factors/evaluation.py
```

**Step 3: 執行完整測試套件**

```bash
uv run pytest tests/factors/ -v
```

預期輸出: 所有測試 `PASSED`（如果有測試失敗，需要更新測試移除對 FactorEvaluator 的依賴）

**Step 4: 檢查是否有相關測試檔案需要移除**

```bash
ls tests/factors/test_evaluation.py 2>/dev/null && echo "Found" || echo "Not found"
```

如果存在，移除它：

```bash
rm tests/factors/test_evaluation.py
```

**Step 5: 提交變更**

```bash
git add -A
git commit -m "refactor: remove deprecated FactorEvaluator and evaluation.py"
```

---

## Task 8: 執行回歸測試與整合測試

**Files:**
- Test: All test files

**目標:** 確保重構沒有破壞現有功能

**Step 1: 執行完整因子測試套件**

```bash
uv run pytest tests/factors/ -v
```

預期輸出: 所有測試 `PASSED`

**Step 2: 執行完整測試套件（包含 backtest 等）**

```bash
uv run pytest tests/ -v -k "not slow"
```

預期輸出: 所有測試 `PASSED`

**Step 3: 如果有測試失敗，分析原因並修復**

常見問題：
- 測試期望 `Dict[str, Any]` 但現在回傳 `FactorAnalysisResult`
  - 解決：更新測試使用 `result.to_dict()` 或直接存取屬性
- 測試使用了 `FactorEvaluator` 的 private 方法
  - 解決：重寫測試使用 `FactorAnalyzer` 的對應方法

**Step 4: 提交修復（如有需要）**

```bash
git add tests/
git commit -m "test: update tests for new FactorAnalyzer API"
```

---

## Task 9: 驗證數值一致性

**Files:**
- Test: `tests/factors/test_turnover_consistency.py` (create new)

**目標:** 確保新的 Polars 實作與舊的 Pandas 實作數值一致

**Step 1: 創建數值一致性測試**

創建 `tests/factors/test_turnover_consistency.py`：

```python
import pytest
import numpy as np
import pandas as pd
from factorium.factors.analyzer import FactorAnalyzer

def test_turnover_numerical_consistency(sample_factor_and_prices):
    """
    Test that Polars-based turnover matches Pandas-based calculation.
    
    This is a regression test to ensure the migration maintains numerical accuracy.
    """
    factor, prices = sample_factor_and_prices
    
    # Calculate using new FactorAnalyzer (Polars)
    analyzer = FactorAnalyzer(factor, prices, quantiles=5)
    analyzer.prepare_data(periods=[1])
    turnover_new = analyzer.calculate_turnover()
    
    # Calculate using old method (Pandas) for comparison
    factor_pd = factor.to_pandas()
    pivoted = factor_pd.pivot(index="end_time", columns="symbol", values="factor")
    ranks = pivoted.rank(axis=1, pct=True)
    rank_autocorr = ranks.corrwith(ranks.shift(1), axis=1)
    turnover_old = 1 - rank_autocorr
    
    # Align indices (new uses start_time, old uses end_time)
    # We need to match them properly
    turnover_old_aligned = turnover_old.dropna()
    turnover_new_aligned = turnover_new.dropna()
    
    # Check that values are close (tolerance 1e-10)
    common_dates = turnover_new_aligned.index.intersection(turnover_old_aligned.index)
    if len(common_dates) > 0:
        np.testing.assert_allclose(
            turnover_new_aligned.loc[common_dates],
            turnover_old_aligned.loc[common_dates],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Turnover values differ between Polars and Pandas implementations"
        )
```

**Step 2: 執行數值一致性測試**

```bash
uv run pytest tests/factors/test_turnover_consistency.py -v
```

預期輸出: `PASSED` (如果失敗，需要調整 calculate_turnover 的實作)

**Step 3: 提交測試**

```bash
git add tests/factors/test_turnover_consistency.py
git commit -m "test: add numerical consistency test for turnover calculation"
```

---

## Task 10: 更新文檔

**Files:**
- Modify: `docs/user-guide/factor.md`
- Create: `docs/user-guide/evaluation-output.md` (optional)

**目標:** 更新文檔說明新的 API 用法

**Step 1: 更新 factor.md 中的 eval() 範例**

在 `docs/user-guide/factor.md` 中找到 `eval()` 相關章節，更新為：

```markdown
## 評估因子表現

使用 `eval()` 方法評估因子的預測能力：

```python
# 基本用法
result = momentum.eval(prices, periods=1, quantiles=5)

# 回傳 FactorAnalysisResult 物件
print(result)
# FactorAnalysisResult: momentum_20
#   Periods: 1, Quantiles: 5
#   Mean IC: 0.0523
#   IC Std: 0.1234
#   IC IR: 0.4238
#   Turnover: 0.3456

# 存取各項指標
print(result.ic_summary)  # {'mean_ic': 0.0523, 'ic_ir': 0.4238, ...}
print(result.turnover_mean)  # 0.3456

# 輸出實驗結果到資料夾
result = momentum.eval(prices, periods=1, output_dir="./experiments")
# 自動創建: experiments/20260129_143052_momentum_20/
```

### 實驗輸出結構

當指定 `output_dir` 時，會自動創建包含以下內容的資料夾：

- `config.json` - 實驗設定與 metadata
- `ic_series.csv` - IC 時間序列
- `ic_summary.csv` - IC 統計摘要
- `turnover.csv` - Turnover 時間序列
- `quantile_returns.csv` - 分層報酬
- `cumulative_returns.csv` - 累積報酬
- `plots/` - 視覺化圖表
  - `ic_timeseries.png`
  - `ic_distribution.png`
  - `quantile_returns.png`
  - `cumulative_returns.png`
```

**Step 2: 新增遷移說明（Breaking Changes）**

在文檔中加入 Breaking Changes 說明：

```markdown
### Breaking Changes (v0.x.x)

`Factor.eval()` 的回傳類型已從 `Dict[str, Any]` 改為 `FactorAnalysisResult`。

**遷移範例：**

舊版本：
```python
result = factor.eval(prices)
ic_mean = result["ic_mean"]
turnover = result["turnover_mean"]
```

新版本：
```python
result = factor.eval(prices)
ic_mean = result.ic_summary["mean_ic"]
turnover = result.turnover_mean

# 或使用 to_dict() 保持向後兼容
result_dict = result.to_dict()
```
```

**Step 3: 提交文檔更新**

```bash
git add docs/
git commit -m "docs: update Factor.eval() documentation for new API"
```

---

## Task 11: 最終驗證與清理

**Files:**
- All modified files

**目標:** 最終檢查與代碼清理

**Step 1: 執行完整測試套件**

```bash
uv run pytest tests/ -v
```

預期輸出: 所有測試 `PASSED`

**Step 2: 執行程式碼風格檢查（如果專案有配置）**

```bash
uv run ruff check src/factorium/factors/
```

修復任何風格問題

**Step 3: 檢查是否有未使用的 import**

```bash
uv run ruff check --select F401 src/factorium/factors/
```

移除未使用的 import

**Step 4: 執行型別檢查（如果專案使用 mypy）**

```bash
uv run mypy src/factorium/factors/analyzer.py
uv run mypy src/factorium/factors/core.py
```

修復型別錯誤

**Step 5: 最終提交**

```bash
git add -A
git commit -m "chore: final cleanup and formatting"
```

---

## 成功標準檢查清單

完成後驗證以下項目：

- [ ] ✅ `FactorAnalyzer.calculate_turnover()` 方法已實作並通過測試
- [ ] ✅ `FactorAnalysisResult` 包含 `turnover_series` 和 `turnover_mean` 欄位
- [ ] ✅ `FactorAnalyzer.analyze()` 整合 turnover 計算
- [ ] ✅ `FactorAnalysisResult.save()` 方法可正確輸出實驗結果
- [ ] ✅ `save()` 方法生成所有必要的圖表
- [ ] ✅ `Factor.eval()` 回傳 `FactorAnalysisResult`
- [ ] ✅ `Factor.eval()` 支援 `output_dir` 參數
- [ ] ✅ `FactorEvaluator` 和 `evaluation.py` 已移除
- [ ] ✅ Turnover 數值與 Pandas 版本一致（容差 < 1e-10）
- [ ] ✅ 所有測試通過
- [ ] ✅ 文檔已更新

---

## 附錄：測試數據準備

如果測試需要 `sample_factor_and_prices` fixture，確保在 `tests/conftest.py` 或 `tests/factors/conftest.py` 中定義：

```python
import pytest
import pandas as pd
import polars as pl
from factorium.factors.core import Factor
from factorium.aggbar import AggBar

@pytest.fixture
def sample_factor_and_prices():
    """Create sample factor and price data for testing."""
    # Create sample data with 3 symbols, 100 time periods
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            data.append({
                "start_time": date,
                "end_time": date + pd.Timedelta(days=1),
                "symbol": symbol,
                "factor": np.random.randn(),  # Random factor values
                "close": 100 + np.random.randn() * 10,  # Random prices
            })
    
    df = pd.DataFrame(data)
    
    # Create Factor and prices
    factor_df = df[["start_time", "end_time", "symbol", "factor"]]
    factor = Factor(pl.from_pandas(factor_df), name="test_factor")
    
    prices_df = df[["start_time", "end_time", "symbol", "close"]].rename(columns={"close": "factor"})
    prices = Factor(pl.from_pandas(prices_df), name="close")
    
    return factor, prices
```

---

## 估計時間

- Task 1: 20 分鐘
- Task 2: 10 分鐘
- Task 3: 10 分鐘
- Task 4: 25 分鐘
- Task 5: 15 分鐘
- Task 6: 15 分鐘
- Task 7: 10 分鐘
- Task 8: 15 分鐘
- Task 9: 15 分鐘
- Task 10: 15 分鐘
- Task 11: 10 分鐘

**總計：約 2.5-3 小時**
