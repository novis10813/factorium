# Multi-Horizon IC Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 讓 `FactorAnalyzer.analyze()` 支援 `periods: int | list[int]`，一次計算多個 forward horizon 的 IC，呈現 IC decay。

**Architecture:** 擴展現有 `analyze()` 方法，當傳入 `list[int]` 時回傳多 horizon 格式的 `FactorAnalysisResult`。向後相容：單一 `int` 時保持現有格式。

**Tech Stack:** Python, Polars, Pandas, pytest

**Related Issue:** https://github.com/novis10813/factorium/issues/4

---

## 設計概述

### API 變更

```python
# 現有 API（向後相容）
result = analyzer.analyze(periods=1)
result.ic_summary  # {"mean_ic": 0.05, "ic_std": 0.03, ...}

# 新 API（multi-horizon）
result = analyzer.analyze(periods=[1, 5, 20])
result.ic_summary  # {1: {"mean_ic": ...}, 5: {...}, 20: {...}}
```

### FactorAnalysisResult 結構

| 屬性 | 單 horizon (`periods=1`) | Multi-horizon (`periods=[1,5,20]`) |
|------|-------------------------|-----------------------------------|
| `periods` | `int` (1) | `list[int]` ([1,5,20]) |
| `ic_series` | `pd.DataFrame` (欄: period_1) | `pd.DataFrame` (欄: period_1, period_5, period_20) |
| `ic_summary` | `dict[str, float]` | `dict[int, dict[str, float]]` |
| `quantile_returns` | `pd.DataFrame` | `dict[int, pd.DataFrame]` |
| `cumulative_returns` | `pd.DataFrame \| None` | `dict[int, pd.DataFrame] \| None` |

---

## Task 1: 擴展 FactorAnalysisResult 型別定義

**Files:**
- Modify: `src/factorium/factors/analyzer.py:14-63`
- Test: `tests/factors/test_analyzer.py`

**Step 1: 寫失敗的測試**

在 `tests/factors/test_analyzer.py` 新增：

```python
def test_analyze_multi_horizon_returns_list_periods(sample_data):
    """analyze(periods=[1, 5]) 應回傳 list periods。"""
    from factorium.factors.analyzer import FactorAnalysisResult

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    result = analyzer.analyze(periods=[1, 5])

    assert isinstance(result, FactorAnalysisResult)
    assert result.periods == [1, 5]


def test_analyze_multi_horizon_ic_summary_structure(sample_data):
    """multi-horizon 時 ic_summary 應為 dict[int, dict]。"""
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    result = analyzer.analyze(periods=[1, 5])

    # ic_summary 應為 {1: {...}, 5: {...}}
    assert isinstance(result.ic_summary, dict)
    assert 1 in result.ic_summary
    assert 5 in result.ic_summary
    assert "mean_ic" in result.ic_summary[1]
    assert "mean_ic" in result.ic_summary[5]


def test_analyze_single_horizon_backward_compatible(sample_data):
    """單一 horizon 時保持向後相容。"""
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    result = analyzer.analyze(periods=1)

    # 單一 horizon 時 ic_summary 應為 {"mean_ic": ..., ...}
    assert isinstance(result.periods, int)
    assert result.periods == 1
    assert "mean_ic" in result.ic_summary
    assert isinstance(result.ic_summary["mean_ic"], float)
```

**Step 2: 執行測試確認失敗**

```bash
pytest tests/factors/test_analyzer.py::test_analyze_multi_horizon_returns_list_periods -v
pytest tests/factors/test_analyzer.py::test_analyze_multi_horizon_ic_summary_structure -v
```

預期：FAIL（目前 analyze() 只接受 int）

**Step 3: 修改型別定義**

修改 `src/factorium/factors/analyzer.py`：

```python
from typing import Union, List, Optional, Dict, Any

@dataclass
class FactorAnalysisResult:
    """..."""
    factor_name: str
    periods: Union[int, List[int]]
    quantiles: int
    ic_series: pd.DataFrame
    ic_summary: Union[Dict[str, float], Dict[int, Dict[str, float]]]
    turnover_series: pd.Series
    turnover_mean: float
    quantile_returns: Union[pd.DataFrame, Dict[int, pd.DataFrame]]
    cumulative_returns: Optional[Union[pd.DataFrame, Dict[int, pd.DataFrame]]] = None
```

**Step 4: 修改 analyze() 方法**

修改 `FactorAnalyzer.analyze()`：

```python
def analyze(self, price_col: str = "close", periods: Union[int, List[int]] = 1) -> FactorAnalysisResult:
    """
    Run full factor analysis.

    Args:
        price_col: Column name for prices.
        periods: Single period (int) or list of periods for multi-horizon analysis.

    Returns:
        FactorAnalysisResult with IC series, summary, and quantile returns.
    """
    # Normalize periods to list for internal processing
    periods_list = [periods] if isinstance(periods, int) else periods
    is_single = isinstance(periods, int)

    # Prepare data
    self.prepare_data(price_col=price_col, periods=periods_list)

    # Calculate IC
    ic_series = self.calculate_ic()
    ic_summary_df = self.calculate_ic_summary()

    # Build ic_summary
    if is_single:
        # 單一 horizon：向後相容格式
        col = f"period_{periods}"
        ic_summary = {
            "mean_ic": float(ic_summary_df.loc["mean", col]) if col in ic_summary_df.columns else 0.0,
            "ic_std": float(ic_summary_df.loc["std", col]) if col in ic_summary_df.columns else 0.0,
            "ic_ir": float(ic_summary_df.loc["ic_ir", col]) if col in ic_summary_df.columns else 0.0,
            "t-stat": float(ic_summary_df.loc["t-stat", col]) if col in ic_summary_df.columns else 0.0,
        }
    else:
        # Multi-horizon：新格式
        ic_summary = {}
        for p in periods_list:
            col = f"period_{p}"
            ic_summary[p] = {
                "mean_ic": float(ic_summary_df.loc["mean", col]) if col in ic_summary_df.columns else 0.0,
                "ic_std": float(ic_summary_df.loc["std", col]) if col in ic_summary_df.columns else 0.0,
                "ic_ir": float(ic_summary_df.loc["ic_ir", col]) if col in ic_summary_df.columns else 0.0,
                "t-stat": float(ic_summary_df.loc["t-stat", col]) if col in ic_summary_df.columns else 0.0,
            }

    # Calculate quantile returns
    if is_single:
        quantile_returns = self.calculate_quantile_returns(quantiles=self.quantiles, period=periods)
    else:
        quantile_returns = {
            p: self.calculate_quantile_returns(quantiles=self.quantiles, period=p)
            for p in periods_list
        }

    # Calculate cumulative returns
    try:
        if is_single:
            cumulative_returns = self.calculate_cumulative_returns(quantiles=self.quantiles, period=periods)
        else:
            cumulative_returns = {
                p: self.calculate_cumulative_returns(quantiles=self.quantiles, period=p)
                for p in periods_list
            }
    except Exception:
        cumulative_returns = None

    # Calculate turnover
    turnover_series = self.calculate_turnover()
    turnover_mean = float(turnover_series.mean())

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

**Step 5: 執行測試確認通過**

```bash
pytest tests/factors/test_analyzer.py -v -k "multi_horizon or backward_compatible"
```

預期：PASS

**Step 6: Commit**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): support multi-horizon IC analysis in analyze()"
```

---

## Task 2: 更新 FactorAnalysisResult.__repr__ 支援 multi-horizon

**Files:**
- Modify: `src/factorium/factors/analyzer.py:55-63`
- Test: `tests/factors/test_analyzer.py`

**Step 1: 寫失敗的測試**

```python
def test_repr_multi_horizon(sample_data):
    """multi-horizon __repr__ 應顯示所有 period 的 IC。"""
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    result = analyzer.analyze(periods=[1, 5])

    repr_str = repr(result)
    assert "my_factor" in repr_str
    assert "[1, 5]" in repr_str
    # 應該有多個 period 的 IC 資訊
    assert "Period 1" in repr_str or "period_1" in repr_str.lower()
```

**Step 2: 執行測試確認失敗**

```bash
pytest tests/factors/test_analyzer.py::test_repr_multi_horizon -v
```

**Step 3: 修改 __repr__**

```python
def __repr__(self) -> str:
    if isinstance(self.periods, int):
        # 單一 horizon
        ic = self.ic_summary
        return f"""FactorAnalysisResult: {self.factor_name}
  Periods: {self.periods}, Quantiles: {self.quantiles}
  Mean IC: {ic.get("mean_ic", 0):.4f}
  IC Std: {ic.get("ic_std", 0):.4f}
  IC IR: {ic.get("ic_ir", 0):.4f}
  Turnover: {self.turnover_mean:.4f}
"""
    else:
        # Multi-horizon
        lines = [f"FactorAnalysisResult: {self.factor_name}"]
        lines.append(f"  Periods: {self.periods}, Quantiles: {self.quantiles}")
        for p in self.periods:
            ic = self.ic_summary.get(p, {})
            lines.append(f"  Period {p}: IC={ic.get('mean_ic', 0):.4f}, IR={ic.get('ic_ir', 0):.4f}")
        lines.append(f"  Turnover: {self.turnover_mean:.4f}")
        return "\n".join(lines) + "\n"
```

**Step 4: 執行測試確認通過**

```bash
pytest tests/factors/test_analyzer.py::test_repr_multi_horizon -v
```

**Step 5: Commit**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): update __repr__ for multi-horizon display"
```

---

## Task 3: 更新 save() 支援 multi-horizon

**Files:**
- Modify: `src/factorium/factors/analyzer.py:65-168`
- Test: `tests/factors/test_analyzer.py`

**Step 1: 寫失敗的測試**

```python
def test_save_multi_horizon_creates_per_period_files(sample_data):
    """multi-horizon save 應為每個 period 建立 quantile_returns 檔案。"""
    import tempfile
    from pathlib import Path

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]
    analyzer = FactorAnalyzer(factor, prices, quantiles=2)
    result = analyzer.analyze(periods=[1, 5])

    with tempfile.TemporaryDirectory() as tmpdir:
        result.save(tmpdir)

        exp_dirs = list(Path(tmpdir).glob("*_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # 應有 quantile_returns_period_1.csv, quantile_returns_period_5.csv
        assert (exp_dir / "quantile_returns_period_1.csv").exists()
        assert (exp_dir / "quantile_returns_period_5.csv").exists()
```

**Step 2: 執行測試確認失敗**

```bash
pytest tests/factors/test_analyzer.py::test_save_multi_horizon_creates_per_period_files -v
```

**Step 3: 修改 save() 方法**

修改 `FactorAnalysisResult.save()` 來處理 multi-horizon：

```python
def save(self, output_dir: str) -> None:
    """Save analysis results to directory with timestamp."""
    from pathlib import Path
    from datetime import datetime
    import json
    import matplotlib.pyplot as plt
    from .plotting_analyzer import FactorAnalyzerPlotter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{self.factor_name}"
    exp_path = Path(output_dir) / folder_name
    exp_path.mkdir(parents=True, exist_ok=True)

    plots_path = exp_path / "plots"
    plots_path.mkdir(exist_ok=True)

    # Save IC series (always single DataFrame)
    self.ic_series.to_csv(exp_path / "ic_series.csv")

    # Save IC summary
    if isinstance(self.periods, int):
        ic_summary_df = pd.DataFrame([self.ic_summary])
    else:
        ic_summary_df = pd.DataFrame(self.ic_summary).T
        ic_summary_df.index.name = "period"
    ic_summary_df.to_csv(exp_path / "ic_summary.csv")

    # Save turnover
    self.turnover_series.to_csv(exp_path / "turnover.csv", header=True)

    # Save quantile returns
    if isinstance(self.periods, int):
        self.quantile_returns.to_csv(exp_path / "quantile_returns.csv")
    else:
        for p, df in self.quantile_returns.items():
            df.to_csv(exp_path / f"quantile_returns_period_{p}.csv")

    # Save cumulative returns
    if self.cumulative_returns is not None:
        if isinstance(self.periods, int):
            self.cumulative_returns.to_csv(exp_path / "cumulative_returns.csv")
        else:
            for p, df in self.cumulative_returns.items():
                df.to_csv(exp_path / f"cumulative_returns_period_{p}.csv")

    # Save plots
    plotter = FactorAnalyzerPlotter()

    try:
        fig_ic_ts = plotter.plot_ic_ts(self.ic_series)
        fig_ic_ts.savefig(plots_path / "ic_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig_ic_ts)
    except Exception as e:
        logger.warning(f"Failed to generate IC timeseries plot: {e}")

    try:
        fig_ic_hist = plotter.plot_ic_hist(self.ic_series)
        fig_ic_hist.savefig(plots_path / "ic_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig_ic_hist)
    except Exception as e:
        logger.warning(f"Failed to generate IC distribution plot: {e}")

    # Quantile returns plots
    if isinstance(self.periods, int):
        try:
            fig_qret = plotter.plot_quantile_returns(self.quantile_returns)
            fig_qret.savefig(plots_path / "quantile_returns.png", dpi=150, bbox_inches="tight")
            plt.close(fig_qret)
        except Exception as e:
            logger.warning(f"Failed to generate quantile returns plot: {e}")
    else:
        for p, df in self.quantile_returns.items():
            try:
                fig_qret = plotter.plot_quantile_returns(df)
                fig_qret.savefig(plots_path / f"quantile_returns_period_{p}.png", dpi=150, bbox_inches="tight")
                plt.close(fig_qret)
            except Exception as e:
                logger.warning(f"Failed to generate quantile returns plot for period {p}: {e}")

    # Cumulative returns plots
    if self.cumulative_returns is not None:
        if isinstance(self.periods, int):
            try:
                fig_cumret = plotter.plot_cumulative_returns(self.cumulative_returns)
                fig_cumret.savefig(plots_path / "cumulative_returns.png", dpi=150, bbox_inches="tight")
                plt.close(fig_cumret)
            except Exception as e:
                logger.warning(f"Failed to generate cumulative returns plot: {e}")
        else:
            for p, df in self.cumulative_returns.items():
                try:
                    fig_cumret = plotter.plot_cumulative_returns(df)
                    fig_cumret.savefig(plots_path / f"cumulative_returns_period_{p}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig_cumret)
                except Exception as e:
                    logger.warning(f"Failed to generate cumulative returns plot for period {p}: {e}")

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
        },
    }

    with open(exp_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Results saved to {exp_path}")
```

**Step 4: 執行測試確認通過**

```bash
pytest tests/factors/test_analyzer.py::test_save_multi_horizon_creates_per_period_files -v
```

**Step 5: Commit**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): update save() to support multi-horizon output"
```

---

## Task 4: 新增 IC decay 繪圖方法

**Files:**
- Modify: `src/factorium/factors/analyzer.py`
- Modify: `src/factorium/factors/plotting_analyzer.py`
- Test: `tests/factors/test_analyzer.py`

**Step 1: 寫失敗的測試**

```python
def test_plot_ic_decay(sample_data):
    """multi-horizon 應支援 plot_ic_decay() 繪製 IC decay 曲線。"""
    import matplotlib.figure as mpl_figure

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1, 3, 5])

    fig = analyzer.plot_ic_decay(periods=[1, 3, 5])
    assert isinstance(fig, mpl_figure.Figure)
```

**Step 2: 執行測試確認失敗**

```bash
pytest tests/factors/test_analyzer.py::test_plot_ic_decay -v
```

**Step 3: 在 FactorAnalyzerPlotter 新增 plot_ic_decay**

修改 `src/factorium/factors/plotting_analyzer.py`：

```python
def plot_ic_decay(self, ic_summary: Dict[int, Dict[str, float]]) -> mpl_figure.Figure:
    """
    Plot IC decay curve across horizons.

    Args:
        ic_summary: Dict mapping period -> {"mean_ic": float, "ic_ir": float, ...}

    Returns:
        matplotlib Figure
    """
    periods = sorted(ic_summary.keys())
    mean_ics = [ic_summary[p]["mean_ic"] for p in periods]
    ic_irs = [ic_summary[p]["ic_ir"] for p in periods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean IC decay
    ax1.bar(range(len(periods)), mean_ics, color="steelblue", alpha=0.7)
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels([str(p) for p in periods])
    ax1.set_xlabel("Horizon (periods)")
    ax1.set_ylabel("Mean IC")
    ax1.set_title("IC Decay by Horizon")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # IC IR decay
    ax2.bar(range(len(periods)), ic_irs, color="darkorange", alpha=0.7)
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels([str(p) for p in periods])
    ax2.set_xlabel("Horizon (periods)")
    ax2.set_ylabel("IC IR")
    ax2.set_title("IC Information Ratio by Horizon")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
```

**Step 4: 在 FactorAnalyzer 新增 plot_ic_decay 方法**

修改 `src/factorium/factors/analyzer.py`：

```python
def plot_ic_decay(self, periods: Optional[List[int]] = None, method: str = "rank") -> mpl_figure.Figure:
    """
    Plot IC decay curve across multiple horizons.

    Args:
        periods: List of periods to plot. If None, uses all available periods.
        method: 'rank' for Spearman, 'normal' for Pearson.

    Returns:
        matplotlib Figure
    """
    from .plotting_analyzer import FactorAnalyzerPlotter

    ic_summary_df = self.calculate_ic_summary(method=method)

    # Build ic_summary dict for plotting
    if periods is None:
        # Extract periods from available columns
        periods = [int(c.replace("period_", "")) for c in ic_summary_df.columns if c.startswith("period_")]

    ic_summary = {}
    for p in periods:
        col = f"period_{p}"
        if col in ic_summary_df.columns:
            ic_summary[p] = {
                "mean_ic": float(ic_summary_df.loc["mean", col]),
                "ic_ir": float(ic_summary_df.loc["ic_ir", col]),
            }

    if not ic_summary:
        raise ValueError("No IC data available for the specified periods.")

    plotter = FactorAnalyzerPlotter()
    return plotter.plot_ic_decay(ic_summary)
```

**Step 5: 執行測試確認通過**

```bash
pytest tests/factors/test_analyzer.py::test_plot_ic_decay -v
```

**Step 6: Commit**

```bash
git add src/factorium/factors/analyzer.py src/factorium/factors/plotting_analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): add plot_ic_decay() for multi-horizon visualization"
```

---

## Task 5: 更新 save() 以包含 IC decay 圖

**Files:**
- Modify: `src/factorium/factors/analyzer.py:65-168`
- Test: `tests/factors/test_analyzer.py`

**Step 1: 寫失敗的測試**

```python
def test_save_multi_horizon_includes_ic_decay_plot(sample_data):
    """multi-horizon save 應包含 ic_decay.png。"""
    import tempfile
    from pathlib import Path

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]
    analyzer = FactorAnalyzer(factor, prices, quantiles=2)
    result = analyzer.analyze(periods=[1, 3, 5])

    with tempfile.TemporaryDirectory() as tmpdir:
        result.save(tmpdir)

        exp_dirs = list(Path(tmpdir).glob("*_*"))
        exp_dir = exp_dirs[0]
        plots_dir = exp_dir / "plots"

        assert (plots_dir / "ic_decay.png").exists()
```

**Step 2: 修改 save() 新增 IC decay 圖**

在 save() 方法中，當 `isinstance(self.periods, list)` 時新增：

```python
# IC decay plot (multi-horizon only)
if isinstance(self.periods, list) and len(self.periods) > 1:
    try:
        fig_decay = plotter.plot_ic_decay(self.ic_summary)
        fig_decay.savefig(plots_path / "ic_decay.png", dpi=150, bbox_inches="tight")
        plt.close(fig_decay)
    except Exception as e:
        logger.warning(f"Failed to generate IC decay plot: {e}")
```

**Step 3: 執行測試確認通過**

```bash
pytest tests/factors/test_analyzer.py::test_save_multi_horizon_includes_ic_decay_plot -v
```

**Step 4: Commit**

```bash
git add src/factorium/factors/analyzer.py tests/factors/test_analyzer.py
git commit -m "feat(analyzer): add IC decay plot to save() for multi-horizon"
```

---

## Task 6: 執行完整測試套件

**Step 1: 執行所有 analyzer 測試**

```bash
pytest tests/factors/test_analyzer.py -v
```

預期：所有測試通過

**Step 2: 執行型別檢查（如有）**

```bash
uv run mypy src/factorium/factors/analyzer.py --ignore-missing-imports
```

**Step 3: 執行 linting**

```bash
uv run ruff check src/factorium/factors/analyzer.py
uv run ruff format src/factorium/factors/analyzer.py
```

**Step 4: 最終 Commit**

```bash
git add .
git commit -m "chore: format and lint multi-horizon IC implementation"
```

---

## 驗收標準

1. `analyze(periods=1)` 行為與之前完全一致（向後相容）
2. `analyze(periods=[1, 5, 20])` 回傳正確的 multi-horizon 格式
3. `ic_summary` 結構：單一 int 時為 `dict[str, float]`，list 時為 `dict[int, dict[str, float]]`
4. `quantile_returns` 和 `cumulative_returns` 結構類似
5. `save()` 正確處理兩種格式
6. `plot_ic_decay()` 可繪製 IC decay 曲線
7. 所有現有測試通過
