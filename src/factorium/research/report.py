"""
Factor analysis and backtest report generation.

Combines factor analysis and backtest results into comprehensive reports.
"""

from typing import Dict, Any, Optional
import polars as pl
import pandas as pd

from ..factors.core import Factor
from ..backtest.vectorized import BacktestResult


class FactorReport:
    """
    Comprehensive report combining factor analysis and backtest results.

    Args:
        factor: The factor being analyzed
        analysis: Analysis results from FactorAnalyzer
        backtest: Backtest results from VectorizedBacktester

    Example:
        >>> from factorium.research import ResearchSession
        >>> session = ResearchSession(data)
        >>> signal = session.factor("close").cs_rank()
        >>> analysis = session.analyze(signal)
        >>> backtest = session.backtest(signal)
        >>> report = FactorReport(signal, analysis, backtest)
        >>> report.summary()
    """

    def __init__(
        self,
        factor: Factor,
        analysis: Dict[str, Any],
        backtest: BacktestResult,
    ):
        self.factor = factor
        self.analysis = analysis
        self.backtest = backtest

    def summary(self) -> Dict[str, Any]:
        """
        Generate summary combining analysis and backtest metrics.

        Returns:
            Dictionary with factor name, IC summary, and backtest metrics
        """
        return {
            "factor_name": self.factor.name,
            "ic_summary": self.analysis.get("ic_summary", {}),
            "backtest_metrics": self.backtest.metrics,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "factor_name": self.factor.name,
            "analysis": self.analysis,
            "metrics": self.backtest.metrics,
            "equity_curve": self.backtest.equity_curve.to_dict(),
            "returns": self.backtest.returns.to_dict(),
        }

    def __repr__(self) -> str:
        """String representation of report."""
        summary = self.summary()
        ic = summary["ic_summary"]
        metrics = summary["backtest_metrics"]

        def fmt_float(val: Any, fmt: str) -> str:
            if isinstance(val, (int, float)):
                return f"{val:{fmt}}"
            return "N/A"

        return f"""FactorReport: {summary["factor_name"]}
IC Summary:
  Mean IC: {fmt_float(ic.get("mean_ic"), ".4f")}
  IC Std: {fmt_float(ic.get("ic_std"), ".4f")}
  
Backtest Metrics:
  Total Return: {fmt_float(metrics.get("total_return"), ".2%")}
  Annual Return: {fmt_float(metrics.get("annual_return"), ".2%")}
  Sharpe Ratio: {fmt_float(metrics.get("sharpe_ratio"), ".2f")}
  Max Drawdown: {fmt_float(metrics.get("max_drawdown"), ".2%")}
"""
