"""
Factor analysis and backtest report generation.

Combines factor analysis and backtest results into comprehensive reports.
"""

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ..factors.analyzer import FactorAnalysisResult
    from .session import ResearchSession

from ..backtest.vectorized import BacktestResult
from ..factors.core import Factor


class FactorReport:
    """
    Comprehensive report combining factor analysis and backtest results.

    Args:
        factor: The factor being analyzed
        analysis: Analysis results from FactorAnalyzer
        backtest: Backtest results from VectorizedBacktester

    Example:
        >>> from factorium.research import ResearchSession, FactorReport
        >>> session = ResearchSession(data)
        >>> signal = session.factor("close").cs_rank()
        >>> report = FactorReport.generate(session, signal)
        >>> print(report)
    """

    @classmethod
    def generate(
        cls,
        session: "ResearchSession",
        factor: "Factor",
        price_col: str = "close",
        quantiles: int = 5,
        **backtest_kwargs,
    ) -> "FactorReport":
        """
        Generate report by running analysis and backtest automatically.

        Args:
            session: ResearchSession with data
            factor: Factor to analyze
            price_col: Price column for analysis
            quantiles: Quantiles for analysis
            **backtest_kwargs: Additional args for backtest (neutralization, etc.)

        Returns:
            FactorReport with complete analysis and backtest results

        Example:
            >>> from factorium.research import ResearchSession, FactorReport
            >>> session = ResearchSession(data)
            >>> signal = session.factor("close").cs_rank()
            >>> report = FactorReport.generate(session, signal)
            >>> print(report)
        """
        # Run analysis
        analysis = session.analyze(factor, price_col=price_col, quantiles=quantiles)

        # Run backtest
        backtest = session.backtest(factor, **backtest_kwargs)

        return cls(factor, analysis, backtest)

    def __init__(
        self,
        factor: Factor,
        analysis: Union[dict[str, Any], "FactorAnalysisResult"],
        backtest: BacktestResult,
    ):
        self.factor = factor
        self.analysis = analysis
        self.backtest = backtest

    def summary(self) -> dict[str, Any]:
        """
        Generate summary combining analysis and backtest metrics.

        Returns:
            Dictionary with factor name, IC summary, and backtest metrics
        """
        if hasattr(self.analysis, "to_dict"):
            analysis_dict = self.analysis.to_dict()
        else:
            analysis_dict = self.analysis

        return {
            "factor_name": self.factor.name,
            "ic_summary": analysis_dict.get("ic_summary", {}),
            "backtest_metrics": self.backtest.metrics,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        if hasattr(self.analysis, "to_dict"):
            analysis_dict = self.analysis.to_dict()
        else:
            analysis_dict = self.analysis

        return {
            "factor_name": self.factor.name,
            "analysis": analysis_dict,
            "metrics": self.backtest.metrics,
            "equity_curve": self.backtest.equity_curve.to_dict(),
            "returns": self.backtest.returns.to_dict(),
        }

    def __repr__(self) -> str:
        """String representation of report."""
        summary = self.summary()
        ic_summary = summary["ic_summary"]
        metrics = summary["backtest_metrics"]

        def fmt_float(val: Any, fmt: str) -> str:
            if isinstance(val, (int, float)):
                return f"{val:{fmt}}"
            return "N/A"

        # ic_summary is now dict[int, dict[str, float]], get first period's stats
        if ic_summary:
            first_period = next(iter(ic_summary.keys()))
            ic = ic_summary.get(first_period, {})
        else:
            ic = {}

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
