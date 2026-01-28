import pytest
import polars as pl
from factorium import AggBar
from factorium.research import ResearchSession, FactorReport


class TestFactorReport:
    """Tests for FactorReport."""

    @pytest.fixture
    def sample_data(self):
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 30, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_report_combines_analysis_and_backtest(self, sample_data):
        """FactorReport should combine analysis and backtest results."""
        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        analysis = session.analyze(signal)
        backtest = session.backtest(signal)

        report = FactorReport(signal, analysis, backtest)

        summary = report.summary()
        assert "factor_name" in summary
        assert "ic_summary" in summary
        assert "backtest_metrics" in summary

    def test_report_to_dict(self, sample_data):
        """to_dict() should return comprehensive dictionary."""
        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        analysis = session.analyze(signal)
        backtest = session.backtest(signal)

        report = FactorReport(signal, analysis, backtest)
        result = report.to_dict()

        assert "factor_name" in result
        assert "analysis" in result
        assert "metrics" in result
        assert "equity_curve" in result
        assert "returns" in result

    def test_report_repr(self, sample_data):
        """__repr__ should display readable summary."""
        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        analysis = session.analyze(signal)
        backtest = session.backtest(signal)

        report = FactorReport(signal, analysis, backtest)
        repr_str = repr(report)

        assert "FactorReport" in repr_str
        assert "IC Summary" in repr_str
        assert "Backtest Metrics" in repr_str

    def test_generate_automates_workflow(self, sample_data):
        """generate() should run both analysis and backtest."""
        from factorium.research import ResearchSession

        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        # Use generate instead of manual steps
        report = FactorReport.generate(session, signal)

        assert isinstance(report, FactorReport)
        assert report.analysis is not None
        assert report.backtest is not None
        # Check if it's FactorAnalysisResult or dict (backward compatibility)
        if hasattr(report.analysis, "ic_summary"):
            assert report.analysis.ic_summary is not None
        else:
            assert "ic_summary" in report.analysis
