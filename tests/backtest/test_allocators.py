# tests/backtest/test_allocators.py
import polars as pl

from factorium.backtest.allocators import MarketNeutralAllocator


class TestMarketNeutralAllocator:
    def test_weights_sum_to_zero(self):
        """Market neutral weights should sum to zero per group."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum()) < 1e-10

    def test_abs_weights_sum_to_one(self):
        """Absolute weights should sum to one per group."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10

    def test_null_signal_gets_zero_weight(self):
        """Null signals should produce zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_identical_signals_produce_zero_weights(self):
        """Identical signals should all be zero after demeaning."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert result["weight"].abs().sum() < 1e-10

    def test_multiple_groups(self):
        """Each group should be independently neutralized."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 5.0, 15.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        for t in [1000, 2000]:
            subset = result.filter(pl.col("end_time") == t)["weight"]
            assert abs(subset.sum()) < 1e-10
            assert abs(subset.abs().sum() - 1.0) < 1e-10

    def test_renormalize_restores_invariants(self):
        """renormalize should restore sum=0, abs_sum=1 after perturbation."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.3, 0.1, -0.05, -0.1],  # broken invariants
        })
        result = MarketNeutralAllocator().renormalize(df, "end_time")
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10

    def test_renormalize_all_zero_stays_zero(self):
        """All-zero weights should stay zero after renormalize."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.0, 0.0, 0.0],
        })
        result = MarketNeutralAllocator().renormalize(df, "end_time")
        assert result["weight"].abs().sum() < 1e-10
