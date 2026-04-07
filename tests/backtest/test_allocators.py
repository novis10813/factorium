# tests/backtest/test_allocators.py
import polars as pl
import pytest

from factorium.backtest.allocators import (
    LongOnlyAllocator,
    MarketNeutralAllocator,
    TopNAllocator,
)


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


class TestLongOnlyAllocator:
    def test_weights_sum_to_one(self):
        """Long-only weights should sum to one."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum() - 1.0) < 1e-10

    def test_all_weights_non_negative(self):
        """All weights should be >= 0."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [1.0, 2.0, 3.0, -1.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert (result["weight"] >= -1e-10).all()

    def test_negative_signals_get_zero_weight(self):
        """Negative signals should get zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, -2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_proportional_to_signal(self):
        """Weights should be proportional to positive signal values."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        weights = result["weight"].to_list()
        # B should be 2x A, C should be 3x A
        assert abs(weights[1] / weights[0] - 2.0) < 1e-10
        assert abs(weights[2] / weights[0] - 3.0) < 1e-10

    def test_all_negative_signals_produce_zero_weights(self):
        """If all signals are negative, all weights should be zero."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [-1.0, -2.0, -3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"].abs().sum() < 1e-10

    def test_null_signal_gets_zero_weight(self):
        """Null signals should produce zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_renormalize_clips_negatives_and_sums_to_one(self):
        """renormalize should clip negatives and scale to sum=1."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.5, 0.3, -0.1],
        })
        result = LongOnlyAllocator().renormalize(df, "end_time")
        assert (result["weight"] >= -1e-10).all()
        assert abs(result["weight"].sum() - 1.0) < 1e-10

    def test_renormalize_all_zero_stays_zero(self):
        """All-zero weights should stay zero after renormalize."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.0, 0.0, 0.0],
        })
        result = LongOnlyAllocator().renormalize(df, "end_time")
        assert result["weight"].abs().sum() < 1e-10


class TestTopNAllocator:
    def test_long_only_top_n_equal_weight(self):
        """Top N long-only should give equal weight 1/N to top N."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=2).allocate(df, "signal", "end_time")
        weights = result.sort("symbol")["weight"].to_list()
        # B (5.0) and D (4.0) are top 2
        assert weights[1] == pytest.approx(0.5)  # B
        assert weights[3] == pytest.approx(0.5)  # D
        assert weights[0] == 0.0  # A
        assert weights[2] == 0.0  # C
        assert weights[4] == 0.0  # E

    def test_long_short_top_n(self):
        """Long-short mode: top N get +1/N, bottom N get -1/N."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=2, long_short=True).allocate(df, "signal", "end_time")
        weights = result.sort("symbol")["weight"].to_list()
        # Top 2: B (5.0), D (4.0) → +0.5
        # Bottom 2: A (1.0), E (2.0) → -0.5
        assert weights[1] == pytest.approx(0.5)   # B
        assert weights[3] == pytest.approx(0.5)   # D
        assert weights[0] == pytest.approx(-0.5)  # A
        assert weights[4] == pytest.approx(-0.5)  # E
        assert weights[2] == 0.0                   # C (middle)

    def test_long_short_weights_sum_to_zero(self):
        """Long-short weights should sum to zero."""
        df = pl.DataFrame({
            "end_time": [1000] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "signal": [1.0, 6.0, 3.0, 5.0, 2.0, 4.0],
        })
        result = TopNAllocator(n=2, long_short=True).allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum()) < 1e-10

    def test_long_only_weights_sum_to_one(self):
        """Long-only top-N weights should sum to one."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=3).allocate(df, "signal", "end_time")
        non_zero = result.filter(pl.col("weight") != 0.0)
        assert abs(non_zero["weight"].sum() - 1.0) < 1e-10

    def test_null_signal_excluded(self):
        """Null signals should not be selected in top N."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [1.0, None, 3.0, 2.0],
        })
        result = TopNAllocator(n=2).allocate(df, "signal", "end_time")
        assert result.filter(pl.col("symbol") == "B")["weight"][0] == 0.0

    def test_renormalize_restores_equal_weight(self):
        """renormalize should restore equal-weight among non-zero positions."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.4, 0.3, 0.0, 0.0],  # perturbed from equal
        })
        result = TopNAllocator(n=2).renormalize(df, "end_time")
        non_zero = result.filter(pl.col("weight") != 0.0)["weight"].to_list()
        assert len(non_zero) == 2
        assert abs(non_zero[0] - non_zero[1]) < 1e-10
        assert abs(sum(non_zero) - 1.0) < 1e-10

    def test_renormalize_long_short_equal_weight(self):
        """renormalize long-short should restore equal-weight on both sides."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.4, 0.3, -0.2, -0.1],  # perturbed
        })
        result = TopNAllocator(n=2, long_short=True).renormalize(df, "end_time")
        pos = result.filter(pl.col("weight") > 0)["weight"].to_list()
        neg = result.filter(pl.col("weight") < 0)["weight"].to_list()
        assert abs(pos[0] - pos[1]) < 1e-10
        assert abs(neg[0] - neg[1]) < 1e-10
        assert abs(sum(pos) + sum(neg)) < 1e-10
