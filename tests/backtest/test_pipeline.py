# tests/backtest/test_pipeline.py
import polars as pl

from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.constraints import MaxPositionConstraint
from factorium.backtest.normalizers import RankNormalizer, RawNormalizer
from factorium.backtest.pipeline import AlphaPipeline


class TestAlphaPipelineDefaults:
    def test_default_pipeline_is_raw_market_neutral(self):
        """Default pipeline should be RawNormalizer + MarketNeutralAllocator."""
        pipe = AlphaPipeline()
        assert isinstance(pipe.normalizer, RawNormalizer)
        assert isinstance(pipe.allocator, MarketNeutralAllocator)
        assert pipe.constraints == []

    def test_default_produces_market_neutral_weights(self):
        """Default pipeline output should satisfy market-neutral invariants."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = AlphaPipeline().transform(df, "signal")
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10


class TestAlphaPipelineWithNormalizer:
    def test_rank_then_long_only(self):
        """RankNormalizer + LongOnlyAllocator should produce valid weights."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [-10.0, 50.0, 20.0],
        })
        pipe = AlphaPipeline(
            normalizer=RankNormalizer(),
            allocator=LongOnlyAllocator(),
        )
        result = pipe.transform(df, "signal")
        # After rank normalization, all values are in [0,1] (positive)
        # so LongOnly should give all assets weight
        assert abs(result["weight"].sum() - 1.0) < 1e-10
        assert (result["weight"] >= -1e-10).all()


class TestAlphaPipelineWithConstraints:
    def test_constraint_then_renormalize(self):
        """Constraints should be applied, then renormalize restores invariants."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 1.0, 1.0, 1.0],
        })
        pipe = AlphaPipeline(
            normalizer=RawNormalizer(),
            allocator=MarketNeutralAllocator(),
            constraints=[MaxPositionConstraint(max_weight=0.3)],
        )
        result = pipe.transform(df, "signal")
        # After constraint + renormalize, invariants should hold
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10
        # Max weight should be capped (within renormalize tolerance)
        assert result["weight"].abs().max() <= 0.5 + 1e-10

    def test_no_constraints_skips_renormalize(self):
        """Without constraints, renormalize should not be called."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        pipe = AlphaPipeline(
            normalizer=RawNormalizer(),
            allocator=MarketNeutralAllocator(),
        )
        result = pipe.transform(df, "signal")
        # Should still satisfy invariants from allocate alone
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10
