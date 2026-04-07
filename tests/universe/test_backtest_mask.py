import polars as pl

from factorium import AggBar
from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.pipeline import AlphaPipeline
from factorium.backtest.vectorized import VectorizedBacktester


def _make_prices() -> AggBar:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(6):
        ts = base_ts + i * 3_600_000
        rows.extend(
            [
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "A",
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.0 + i,
                    "volume": 1_000.0,
                    "in_universe": True,
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "B",
                    "open": 50.0 + i,
                    "high": 51.0 + i,
                    "low": 49.0 + i,
                    "close": 50.0 + i,
                    "volume": 1_000.0,
                    "in_universe": True,
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "C",
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.0 + i,
                    "volume": 1_000.0,
                    "in_universe": False,
                },
            ]
        )
    return AggBar(pl.DataFrame(rows))


def _make_signal() -> pl.DataFrame:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(6):
        ts = base_ts + i * 3_600_000
        rows.extend(
            [
                {"start_time": ts, "end_time": ts + 3_600_000, "symbol": "A", "factor": 3.0 + i},
                {"start_time": ts, "end_time": ts + 3_600_000, "symbol": "B", "factor": 2.0 + i},
                {"start_time": ts, "end_time": ts + 3_600_000, "symbol": "C", "factor": 1.0 + i},
            ]
        )
    return pl.DataFrame(rows)


def test_backtester_mask_sets_outside_symbols_weight_to_zero() -> None:
    prices = _make_prices()
    signal = _make_signal()

    bt = VectorizedBacktester(
        prices=prices,
        signal=signal,
        mask="in_universe",
        pipeline=AlphaPipeline(allocator=LongOnlyAllocator()),
    )
    combined = bt._prepare_data()
    weighted = bt._calculate_weights(combined)

    c_weights = weighted.filter((pl.col("symbol") == "C") & (pl.col("prev_signal").is_not_null()))["weight"].to_list()
    assert set(c_weights) == {0.0}


def test_backtester_mask_applies_before_market_neutralization() -> None:
    prices = _make_prices()
    signal = _make_signal()

    bt = VectorizedBacktester(
        prices=prices,
        signal=signal,
        mask="in_universe",
        pipeline=AlphaPipeline(allocator=MarketNeutralAllocator()),
    )
    weighted = bt._calculate_weights(bt._prepare_data())

    sample_time = sorted(set(weighted["end_time"].to_list()))[1]
    sample = weighted.filter(pl.col("end_time") == sample_time)

    b_weight = sample.filter(pl.col("symbol") == "B")["weight"].to_list()[0]
    c_weight = sample.filter(pl.col("symbol") == "C")["weight"].to_list()[0]
    assert b_weight < 0
    assert c_weight == 0.0


def test_backtester_without_mask_keeps_backward_compatible_flow() -> None:
    prices = _make_prices()
    signal = _make_signal()

    bt = VectorizedBacktester(prices=prices, signal=signal)
    result = bt.run()

    assert len(result.equity_curve) > 0
