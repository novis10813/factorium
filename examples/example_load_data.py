#!/usr/bin/env python3
"""Example: Load 10 crypto symbols for 1 month and compute factors."""

import time
from datetime import datetime, timedelta

from factorium.data import BinanceDataLoader
from factorium.factors import Factor

SYMBOLS = [
    "BTCUSDT",
    "CAKEUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
]


def main():
    loader = BinanceDataLoader()

    print(f"Loading {len(SYMBOLS)} symbols, 30 days, 1-minute bars...")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("-" * 60)

    start = time.perf_counter()

    agg = loader.load_aggbar_fast(
        symbols=SYMBOLS,
        data_type="aggTrades",
        market_type="futures",
        futures_type="um",
        days=30,
        interval_ms=60_000,
        use_cache=True,
    )

    load_time = time.perf_counter() - start
    print(f"Loaded {len(agg):,} bars in {load_time:.2f}s")
    print(f"Symbols: {agg.symbols}")
    print(f"Columns: {agg.cols}")
    print("-" * 60)

    close = Factor(agg, "close")
    volume = Factor(agg, "volume")

    print("Computing factors...")
    start = time.perf_counter()

    returns = close.ts_delta(1) / close.ts_shift(1)
    ma_20 = close.ts_mean(20)
    volatility = returns.ts_std(20)
    momentum = close.ts_delta(60) / close.ts_shift(60)
    volume_ma = volume.ts_mean(20)
    volume_ratio = volume / volume_ma

    ma_fast = close.ts_mean(10)
    ma_slow = close.ts_mean(30)
    ma_cross = ma_fast - ma_slow

    rank_momentum = momentum.cs_rank()
    zscore_volume = volume_ratio.cs_zscore()

    factor_time = time.perf_counter() - start
    print(f"Computed 8 factors in {factor_time:.2f}s")
    print("-" * 60)

    print("\n[Sample: Momentum Cross-Sectional Rank]")
    print(rank_momentum.data.dropna().tail(20).to_string())

    print("\n[Sample: Volume Z-Score]")
    print(zscore_volume.data.dropna().tail(20).to_string())

    print("\n[Sample: MA Crossover]")
    print(ma_cross.data.dropna().tail(20).to_string())

    print("-" * 60)
    print(f"Total time: {load_time + factor_time:.2f}s")
    print(f"  - Data loading: {load_time:.2f}s")
    print(f"  - Factor computation: {factor_time:.2f}s")


if __name__ == "__main__":
    main()
