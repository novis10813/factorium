import argparse
import time
import tracemalloc

import numpy as np
import pandas as pd

from factorium import Factor


def build_dataset(n_symbols: int, n_periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    symbols = [f"S{idx:04d}" for idx in range(n_symbols)]

    end_time = np.tile(dates.values, n_symbols)
    start_time = end_time
    symbol_col = np.repeat(np.array(symbols, dtype=object), n_periods)
    factor_vals = rng.standard_normal(n_symbols * n_periods)

    return pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "symbol": symbol_col,
            "factor": factor_vals,
        }
    )


def run_benchmark(n_symbols: int, n_periods: int) -> None:
    df = build_dataset(n_symbols, n_periods)
    factor = Factor(df, name="bench")

    build_start = time.perf_counter()
    result = factor.ts_mean(20).cs_rank().ts_zscore(10).ts_rank(5)
    build_time = time.perf_counter() - build_start

    tracemalloc.start()
    collect_start = time.perf_counter()
    collected = result.data
    collect_time = time.perf_counter() - collect_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Benchmark Summary")
    print(f"symbols: {n_symbols}")
    print(f"periods: {n_periods}")
    print(f"rows: {len(collected)}")
    print(f"lazy_build_sec: {build_time:.4f}")
    print(f"collect_sec: {collect_time:.4f}")
    print(f"peak_mem_mb: {peak / 1024 / 1024:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polars Factor benchmark")
    parser.add_argument("--n-symbols", type=int, default=500)
    parser.add_argument("--n-periods", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.n_symbols, args.n_periods)
