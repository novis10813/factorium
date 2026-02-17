from typing import Union, Optional, Callable, TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from abc import ABC
import pandas as pd
import polars as pl
from pathlib import Path
import numpy as np

from ..constants import EPSILON

if TYPE_CHECKING:
    from ..aggbar import AggBar


class BaseFactor(ABC):
    def __init__(
        self, data: Union["AggBar", pd.DataFrame, pl.DataFrame, pl.LazyFrame, Path], name: Optional[str] = None
    ):
        self._name = name or "factor"
        self._lf = self._to_lazy(data)

    def _to_lazy(self, data: Union["AggBar", pd.DataFrame, pl.DataFrame, pl.LazyFrame, Path]) -> pl.LazyFrame:
        if isinstance(data, Path):
            if data.suffix == ".csv":
                lf = pl.scan_csv(str(data))
            elif data.suffix == ".parquet":
                lf = pl.scan_parquet(str(data))
            else:
                raise ValueError(f"Invalid file extension: {data.suffix}")
            return self._normalize_schema_lazy(lf)

        if isinstance(data, pl.LazyFrame):
            return self._normalize_schema_lazy(data)

        if isinstance(data, pl.DataFrame):
            return self._normalize_schema_lazy(data.lazy())

        # Check for AggBar (prefer Polars path if available)
        if hasattr(data, "to_polars"):
            return self._normalize_schema_lazy(data.to_polars().lazy())

        # Legacy: AggBar with only to_df (Pandas)
        if hasattr(data, "to_df"):
            return self._normalize_schema_pandas(data.to_df())

        if isinstance(data, pd.DataFrame):
            return self._normalize_schema_pandas(data)

        raise ValueError(f"Invalid data type: {type(data)}")

    def _normalize_schema_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        col_names = lf.collect_schema().names()

        if len(col_names) == 4 and "factor" not in col_names:
            rename_map = {
                col_names[0]: "start_time",
                col_names[1]: "end_time",
                col_names[2]: "symbol",
                col_names[3]: "factor",
            }
            lf = lf.rename(rename_map)
            col_names = ["start_time", "end_time", "symbol", "factor"]
        elif "factor" not in col_names:
            factor_columns = [col for col in col_names if col not in ["start_time", "end_time", "symbol"]]
            if not factor_columns:
                raise ValueError("No factor columns found")
            lf = lf.select(["start_time", "end_time", "symbol", factor_columns[0]]).rename(
                {factor_columns[0]: "factor"}
            )
            col_names = ["start_time", "end_time", "symbol", "factor"]

        required_cols = {"start_time", "end_time", "symbol", "factor"}
        if not required_cols.issubset(set(col_names)):
            raise ValueError(f"Missing required columns. Required: {required_cols}, Got: {set(col_names)}")

        return lf.select(["start_time", "end_time", "symbol", "factor"]).sort(["end_time", "symbol"])

    def _normalize_schema_pandas(self, df: pd.DataFrame) -> pl.LazyFrame:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if len(df.columns) == 4 and "factor" not in df.columns:
            df.columns = ["start_time", "end_time", "symbol", "factor"]
        elif "factor" not in df.columns:
            factor_columns = [col for col in df.columns if col not in ["start_time", "end_time", "symbol"]]
            if not factor_columns:
                raise ValueError("No factor columns found")
            df = df[["start_time", "end_time", "symbol", factor_columns[0]]]
            df.columns = ["start_time", "end_time", "symbol", "factor"]

        required_cols = {"start_time", "end_time", "symbol", "factor"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns. Required: {required_cols}, Got: {set(df.columns)}")

        df = df.sort_values(by=["end_time", "symbol"]).reset_index(drop=True)
        df = df[["start_time", "end_time", "symbol", "factor"]]
        return pl.from_pandas(df).lazy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def data(self) -> pl.DataFrame:
        """Return internal data as eager Polars DataFrame."""
        return self._lf.collect()

    @property
    def _data(self) -> pd.DataFrame:
        """Return internal data as pandas DataFrame for backward compatibility with mixins."""
        return self._lf.collect().to_pandas()

    @_data.setter
    def _data(self, value: pd.DataFrame) -> None:
        """Set internal data from pandas DataFrame."""
        self._lf = pl.from_pandas(value).lazy()

    @property
    def lazy(self) -> pl.LazyFrame:
        """Return internal data as Polars LazyFrame."""
        return self._lf

    def to_pandas(self) -> pd.DataFrame:
        """Convert internal data to pandas DataFrame."""
        return self._lf.collect().to_pandas()

    def _validate_window(self, window: int) -> None:
        if window <= 0:
            raise ValueError("Window must be positive")

    def _validate_factor(self, other: Self, op_name: str) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"{op_name}: other must be a Factor object")

    @staticmethod
    def _replace_inf(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)

    def _to_polars(self) -> pl.DataFrame:
        """Convert internal data to Polars (eager)."""
        return self._lf.collect()

    def _from_polars(self, pl_df: pl.DataFrame, name: str) -> Self:
        """Create new Factor from Polars DataFrame."""
        return self.__class__(pl_df, name)

    def _binary_op(
        self, other: Union["BaseFactor", float], op_func: Callable, op_name: str, scalar_suffix: Optional[str] = None
    ) -> Self:
        if isinstance(other, self.__class__):
            # Use Polars LazyFrame join for factor-factor operations
            # WORKAROUND: Materialize LazyFrames before full join to avoid Polars optimizer bug
            # See: https://github.com/pola-rs/polars/issues/26306
            left_lf = self._lf.collect().lazy()
            right_lf = (
                other._lf.collect()
                .lazy()
                .rename(
                    {
                        "start_time": "start_time_right",
                        "end_time": "end_time_right",
                        "symbol": "symbol_right",
                        "factor": "other",
                    }
                )
            )

            merged_lf = left_lf.join(
                right_lf,
                left_on=["start_time", "end_time", "symbol"],
                right_on=["start_time_right", "end_time_right", "symbol_right"],
                how="full",
            )

            key_exprs = [
                pl.coalesce([pl.col("start_time"), pl.col("start_time_right")]).alias("start_time"),
                pl.coalesce([pl.col("end_time"), pl.col("end_time_right")]).alias("end_time"),
                pl.coalesce([pl.col("symbol"), pl.col("symbol_right")]).alias("symbol"),
            ]

            result_lf = merged_lf.with_columns(
                key_exprs + [op_func(pl.col("factor"), pl.col("other")).alias("factor")]
            ).select(["start_time", "end_time", "symbol", "factor"])

            return self.__class__(result_lf, f"({self.name}{op_name}{other.name})")
        else:
            # Scalar operation using Polars expressions
            # The op_func is a lambda that takes (x_expr, y_literal) and returns result_expr
            result_lf = self._lf.with_columns(op_func(pl.col("factor"), pl.lit(other)).alias("factor"))
            suffix = scalar_suffix if scalar_suffix is not None else str(other)
            return self.__class__(result_lf, f"({self.name}{op_name}{suffix})")

    def _comparison_op(self, other: Union["BaseFactor", float], comp_func: Callable, op_name: str) -> Self:
        if isinstance(other, self.__class__):
            # Use Polars LazyFrame join for factor-factor operations
            # WORKAROUND: Materialize LazyFrames before full join to avoid Polars optimizer bug
            # See: https://github.com/pola-rs/polars/issues/26306
            left_lf = self._lf.collect().lazy()
            right_lf = (
                other._lf.collect()
                .lazy()
                .rename(
                    {
                        "start_time": "start_time_right",
                        "end_time": "end_time_right",
                        "symbol": "symbol_right",
                        "factor": "other",
                    }
                )
            )

            merged_lf = left_lf.join(
                right_lf,
                left_on=["start_time", "end_time", "symbol"],
                right_on=["start_time_right", "end_time_right", "symbol_right"],
                how="full",
            )

            key_exprs = [
                pl.coalesce([pl.col("start_time"), pl.col("start_time_right")]).alias("start_time"),
                pl.coalesce([pl.col("end_time"), pl.col("end_time_right")]).alias("end_time"),
                pl.coalesce([pl.col("symbol"), pl.col("symbol_right")]).alias("symbol"),
            ]

            result_lf = merged_lf.with_columns(
                key_exprs + [comp_func(pl.col("factor"), pl.col("other")).cast(pl.Int64).alias("factor")]
            ).select(["start_time", "end_time", "symbol", "factor"])
        else:
            # Scalar comparison using Polars expressions
            result_lf = self._lf.with_columns(comp_func(pl.col("factor"), pl.lit(other)).cast(pl.Int64).alias("factor"))
        return self.__class__(result_lf, f"({self.name}{op_name}{getattr(other, 'name', other)})")

    def __mul__(self, other: Union["BaseFactor", float]) -> Self:
        return self._binary_op(other, lambda x, y: x * y, "*")

    def __neg__(self) -> Self:
        return self.__mul__(-1)

    def __add__(self, other: Union["BaseFactor", float]) -> Self:
        return self._binary_op(other, lambda x, y: x + y, "+")

    def __sub__(self, other: Union["BaseFactor", float]) -> Self:
        return self._binary_op(other, lambda x, y: x - y, "-")

    def __truediv__(self, other: Union["BaseFactor", float]) -> Self:
        if isinstance(other, self.__class__):
            # Factor / Factor division with safe division
            # WORKAROUND: Materialize LazyFrames before full join to avoid Polars optimizer bug
            # See: https://github.com/pola-rs/polars/issues/26306
            left_lf = self._lf.collect().lazy()
            right_lf = (
                other._lf.collect()
                .lazy()
                .rename(
                    {
                        "start_time": "start_time_right",
                        "end_time": "end_time_right",
                        "symbol": "symbol_right",
                        "factor": "other",
                    }
                )
            )

            merged_lf = left_lf.join(
                right_lf,
                left_on=["start_time", "end_time", "symbol"],
                right_on=["start_time_right", "end_time_right", "symbol_right"],
                how="full",
            )

            key_exprs = [
                pl.coalesce([pl.col("start_time"), pl.col("start_time_right")]).alias("start_time"),
                pl.coalesce([pl.col("end_time"), pl.col("end_time_right")]).alias("end_time"),
                pl.coalesce([pl.col("symbol"), pl.col("symbol_right")]).alias("symbol"),
            ]

            result_expr = (
                pl.when(pl.col("other").abs() <= EPSILON)
                .then(pl.lit(None))
                .otherwise(pl.col("factor") / pl.col("other"))
                .alias("factor")
            )

            result_lf = merged_lf.with_columns(key_exprs + [result_expr]).select(
                ["start_time", "end_time", "symbol", "factor"]
            )

            return self.__class__(result_lf, f"({self.name}/{other.name})")
        else:
            # Scalar division with safe division
            result_lf = self._lf.with_columns(
                pl.when(pl.lit(other).abs() <= EPSILON)
                .then(pl.lit(None))
                .otherwise(pl.col("factor") / pl.lit(other))
                .alias("factor")
            )
            return self.__class__(result_lf, f"({self.name}/{other})")

    def __rtruediv__(self, other: Union["BaseFactor", float]) -> Self:
        if isinstance(other, self.__class__):
            return other.__truediv__(self)
        else:
            # scalar / factor with safe division
            result_lf = self._lf.with_columns(
                pl.when(pl.col("factor").abs() <= EPSILON)
                .then(pl.lit(None))
                .otherwise(pl.lit(other) / pl.col("factor"))
                .alias("factor")
            )
            return self.__class__(result_lf, f"({other}/{self.name})")

    def __radd__(self, other: Union["BaseFactor", float]) -> Self:
        return self.__add__(other)

    def __rsub__(self, other: Union["BaseFactor", float]) -> Self:
        if isinstance(other, self.__class__):
            return other.__sub__(self)
        else:
            # scalar - factor
            result_lf = self._lf.with_columns((pl.lit(other) - pl.col("factor")).alias("factor"))
            return self.__class__(result_lf, f"({other}-{self.name})")

    def __rmul__(self, other: Union["BaseFactor", float]) -> Self:
        return self.__mul__(other)

    def __lt__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x < y, "<")

    def __le__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x <= y, "<=")

    def __gt__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x > y, ">")

    def __ge__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x >= y, ">=")

    def __eq__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x == y, "==")

    def __ne__(self, other: Union["BaseFactor", float]) -> Self:
        return self._comparison_op(other, lambda x, y: x != y, "!=")

    def __len__(self) -> int:
        """Get number of rows.

        Note: This triggers a lightweight aggregation query (COUNT),
        which is much faster than collecting the full dataset but still
        requires execution. Avoid calling in tight loops.
        """
        return self._lf.select(pl.len()).collect().item()
