import polars as pl

from math import nan
from typing import Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ...constants import EPSILON


class MathOpsMixin:
    def abs(self) -> Self:
        result_lf = self._lf.with_columns(pl.col("factor").abs().alias("factor"))
        return self.__class__(result_lf, f"abs({self.name})")

    def sign(self) -> Self:
        result_lf = self._lf.with_columns(pl.col("factor").sign().alias("factor"))
        return self.__class__(result_lf, f"sign({self.name})")

    def inverse(self) -> Self:
        result_lf = self._lf.with_columns(
            pl.when(pl.col("factor").abs() <= EPSILON)
            .then(pl.lit(None))
            .otherwise(1 / pl.col("factor"))
            .alias("factor")
        )
        return self.__class__(result_lf, f"inverse({self.name})")

    def log(self, base: Optional[float] = None) -> Self:
        if base is None:
            result_lf = self._lf.with_columns(
                pl.when(pl.col("factor") > 0).then(pl.col("factor").log()).otherwise(None).alias("factor")
            )
            name = f"log({self.name})"
        else:
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}. Base must be greater than 0 and not equal to 1.")
            # Use log change of base formula: log_b(x) = ln(x) / ln(b)
            # Calculate ln(base) using Polars expression
            result_lf = self._lf.with_columns(
                pl.when(pl.col("factor") > 0)
                .then(pl.col("factor").log() / pl.lit(base).log())
                .otherwise(None)
                .alias("factor")
            )
            name = f"log({self.name},{base})"
        return self.__class__(result_lf, name)

    def ln(self) -> Self:
        return self.log()

    def sqrt(self) -> Self:
        result_lf = self._lf.with_columns(
            pl.when(pl.col("factor") > 0).then(pl.col("factor").sqrt()).otherwise(None).alias("factor")
        )
        return self.__class__(result_lf, f"sqrt({self.name})")

    def signed_log1p(self) -> Self:
        result_lf = self._lf.with_columns((pl.col("factor").sign() * pl.col("factor").abs().log1p()).alias("factor"))
        return self.__class__(result_lf, f"signed_log1p({self.name})")

    def signed_pow(self, exponent: Union[Self, float]) -> Self:
        if isinstance(exponent, self.__class__):
            # Factor-factor path
            result_lf = self._lf.join(exponent._lf, on=["start_time", "end_time", "symbol"], suffix="_exp")
            result_lf = result_lf.with_columns(
                (pl.col("factor").sign() * pl.col("factor").abs().pow(pl.col("factor_exp"))).alias("factor")
            )
            result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
            return self.__class__(result_lf, f"signed_pow({self.name},{exponent})")
        else:
            # Scalar path
            result_lf = self._lf.with_columns(
                (pl.col("factor").sign() * pl.col("factor").abs().pow(pl.lit(exponent))).alias("factor")
            )
            return self.__class__(result_lf, f"signed_pow({self.name},{exponent})")

    def pow(self, exponent: Union[Self, float]) -> Self:
        if isinstance(exponent, self.__class__):
            # Factor-factor path
            result_lf = self._lf.join(exponent._lf, on=["start_time", "end_time", "symbol"], suffix="_exp")
            result_lf = result_lf.with_columns(pl.col("factor").pow(pl.col("factor_exp")).alias("factor"))
            result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
            return self.__class__(result_lf, f"pow({self.name},{exponent})")
        else:
            # Scalar path
            result_lf = self._lf.with_columns(pl.col("factor").pow(pl.lit(exponent)).alias("factor"))
            return self.__class__(result_lf, f"pow({self.name},{exponent})")

    def add(self, other: Union[Self, float]) -> Self:
        return self.__add__(other)

    def sub(self, other: Union[Self, float]) -> Self:
        return self.__sub__(other)

    def mul(self, other: Union[Self, float]) -> Self:
        return self.__mul__(other)

    def div(self, other: Union[Self, float]) -> Self:
        return self.__truediv__(other)

    def where(self, cond: Self, other: Union[Self, float] = nan) -> Self:
        if not isinstance(cond, self.__class__):
            raise ValueError(f"Condition must be a Factor, got {type(cond)}")

        result_lf = self._lf.join(cond._lf, on=["start_time", "end_time", "symbol"], suffix="_cond")

        if isinstance(other, self.__class__):
            result_lf = result_lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
            result_lf = result_lf.with_columns(
                pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
                .then(pl.col("factor"))
                .otherwise(pl.col("factor_other"))
                .alias("factor")
            )
        else:
            result_lf = result_lf.with_columns(
                pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
                .then(pl.col("factor"))
                .otherwise(pl.lit(other))
                .alias("factor")
            )

        result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
        return self.__class__(result_lf, f"where({self.name})")

    def max(self, other: Union[Self, float]) -> Self:
        if isinstance(other, self.__class__):
            # Factor-factor path
            result_lf = self._lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
            result_lf = result_lf.with_columns(
                pl.max_horizontal(pl.col("factor"), pl.col("factor_other")).alias("factor")
            )
            result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
            return self.__class__(result_lf, f"max({self.name},{other})")
        else:
            # Scalar path
            result_lf = self._lf.with_columns(pl.max_horizontal(pl.col("factor"), pl.lit(other)).alias("factor"))
            return self.__class__(result_lf, f"max({self.name},{other})")

    def min(self, other: Union[Self, float]) -> Self:
        if isinstance(other, self.__class__):
            # Factor-factor path
            result_lf = self._lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
            result_lf = result_lf.with_columns(
                pl.min_horizontal(pl.col("factor"), pl.col("factor_other")).alias("factor")
            )
            result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
            return self.__class__(result_lf, f"min({self.name},{other})")
        else:
            # Scalar path
            result_lf = self._lf.with_columns(pl.min_horizontal(pl.col("factor"), pl.lit(other)).alias("factor"))
            return self.__class__(result_lf, f"min({self.name},{other})")

    def reverse(self) -> Self:
        return self.__neg__()
