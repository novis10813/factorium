from typing import List, Self, Union

import numpy as np
import pandas as pd


class CrossSectionalOpsMixin:
    def cs_rank(self) -> Self:
        """Cross-sectional rank (percentile). Strict: Returns NaN if any input is NaN."""

        def rank_op(group):
            if group.nunique() <= 1:
                return pd.Series(0.5, index=group.index)
            return group.rank(method="min", pct=True)

        return self._cs_op(rank_op, "cs_rank")

    def rank(self) -> Self:
        """Alias for cs_rank."""
        return self.cs_rank()

    def cs_zscore(self) -> Self:
        """Cross-sectional z-score standardization. Strict: Returns NaN if any input is NaN."""

        def zscore_op(group):
            std = group.std()
            if std < 1e-10:
                return pd.Series(0.0, index=group.index)
            return (group - group.mean()) / std

        return self._cs_op(zscore_op, "cs_zscore")

    def cs_demean(self) -> Self:
        """Cross-sectional de-meaning. Strict: Returns NaN if any input is NaN."""
        return self._cs_op(lambda x: x - x.mean(), "cs_demean")

    def cs_winsorize(self, limits: Union[float, List[float]] = 0.025) -> Self:
        """
        Cross-sectional winsorization. Strict: Returns NaN if any input is NaN.
        Limits can be a single float (applied to both sides) or [lower, upper].
        """
        if isinstance(limits, float):
            lower_lim = upper_lim = limits
        else:
            lower_lim, upper_lim = limits

        def winsorize_op(group):
            lower_val = group.quantile(lower_lim)
            upper_val = group.quantile(1 - upper_lim)
            return group.clip(lower=lower_val, upper=upper_val)

        return self._cs_op(winsorize_op, f"cs_winsorize({limits})")

    def cs_neutralize(self, other: Self) -> Self:
        """
        Cross-sectional neutralization against another factor.
        Returns the residuals of: self = alpha + beta * other + residuals.
        Strict: Returns NaN if any input (self or other) is NaN in the cross-section.
        """
        self._validate_factor(other, "cs_neutralize")

        # Merge data first
        merged = pd.merge(self._data, other.data, on=["start_time", "end_time", "symbol"], suffixes=("_y", "_x"))

        if merged.empty:
            raise ValueError("No common data for neutralization")

        # We cannot use _cs_op easily here because it involves two factors.
        # We manually iterate to avoid pandas apply issues.
        residual_series_list = []

        for _, group in merged.groupby("end_time"):
            # Check for NaNs strictly in the whole group (both x and y)
            if group[["factor_x", "factor_y"]].isna().any().any():
                residual_series_list.append(pd.Series(np.nan, index=group.index))
                continue

            y = group["factor_y"].values
            x = group["factor_x"].values

            # Check for constant x (cannot regress)
            if np.std(x) < 1e-10:
                residual_series_list.append(pd.Series(np.nan, index=group.index))
                continue

            # Add constant for intercept
            A = np.vstack([x, np.ones(len(x))]).T

            try:
                # Solve least squares: y = [x 1] * [beta alpha]^T
                beta_alpha, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ beta_alpha
                residual_series_list.append(pd.Series(resid, index=group.index))
            except Exception:
                residual_series_list.append(pd.Series(np.nan, index=group.index))

        result_data = merged.copy()
        if residual_series_list:
            full_resid = pd.concat(residual_series_list)
            # Align by index in case order changed (though concat usually preserves it if not sorted)
            result_data["factor"] = full_resid
        else:
            result_data["factor"] = np.nan

        result_data = result_data[["start_time", "end_time", "symbol", "factor"]]
        return self.__class__(result_data, f"cs_neutralize({self.name},{other.name})")

    def mean(self) -> Self:
        """Cross-sectional mean. Strict: Returns NaN if any input is NaN."""
        return self._cs_op(lambda x: pd.Series(x.mean(), index=x.index), "mean")

    def median(self) -> Self:
        """Cross-sectional median. Strict: Returns NaN if any input is NaN."""
        return self._cs_op(lambda x: pd.Series(x.median(), index=x.index), "median")
