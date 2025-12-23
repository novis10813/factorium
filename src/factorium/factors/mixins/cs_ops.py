import pandas as pd
import numpy as np
from typing import Self


class CrossSectionalOpsMixin:
    def rank(self) -> Self:
        def rank_op(group):
            if group.nunique() == 1:
                return pd.Series(np.nan, index=group.index)
            return group.rank(method='min', pct=True)
        return self._cs_op(rank_op, 'rank', require_no_nan=True)
    
    def mean(self) -> Self:
        return self._cs_op(lambda x: x.mean(), 'mean', require_no_nan=True)
    
    def median(self) -> Self:
        return self._cs_op(lambda x: x.median(), 'median', require_no_nan=True)
