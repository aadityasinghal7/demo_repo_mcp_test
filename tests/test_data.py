import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_nan_inf_values():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1.0, np.nan, np.inf, -np.inf],
            "weight": [0.5, 0.5, 1.0, 1.0],
        }
    )

    # Calculate expected Series manually:
    # group 'x': values 1.0 and nan, weights 0.5 and 0.5
    # weighted sum = 1.0*0.5 + nan*0.5 = nan  (because nan propagates)
    # sum weights = 0.5 + 0.5 = 1.0
    # result for 'x' = nan / 1.0 = nan

    # group 'y': values inf and -inf, weights 1.0 and 1.0
    # weighted sum = inf*1 + -inf*1 = nan (inf - inf = nan)
    # sum weights = 2.0
    # result for 'y' = nan / 2 = nan

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Result is a Series indexed by 'x' and 'y'
    assert isinstance(result, pd.Series)
    assert set(result.index) == {"x", "y"}
    assert pd.isna(result.loc["x"])
    assert pd.isna(result.loc["y"])
