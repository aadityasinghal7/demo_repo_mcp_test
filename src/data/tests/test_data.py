import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_zero_weights():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 1, 1],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    expected = pd.Series([np.nan, 35.0], index=["x", "y"])
    pd.testing.assert_series_equal(result, expected)
