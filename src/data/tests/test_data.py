import numpy as np
import pandas as pd

from src.data.data_processing import groupweightedaverage, weigthed_average


def test_weigthed_average():
    df = pd.DataFrame(
        {
            "A": [2, 0, 0],
            "B": [1, 2, 3],
        }
    )
    weights = {"A": 1.0, "B": 0.0}
    result = weigthed_average(df, weights)
    expected = pd.Series([2.0, 0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_groupweightedaverage_zero_sum_weights():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "wt": [0.0, 0.0, 0.0, 0.0],  # sum weights per group are zero
        }
    )
    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")
    expected = pd.Series([np.nan, np.nan], index=["A", "B"])
    pd.testing.assert_series_equal(result, expected)
