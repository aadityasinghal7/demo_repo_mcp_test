import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average, groupweightedaverage


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


def test_groupweightedaverage_zero_group_weight():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 0, 0],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # When sum of weights is zero, division results in inf or NaN.
    # Expect the resulting Series to have NaN values for groups with zero weight sum.
    expected = pd.Series([np.nan, np.nan], index=["x", "y"])

    pd.testing.assert_series_equal(result, expected)