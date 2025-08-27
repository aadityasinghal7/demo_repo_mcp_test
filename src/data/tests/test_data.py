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


def test_groupweightedaverage_zero_sum_weights():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weights": [0, 0, 5, -5],  # sum weights for group 'x' = 0, for 'y' = 0
        }
    )
    # This should not raise ZeroDivisionError and result should be NaN for groups with zero total weight
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")
    assert "x" in result.index and "y" in result.index
    assert pd.isna(result.loc["x"]), "Expected NaN for group 'x' due to zero total weight"
    assert pd.isna(result.loc["y"]), "Expected NaN for group 'y' due to zero total weight"