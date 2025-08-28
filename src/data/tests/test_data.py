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
            "group": ["X", "X", "Y", "Y"],
            "value": [10, 20, 30, 40],
            "weights": [0, 0, 0, 0],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")
    expected = pd.Series([np.nan, np.nan], index=pd.Index(["X", "Y"], name="group"))
    pd.testing.assert_series_equal(result, expected)