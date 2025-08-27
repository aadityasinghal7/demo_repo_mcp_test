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
            "group": ["g1", "g1", "g2", "g2"],
            "value": [10, 20, 30, 40],
            "weights": [0, 0, 0, 0],  # sum of weights zero for both groups
        }
    )
    # This should handle division by zero, resulting in NaN for both groups
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")
    expected = pd.Series([np.nan, np.nan], index=["g1", "g2"])
    pd.testing.assert_series_equal(result, expected, check_names=False)