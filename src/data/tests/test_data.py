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


def test_groupweightedaverage_with_zero_weights_in_group():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "z", "z"],
            "value": [10, 20, 30, 40, 50, 60],
            "weight": [1, 0, 0, 0, 2, 3],
        }
    )
    # Here, group 'y' has zero total weight (0 + 0)
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For group 'x': weighted average = (10*1 + 20*0) / (1 + 0) = 10
    # For group 'y': weighted average = NaN (division by zero)
    # For group 'z': weighted average = (50*2 + 60*3) / (2 + 3) = (100 + 180) / 5 = 56
    expected = pd.Series({"x": 10, "y": np.nan, "z": 56})

    pd.testing.assert_series_equal(result, expected, check_names=False)