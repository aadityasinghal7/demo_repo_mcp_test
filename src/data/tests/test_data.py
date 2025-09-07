import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average


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


def test_groupweightedaverage_zero_total_weights_group():
    # Create a DataFrame with groups and weights where one group has zero total weights
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Z"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 0, 0, 0],  # group Z weight sum is zero
        }
    )

    # Calculate the group weighted average
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For groups X and Y, expected values:
    # X: (10*1 + 20*2) / (1+2) = (10 + 40) / 3 = 50 / 3 = 16.666...
    # Y: (30*0 + 40*0) / (0+0) -> division by zero, expect NaN
    # Z: (50*0) / 0 -> division by zero, expect NaN

    expected = pd.Series(
        [50 / 3, np.nan, np.nan], index=pd.Index(["X", "Y", "Z"], name="group")
    )

    pd.testing.assert_series_equal(
        result.sort_index(), expected.sort_index(), check_names=False
    )
