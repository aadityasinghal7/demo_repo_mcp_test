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


def test_groupweightedaverage_zero_weight_group():
    df = pd.DataFrame(
        {
            "group": ["x", "y", "z", "y", "x"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 0, 0, 0, 1],  # group "y" total weight = 0, "z" total weight = 0
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For groups with total weight zero, expect NaN (division by zero)
    expected = pd.Series(
        {"x": (10*1 + 50*1) / (1 + 1), "y": np.nan, "z": np.nan}
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False, check_dtype=False)