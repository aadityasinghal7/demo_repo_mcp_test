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


def test_groupweightedaverage_basic_functionality():
    # Create a simple DataFrame with groups, values, and weights
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 1, 2],
        }
    )

    # Manually calculate expected weighted averages per group
    # For group X: weighted average = (10*1 + 20*3) / (1 + 3) = (10 + 60)/4 = 17.5
    # For group Y: weighted average = (30*2 + 40*1 + 50*2) / (2 + 1 + 2) = (60 + 40 + 100)/5 = 40.0
    expected = pd.Series({"X": 17.5, "Y": 40.0})

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
