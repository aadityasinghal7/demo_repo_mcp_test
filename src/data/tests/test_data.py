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


def test_groupweightedaverage_basic():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 1, 3, 6],
        }
    )
    # Expected group-wise weighted averages:
    # For group "X":
    #   weights: 1 + 2 = 3
    #   weighted sum: 10*1 + 20*2 = 10 + 40 = 50
    #   average = 50 / 3 = 16.(6)
    # For group "Y":
    #   weights: 1 + 3 + 6 = 10
    #   weighted sum: 30*1 + 40*3 + 50*6 = 30 + 120 + 300 = 450
    #   average = 450 / 10 = 45
    expected = pd.Series({"X": 50 / 3, "Y": 45})
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False)