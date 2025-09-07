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
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 1],
        }
    )

    # Manually calculate expected grouped weighted averages
    # For group X:
    # weighted sum = 10*1 + 20*3 = 10 + 60 = 70
    # sum weights = 1 + 3 = 4
    # weighted average = 70 / 4 = 17.5
    #
    # For group Y:
    # weighted sum = 30*2 + 40*2 + 50*1 = 60 + 80 + 50 = 190
    # sum weights = 2 + 2 + 1 = 5
    # weighted average = 190 / 5 = 38.0

    expected = pd.Series({"X": 17.5, "Y": 38.0})

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
