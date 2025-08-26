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


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y", "Z"],
            "value": [10, 20, 10, 30, 20, 40],
            "weight": [1, 2, 3, 1, 1, 4],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "X": (10*1 + 20*2) / (1 + 2),  # (10 + 40)/3 = 50/3 ≈ 16.6667
            "Y": (10*3 + 30*1 + 20*1) / (3 + 1 + 1),  # (30 + 30 + 20)/5 = 80/5 = 16.0
            "Z": 40,  # only one element: 40*4/4 = 40
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())