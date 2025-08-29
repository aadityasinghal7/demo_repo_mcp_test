import numpy as np
import pandas as pd

from src.data.data_processing import groupweightedaverage, weigthed_average


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
            "value": [10, 20, 10, 20, 30],
            "weight": [1, 2, 1, 3, 6],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected_values = {
        "X": (10 * 1 + 20 * 2) / (1 + 2),  # (10 + 40) / 3 = 50 / 3 = 16.666...
        "Y": (10 * 1 + 20 * 3 + 30 * 6)
        / (1 + 3 + 6),  # (10 + 60 + 180) / 10 = 250 / 10 = 25
    }
    expected = pd.Series(expected_values)
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
