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
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 1],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "x": (10 * 1 + 20 * 3) / (1 + 3),
            "y": (30 * 2 + 40 * 2 + 50 * 1) / (2 + 2 + 1),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
