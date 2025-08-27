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
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weights": [1, 2, 1, 3, 1],
        }
    )
    # Compute expected group weighted averages manually:
    # For group 'x':
    #  sum(value * weights) = 10*1 + 20*2 = 10 + 40 = 50
    #  sum(weights) = 1 + 2 = 3
    #  weighted average = 50/3 ≈ 16.6667
    #
    # For group 'y':
    #  sum(value * weights) = 30*1 + 40*3 + 50*1 = 30 + 120 + 50 = 200
    #  sum(weights) = 1 + 3 + 1 = 5
    #  weighted average = 200/5 = 40

    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")
    expected = pd.Series({"x": 50 / 3, "y": 40})
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())