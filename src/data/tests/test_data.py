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
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 1, 2, 3],
        }
    )
    # Manually compute expected weighted averages per group:
    # For group 'x':
    # Weighted sum = 10*1 + 20*2 = 10 + 40 = 50
    # Sum weights = 1 + 2 = 3
    # Weighted average = 50 / 3 ≈ 16.6667
    #
    # For group 'y':
    # Weighted sum = 30*1 + 40*2 + 50*3 = 30 + 80 + 150 = 260
    # Sum weights = 1 + 2 + 3 = 6
    # Weighted average = 260 / 6 ≈ 43.3333
    expected = pd.Series(
        [50 / 3, 260 / 6],
        index=pd.Index(["x", "y"], name="group"),
        dtype=float,
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result, expected)