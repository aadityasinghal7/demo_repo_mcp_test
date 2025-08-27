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
            "weight": [1, 2, 3, 4, 5],
        }
    )
    # Calculate expected weighted averages per group manually:
    # For group X: weighted sum = 10*1 + 20*2 = 10 + 40 = 50; total weights = 1 + 2 = 3; avg = 50/3 ≈ 16.6667
    # For group Y: weighted sum = 30*3 + 40*4 + 50*5 = 90 + 160 + 250 = 500; total weights = 3 + 4 + 5 = 12; avg = 500/12 ≈ 41.6667
    expected = pd.Series(data=[50 / 3, 500 / 12], index=["X", "Y"])
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-6)