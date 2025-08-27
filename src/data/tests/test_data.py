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
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 10, 20, 30],
            "weight": [1, 3, 2, 2, 6],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    # Calculate expected manually:
    # For group X: weighted avg = (10*1 + 20*3) / (1+3) = (10 + 60) / 4 = 17.5
    # For group Y: weighted avg = (10*2 + 20*2 + 30*6) / (2+2+6) = (20 + 40 + 180) / 10 = 24.0
    expected = pd.Series({"X": 17.5, "Y": 24.0})
    pd.testing.assert_series_equal(result, expected)


