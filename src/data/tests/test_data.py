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
            "value": [1, 2, 3, 4, 5],
            "weight": [2, 1, 1, 3, 2],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    expected = pd.Series(
        {
            "x": (1*2 + 2*1) / (2 + 1),  # (2 + 2)/3 = 4/3 = 1.3333...
            "y": (3*1 + 4*3 + 5*2) / (1 + 3 + 2),  # (3 + 12 + 10)/6 = 25/6 = 4.1666...
        }
    )
    pd.testing.assert_series_equal(result, expected, check_names=False)