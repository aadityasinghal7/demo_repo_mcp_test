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
            "value": [1, 2, 3, 4, 5],
            "weight": [0.1, 0.9, 0.5, 0.3, 0.2],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "X": (1*0.1 + 2*0.9) / (0.1 + 0.9),
            "Y": (3*0.5 + 4*0.3 + 5*0.2) / (0.5 + 0.3 + 0.2),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())