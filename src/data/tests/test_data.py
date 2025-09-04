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


def test_groupweightedaverage_basic():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B", "C"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "wt": [1.0, 3.0, 2.0, 2.0, 1.0],
        }
    )
    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")
    expected = pd.Series(
        {
            "A": (10.0 * 1.0 + 20.0 * 3.0) / (1.0 + 3.0),  # 17.5
            "B": (30.0 * 2.0 + 40.0 * 2.0) / (2.0 + 2.0),  # 35.0
            "C": 50.0 / 1.0,  # 50.0
        }
    )
    assert np.allclose(
        result.sort_index(), expected.sort_index()
    ), f"Expected {expected}, but got {result}"
