import numpy as np
import pandas as pd
import pytest

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
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 3, 4, 5],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "X": (10 * 1 + 20 * 2) / (1 + 2),  # (10 + 40) / 3 = 50 / 3 = 16.666...
            "Y": (30 * 3 + 40 * 4 + 50 * 5)
            / (3 + 4 + 5),  # (90+160+250)/12= 500/12=41.666...
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
