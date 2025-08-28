import numpy as np
import pandas as pd
import pytest

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


def test_groupweightedaverage_unit_test():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y", "Z"],
            "value": [10, 20, 10, 30, 50, 40],
            "weight": [1, 2, 3, 1, 1, 4],
        }
    )
    # Expected calculations:
    # Group X: weighted avg = (10*1 + 20*2) / (1+2) = (10 + 40)/3 = 50/3 ≈ 16.6667
    # Group Y: weighted avg = (10*3 + 30*1 + 50*1) / (3+1+1) = (30 + 30 + 50)/5 = 110/5 = 22
    # Group Z: weighted avg = (40*4) / 4 = 160 / 4 = 40
    expected = pd.Series(
        {
            "X": (10 * 1 + 20 * 2) / (1 + 2),
            "Y": (10 * 3 + 30 * 1 + 50 * 1) / (3 + 1 + 1),
            "Z": (40 * 4) / 4,
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False)