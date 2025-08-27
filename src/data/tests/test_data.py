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
    # Create a DataFrame with multiple groups and weights
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Z"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 5],
        }
    )
    # Calculate weighted average per group manually
    # For group X: (10*1 + 20*3) / (1+3) = (10 + 60) / 4 = 70 / 4 = 17.5
    # For group Y: (30*2 + 40*2) / (2+2) = (60 + 80) / 4 = 140 / 4 = 35
    # For group Z: (50*5) / 5 = 250 / 5 = 50
    expected = pd.Series(
        {
            "X": 17.5,
            "Y": 35.0,
            "Z": 50.0,
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    # Check index matches
    assert set(result.index) == set(expected.index)
    # Check values (use np.isclose inside all)
    assert np.allclose(result.values, expected.values), f"Expected {expected}, but got {result}"