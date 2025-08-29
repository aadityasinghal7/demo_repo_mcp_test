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


def test_groupweightedaverage_error_handling():
    df = pd.DataFrame(
        {
            "group": ["x", "y", "x", "y"],
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 3, 4],
        }
    )
    # Test missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="missing_group", value="value", weights="weight")

    # Test missing value column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="group", value="missing_value", weights="weight")

    # Test missing weights column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="group", value="value", weights="missing_weight")