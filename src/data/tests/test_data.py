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
            "group": ["x", "y", "x"],
            "value": [1, 2, 3],
            "weight": [0.1, 0.2, 0.3],
        }
    )
    # Missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="missing_group", value="value", weights="weight")
    # Missing weight column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="group", value="value", weights="missing_weight")
    # Missing value column
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="group", value="missing_value", weights="weight")