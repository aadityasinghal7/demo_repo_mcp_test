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


def test_groupweightedaverage_error_handling():
    df = pd.DataFrame(
        {
            "brand": ["X", "Y", "X"],
            "value": [10, 20, 30],
            "weight": [1, 2, 3],
        }
    )
    # Missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="missing_group", value="value", weights="weight"
        )

    # Missing value column - will raise KeyError when accessing df[value]
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="brand", value="missing_value", weights="weight"
        )

    # Missing weights column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="brand", value="value", weights="missing_weight"
        )

    # Groupby column present but weights column with None
    with pytest.raises(KeyError):
        groupweightedaverage(df, groupby="brand", value="value", weights=None)

    # weights column present but groupby column as None
    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby=None, value="value", weights="weight")
