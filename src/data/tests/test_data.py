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
            "group": ["x", "y", "x"],
            "value": [10, 20, 30],
            "weight": [1.0, 0.5, 0.5],
        }
    )

    # Missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="missing_group", value="value", weights="weight"
        )

    # Missing value column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="group", value="missing_value", weights="weight"
        )

    # Missing weights column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="group", value="value", weights="missing_weight"
        )

    # weights column with wrong dtype (for completeness, though not required)
    df_wrong_type = df.copy()
    df_wrong_type["weight"] = ["a", "b", "c"]
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_wrong_type, groupby="group", value="value", weights="weight"
        )
