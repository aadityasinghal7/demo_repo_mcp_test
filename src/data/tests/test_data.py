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
            "brand": ["A", "B", "A", "C"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "weight": [0.1, 0.2, 0.3, 0.4],
        }
    )
    # missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="missing_group", value="value", weights="weight"
        )

    # missing weights column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="brand", value="value", weights="missing_weight"
        )

    # missing value column raises no error in source, so test missing value column raises KeyError
    # This is not explicitly handled, but indexing will raise KeyError
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="brand", value="missing_value", weights="weight"
        )
