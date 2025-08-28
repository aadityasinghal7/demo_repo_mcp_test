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


def test_groupweightedaverage_non_numeric_values_or_weights():
    df_value_non_numeric = pd.DataFrame(
        {
            "brand": ["A", "A", "B"],
            "value": [1.0, "foo", 3.0],
            "wt": [1.0, 1.0, 2.0],
        }
    )
    df_weight_non_numeric = pd.DataFrame(
        {
            "brand": ["A", "A", "B"],
            "value": [1.0, 3.0, 10.0],
            "wt": [1.0, "bar", 2.0],
        }
    )

    # Expecting the function to raise TypeError or ValueError due to invalid ops with strings
    with pytest.raises((TypeError, ValueError)):
        groupweightedaverage(df_value_non_numeric, groupby="brand", value="value", weights="wt")

    with pytest.raises((TypeError, ValueError)):
        groupweightedaverage(df_weight_non_numeric, groupby="brand", value="value", weights="wt")