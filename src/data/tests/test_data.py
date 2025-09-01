import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


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


def test_weigthed_average_edge_cases():
    # Edge case: zero total weight
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        _ = weigthed_average(df, weights_zero)

    # Edge case: missing columns in df that weights refer to
    weights_missing_cols = {"A": 0.5, "C": 0.5}  # "C" does not exist in df
    with pytest.raises(KeyError):
        _ = weigthed_average(df, weights_missing_cols)

    # Edge case: NaN and Inf values in the DataFrame
    df_nan_inf = pd.DataFrame(
        {
            "A": [np.nan, 2, 3],
            "B": [np.inf, 4, -np.inf],
        }
    )
    weights_valid = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df_nan_inf, weights_valid)

    # Manually calculate expected result
    expected = (df_nan_inf["A"] * 0.5 + df_nan_inf["B"] * 0.5) / (0.5 + 0.5)
    # Check that result matches expected including NaN and Inf
    pd.testing.assert_series_equal(result, expected)
