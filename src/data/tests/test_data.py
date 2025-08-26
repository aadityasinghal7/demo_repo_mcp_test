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
    # Case 1: zero total weight -> should raise ZeroDivisionError
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)

    # Case 2: missing columns in DataFrame -> should raise KeyError
    weights_missing = {"A": 1.0, "C": 1.0}  # C not in df
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # Case 3: NaN values in DataFrame, weights non-zero
    df_nan = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, np.nan],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights)
    expected_nan = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / 1.0
    pd.testing.assert_series_equal(result_nan, expected_nan)

    # Case 4: Inf values in DataFrame, weights non-zero
    df_inf = pd.DataFrame(
        {
            "A": [1.0, np.inf, 3.0],
            "B": [4.0, 5.0, -np.inf],
        }
    )
    weights = {"A": 0.6, "B": 0.4}
    result_inf = weigthed_average(df_inf, weights)
    expected_inf = (df_inf["A"] * 0.6 + df_inf["B"] * 0.4) / 1.0
    pd.testing.assert_series_equal(result_inf, expected_inf)
