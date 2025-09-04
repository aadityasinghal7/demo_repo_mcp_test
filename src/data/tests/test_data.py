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
    # Edge case 1: Zero total weight (should raise ZeroDivisionError)
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights_zero = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)

    # Edge case 2: Missing column in DataFrame (should raise KeyError)
    weights_missing = {"A": 0.5, "C": 0.5}  # 'C' missing
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # Edge case 3: NaN values in DataFrame, expect NaN results propagated
    df_nan = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, np.nan],
        }
    )
    weights_nan = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights_nan)
    expected_nan = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / 1.0
    pd.testing.assert_series_equal(result_nan, expected_nan)

    # Edge case 4: Inf values in DataFrame
    df_inf = pd.DataFrame(
        {
            "A": [1.0, np.inf, 3.0],
            "B": [4.0, 5.0, -np.inf],
        }
    )
    weights_inf = {"A": 0.4, "B": 0.6}
    result_inf = weigthed_average(df_inf, weights_inf)
    expected_inf = (df_inf["A"] * 0.4 + df_inf["B"] * 0.6) / 1.0
    pd.testing.assert_series_equal(result_inf, expected_inf)
