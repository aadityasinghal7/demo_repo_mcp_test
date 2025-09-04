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
    # Zero total weight -> sum weights = 0, which will cause division by zero.
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)

    # Missing columns in df
    weights_missing = {"A": 0.5, "C": 0.5}  # 'C' missing in df
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # NaN and Inf values handled as normal (should propagate)
    df_nan_inf = pd.DataFrame(
        {
            "A": [1, np.nan, 3, np.inf],
            "B": [4, 5, np.nan, -np.inf],
        }
    )
    weights_nan_inf = {"A": 0.6, "B": 0.4}
    result_nan_inf = weigthed_average(df_nan_inf, weights_nan_inf)
    expected_nan_inf = (df_nan_inf["A"] * 0.6 + df_nan_inf["B"] * 0.4) / 1.0
    pd.testing.assert_series_equal(result_nan_inf, expected_nan_inf)
