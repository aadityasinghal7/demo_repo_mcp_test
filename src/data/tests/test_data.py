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
    # Edge case 1: zero total weight -> expect division by zero yields inf or NaN
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights_zero = {"A": 0.0, "B": 0.0}
    result_zero_weight = weigthed_average(df, weights_zero)
    # sum of weights zero -> weighted_sum / 0, expect inf or nan values
    assert result_zero_weight.isin([np.inf, -np.inf, np.nan]).all() or result_zero_weight.isna().all()

    # Edge case 2: missing columns in df -> should raise KeyError
    weights_missing = {"A": 0.5, "C": 0.5}  # C not in df
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # Edge case 3: NaN and Inf values in df
    df_nan_inf = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [np.inf, 5, -np.inf],
        }
    )
    weights = {"A": 0.6, "B": 0.4}
    result_nan_inf = weigthed_average(df_nan_inf, weights)
    expected = (df_nan_inf["A"] * 0.6 + df_nan_inf["B"] * 0.4) / (0.6 + 0.4)
    # Use numpy to assert close with equal_nan True, also handle inf
    assert np.allclose(result_nan_inf.replace([np.inf, -np.inf], np.nan), expected.replace([np.inf, -np.inf], np.nan), equal_nan=True)


