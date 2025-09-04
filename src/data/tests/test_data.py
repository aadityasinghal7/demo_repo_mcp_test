import numpy as np
import pandas as pd

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
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    weights_zero = {"A": 0.0, "B": 0.0}
    try:
        _ = weigthed_average(df, weights_zero)
        assert False, "Expected ZeroDivisionError due to zero total weight"
    except ZeroDivisionError:
        pass

    # Edge case: missing columns in df for given weights
    df_missing = pd.DataFrame(
        {
            "A": [1.0, 2.0],
        }
    )
    weights_missing = {"A": 1.0, "B": 2.0}
    try:
        _ = weigthed_average(df_missing, weights_missing)
        assert False, "Expected KeyError due to missing column in df"
    except KeyError:
        pass

    # Edge case: NaN and Inf values in dataframe
    df_nan_inf = pd.DataFrame(
        {
            "A": [np.nan, 2.0, 3.0],
            "B": [1.0, np.inf, -np.inf],
        }
    )
    weights_valid = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df_nan_inf, weights_valid)
    expected = (df_nan_inf["A"] * 0.5 + df_nan_inf["B"] * 0.5) / (0.5 + 0.5)
    # compare with expected handling NaN and Inf
    # Use pandas testing assert_series_equal with check_names=False and check_dtype=False
    pd.testing.assert_series_equal(
        result, expected, check_names=False, check_dtype=False
    )
