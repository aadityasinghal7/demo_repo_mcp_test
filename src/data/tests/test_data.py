import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_edge_cases():
    # Case 1: zero total weight (should raise ZeroDivisionError)
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)

    # Case 2: missing columns in df (key in weights not in df)
    df_missing = pd.DataFrame({"A": [1, 2]})
    weights_missing = {"A": 1.0, "B": 2.0}  # "B" not in df_missing
    with pytest.raises(KeyError):
        weigthed_average(df_missing, weights_missing)

    # Case 3: NaN values in df
    df_nan = pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 4.0]})
    weights = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights)
    expected_nan = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / sum(weights.values())
    pd.testing.assert_series_equal(result_nan, expected_nan)

    # Case 4: Inf/-Inf values in df
    df_inf = pd.DataFrame({"A": [1.0, np.inf], "B": [-np.inf, 4.0]})
    weights = {"A": 0.5, "B": 0.5}
    result_inf = weigthed_average(df_inf, weights)
    expected_inf = (df_inf["A"] * 0.5 + df_inf["B"] * 0.5) / sum(weights.values())
    pd.testing.assert_series_equal(result_inf, expected_inf)
