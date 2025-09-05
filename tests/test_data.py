import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_edge_cases():
    # Case 1: Empty weights dict
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)

    # Case 2: Weights sum to zero
    weights = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)

    # Case 3: DataFrame columns contain NaN values
    df_nan = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, 6]})
    weights = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights)
    expected_nan = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result_nan, expected_nan)

    # Case 4: DataFrame columns contain Inf values
    df_inf = pd.DataFrame({"A": [1, np.inf, 3], "B": [4, 5, -np.inf]})
    weights = {"A": 0.3, "B": 0.7}
    result_inf = weigthed_average(df_inf, weights)
    expected_inf = (df_inf["A"] * 0.3 + df_inf["B"] * 0.7) / (0.3 + 0.7)
    pd.testing.assert_series_equal(result_inf, expected_inf)
