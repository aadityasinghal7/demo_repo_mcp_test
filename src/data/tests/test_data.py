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
    # Case 1: Zero total weight -> should raise ZeroDivisionError or return inf/nan depending on behavior
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        _ = weigthed_average(df, weights_zero)

    # Case 2: NaN values in data
    df_nan = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, 6]})
    weights = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights)
    expected_nan = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / (0.5 + 0.5)
    # check if results contain NaN where expected
    assert result_nan.isna().equals(expected_nan.isna())
    # for non-NaN positions compare values
    non_na = ~expected_nan.isna()
    assert np.allclose(result_nan[non_na], expected_nan[non_na])

    # Case 3: Inf and -Inf values
    df_inf = pd.DataFrame({"A": [1, np.inf, 3], "B": [4, 5, -np.inf]})
    result_inf = weigthed_average(df_inf, weights)
    expected_inf = (df_inf["A"] * 0.5 + df_inf["B"] * 0.5) / (0.5 + 0.5)
    # Since pandas propagates inf in operations, result should equal expected_inf
    assert result_inf.equals(expected_inf)

    # Case 4: Non-numeric data in dataframe should raise in arithmetic or produce NaN
    df_str = pd.DataFrame({"A": [1, "a", 3], "B": [4, 5, 6]})
    with pytest.raises(TypeError):
        # This will raise because "a" cannot be multiplied by float weight
        _ = weigthed_average(df_str, weights)

    # Additionally test with coercing to numeric with errors='coerce'
    df_str["A"] = pd.to_numeric(df_str["A"], errors="coerce")
    result_coerce = weigthed_average(df_str, weights)
    expected_coerce = (df_str["A"] * 0.5 + df_str["B"] * 0.5) / (0.5 + 0.5)
    assert result_coerce.isna().equals(expected_coerce.isna())
    non_na = ~expected_coerce.isna()
    assert np.allclose(result_coerce[non_na], expected_coerce[non_na])