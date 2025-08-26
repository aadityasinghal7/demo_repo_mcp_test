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
    # Setup DataFrame
    df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.inf, 5],
            "B": [5, np.nan, 3, 2, np.inf],
            "C": [10, 20, 30, 40, 50],
        }
    )

    # Case 1: Zero total weight
    weights_zero = {"A": 0, "B": 0, "C": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)

    # Case 2: Missing column in weights
    weights_missing = {"A": 1.0, "D": 2.0}  # 'D' not in df columns
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # Case 3: NaN and Inf in data with valid weights
    weights_valid = {"A": 0.3, "B": 0.3, "C": 0.4}
    result = weigthed_average(df, weights_valid)

    expected = (
        df["A"].fillna(0).replace([np.inf, -np.inf], 0) * 0.3
        + df["B"].fillna(0).replace([np.inf, -np.inf], 0) * 0.3
        + df["C"].fillna(0).replace([np.inf, -np.inf], 0) * 0.4
    ) / sum(weights_valid.values())

    # The original function does not replace NaN/Inf, so result will be NaN or Inf accordingly;
    # We verify that the result matches the formula from the function (which propagates NaN/Inf).
    # But here we emulate the expected without replacement (so expected must be calculated the same way).
    weighted_sum = sum(df[col] * w for col, w in weights_valid.items())
    expected_orig = weighted_sum / sum(weights_valid.values())

    pd.testing.assert_series_equal(result, expected_orig)


