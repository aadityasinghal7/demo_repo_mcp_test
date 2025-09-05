import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_with_nan_and_inf_values():
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, np.inf],
            "B": [10.0, np.nan, 30.0, np.inf, -np.inf],
            "C": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}

    # The function does not do any cleaning, so the output will propagate NaN or Inf.
    # Calculate expected weighted sum and total weight
    total_weight = sum(weights.values())
    weighted_sum = (
        df["A"] * weights["A"] + df["B"] * weights["B"] + df["C"] * weights["C"]
    )
    expected = weighted_sum / total_weight

    result = weigthed_average(df, weights)
    pd.testing.assert_series_equal(result, expected)

    # Additionally test that NaN and Inf values remain in the result at expected positions
    assert result.isnull().sum() == expected.isnull().sum()
    assert np.isposinf(result).sum() == np.isposinf(expected).sum()
    assert np.isneginf(result).sum() == np.isneginf(expected).sum()
