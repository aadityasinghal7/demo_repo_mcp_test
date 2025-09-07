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


def test_weigthed_average_nan_and_inf_handling():
    df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, np.inf],
            "B": [5, np.nan, 7, np.inf, 9],
            "C": [np.nan, 12, 13, 14, 15],
        }
    )
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}

    result = weigthed_average(df, weights)

    # Expected calculation:
    # weighted sum = 0.2*A + 0.3*B + 0.5*C (element-wise)
    # total weight = 1.0
    # Note: Pandas will propagate NaN if any value in row is NaN
    # Similarly, np.inf is preserved in operations.
    expected = (df["A"] * 0.2 + df["B"] * 0.3 + df["C"] * 0.5) / 1.0

    pd.testing.assert_series_equal(result, expected)
