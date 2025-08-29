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
    # Case 1: zero total weight should raise ZeroDivisionError
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)

    # Case 2: missing column in DataFrame should raise KeyError
    weights = {"A": 1.0, "C": 2.0}  # 'C' not in df
    with pytest.raises(KeyError):
        weigthed_average(df, weights)

    # Case 3: NaN and Inf in DataFrame propagate in results
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, np.inf],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df, weights)
    expected = (df["A"] * 0.5 + df["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)


