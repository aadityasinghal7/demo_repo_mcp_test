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
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df1, weights1)

    # Case 2: missing columns in df should raise KeyError
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 1.0, "B": 1.0}  # 'B' missing in df
    with pytest.raises(KeyError):
        weigthed_average(df2, weights2)

    # Case 3: NaN and Inf values propagate correctly
    df3 = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [np.inf, 1.0, 2.0],
        }
    )
    weights3 = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df3, weights3)
    expected = (df3["A"] * 0.5 + df3["B"] * 0.5) / 1.0
    # The result will contain inf or nan at respective positions
    assert result.equals(expected), f"Expected {expected}, but got {result}"
