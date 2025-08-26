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
    # Case 1: Zero total weight should raise ZeroDivisionError
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights_zero = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df1, weights_zero)

    # Case 2: Missing weighted column - KeyError expected when weights include missing column
    df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights_missing = {"A": 0.5, "C": 0.5}  # "C" missing from df2
    with pytest.raises(KeyError):
        weigthed_average(df2, weights_missing)

    # Case 3: NaN and Inf values in df should propagate accordingly
    df3 = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, 4.0],
            "B": [np.inf, 2.0, 3.0, -np.inf],
        }
    )
    weights3 = {"A": 0.6, "B": 0.4}
    result3 = weigthed_average(df3, weights3)
    expected3 = (df3["A"] * 0.6 + df3["B"] * 0.4) / (0.6 + 0.4)
    pd.testing.assert_series_equal(result3, expected3)

    # Case 4: NaN and Inf replaced with 0 should work as normal numbers
    df4 = df3.replace([np.inf, -np.inf], np.nan).fillna(0)
    weights4 = {"A": 0.7, "B": 0.3}
    result4 = weigthed_average(df4, weights4)
    expected4 = (df4["A"] * 0.7 + df4["B"] * 0.3) / (0.7 + 0.3)
    pd.testing.assert_series_equal(result4, expected4)