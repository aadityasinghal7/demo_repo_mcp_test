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
    # Case 1: zero total weight -> division by zero (should raise ZeroDivisionError)
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0.0, "B": 0.0}
    try:
        _ = weigthed_average(df1, weights1)
        assert False, "Expected ZeroDivisionError for zero total weight"
    except ZeroDivisionError:
        pass

    # Case 2: missing column in df raises KeyError when accessed
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 1.0, "B": 2.0}  # 'B' missing
    try:
        _ = weigthed_average(df2, weights2)
        assert False, "Expected KeyError for missing column"
    except KeyError:
        pass

    # Case 3: NaN and Inf values propagate correctly (no raise, result contains NaN/Inf as expected)
    df3 = pd.DataFrame(
        {
            "A": [np.nan, 2, 3],
            "B": [np.inf, 5, 6],
        }
    )
    weights3 = {"A": 0.5, "B": 0.5}
    result3 = weigthed_average(df3, weights3)
    expected3 = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result3, expected3)
