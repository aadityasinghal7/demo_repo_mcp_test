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
    # 1. Zero total weight
    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights1 = {"A": 0.0, "B": 0.0}
    try:
        _ = weigthed_average(df1, weights1)
        assert False, "Expected ZeroDivisionError when total weight is zero"
    except ZeroDivisionError:
        pass

    # 2. Missing columns in DataFrame
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 3],
        }
    )
    weights2 = {"A": 1.0, "B": 1.0}  # B missing from df2
    try:
        _ = weigthed_average(df2, weights2)
        assert False, "Expected KeyError when weight references column not in DataFrame"
    except KeyError:
        pass

    # 3. NaN and Inf values
    df3 = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf],
            "B": [4.0, 5.0, np.nan, -np.inf],
        }
    )
    weights3 = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df3, weights3)
    expected = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected, check_names=False)
