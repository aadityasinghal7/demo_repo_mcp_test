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
    # Zero total weight
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights = {"A": 0.0, "B": 0.0}
    try:
        _ = weigthed_average(df, weights)
        assert False, "Zero total weight should raise ZeroDivisionError"
    except ZeroDivisionError:
        pass

    # Missing column in weights
    weights = {"A": 1.0, "C": 1.0}  # "C" not in df
    try:
        _ = weigthed_average(df, weights)
        assert False, "Missing column in dataframe should raise KeyError"
    except KeyError:
        pass

    # NaN and Inf values
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 2.0, -np.inf]})
    weights = {"A": 0.5, "B": 0.5}
    # Replace inf with nan then fill nan with 0 as per README recommendation
    df_clean = df.replace([np.inf, -np.inf], pd.NA).fillna(0)
    result = weigthed_average(df_clean, weights)
    expected = (df_clean["A"] * 0.5 + df_clean["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)
