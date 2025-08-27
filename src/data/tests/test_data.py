import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average, groupweightedaverage


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


def test_groupweightedaverage_with_nan_and_inf():
    df = pd.DataFrame(
        {
            "brand": ["X", "X", "Y", "Y", "Z", "Z"],
            "value": [10.0, np.nan, np.inf, 5.0, 3.0, 7.0],
            "wt": [1.0, 2.0, 1.0, np.inf, np.nan, 1.0],
        }
    )

    # Expected behavior:
    # For brand "X":
    #   Weighted sum = (10 * 1) + (NaN * 2) = 10 + NaN = NaN
    #   Sum weights = 1 + 2 = 3
    #   Result: NaN / 3 = NaN
    # For brand "Y":
    #   Weighted sum = (Inf * 1) + (5 * Inf) = Inf + Inf = Inf
    #   Sum weights = 1 + Inf = Inf
    #   Result: Inf / Inf = NaN (pandas treats Inf/Inf as NaN)
    # For brand "Z":
    #   Weighted sum = (3 * NaN) + (7 * 1) = NaN + 7 = NaN
    #   Sum weights = NaN + 1 = NaN
    #   Result: NaN / NaN = NaN

    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")

    expected = pd.Series(
        {
            "X": np.nan,
            "Y": np.nan,
            "Z": np.nan,
        }
    )

    pd.testing.assert_series_equal(result, expected, check_names=False, check_dtype=False)
