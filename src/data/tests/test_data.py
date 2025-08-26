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
            "A": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "B": [np.nan, 2.0, 3.0, 4.0, np.inf],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df, weights)
    expected = (df["A"] * weights["A"] + df["B"] * weights["B"]) / sum(weights.values())
    # Check equality that respects NaN and inf propagation
    pd.testing.assert_series_equal(result, expected)