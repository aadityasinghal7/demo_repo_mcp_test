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


def test_weigthed_average_nan_handling():
    # Data includes NaN values
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, 4.0],
            "B": [2.0, 3.0, np.nan, 5.0],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    # Compute weighted average manually with pandas semantics (NaNs propagate)
    weighted_sum = df["A"] * weights["A"] + df["B"] * weights["B"]
    total_weight = sum(weights.values())
    expected = weighted_sum / total_weight
    result = weigthed_average(df, weights)
    # Use pandas function to check NaN equality
    pd.testing.assert_series_equal(result, expected, check_names=False)