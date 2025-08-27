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


def test_groupweightedaverage_zero_total_weight_group():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 1, -1],  # sum of weights for group Y is 0, for X is also 0
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For group X: sum weights = 0 -> expect inf or nan; 
    # For group Y: sum weights = 0 (1 + -1) -> expect inf or nan
    # Check if result contains infinities or NaNs appropriately
    assert np.isnan(result["X"]) or np.isinf(result["X"]), f"Expected NaN or Inf for group X, got {result['X']}"
    assert np.isnan(result["Y"]) or np.isinf(result["Y"]), f"Expected NaN or Inf for group Y, got {result['Y']}"