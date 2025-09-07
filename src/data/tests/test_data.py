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


def test_groupweightedaverage_zero_weights():
    # Create a DataFrame with groups where weights are zero causing division by zero situations
    df = pd.DataFrame({
        "group": ["x", "x", "y", "y", "z", "z"],
        "value": [10, 20, 30, 40, 50, 60],
        "weight": [0, 0, 0, 0, 1, 2]  # groups x and y have zero total weight
    })

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Expected output:
    # For group 'x' and 'y', the sum of weights is zero → result should be NaN (division by zero)
    # For group 'z', normal weighted average should be computed
    expected = pd.Series(
        {
            "x": np.nan,
            "y": np.nan,
            "z": (50*1 + 60*2) / (1+2)
        }
    )

    pd.testing.assert_series_equal(result, expected)


