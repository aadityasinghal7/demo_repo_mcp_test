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


def test_groupweightedaverage_zero_total_weight():
    import pandas as pd

    from src.data.data_processing import groupweightedaverage

    # Create a DataFrame where weights sum to zero for at least one group
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "weight": [
                1,
                -1,
                0,
                0,
            ],  # For group A, weights sum to 0; for group B, sum to 0
        }
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Since division by zero would produce inf or NaN, check these cases explicitly
    assert (
        pd.isna(result["A"])
        or result["A"] == float("inf")
        or result["A"] == float("-inf")
    )
    assert (
        pd.isna(result["B"])
        or result["B"] == float("inf")
        or result["B"] == float("-inf")
    )
