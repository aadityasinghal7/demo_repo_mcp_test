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
    # Create a DataFrame where weights sum to zero for each group
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 0, 0],
        }
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Expect the result to contain NaN for both groups because denominator is zero
    assert result.index.tolist() == ["x", "y"]
    assert result.isna().all()
