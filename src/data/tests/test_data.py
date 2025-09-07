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


def test_groupweightedaverage_zero_sum_weights():
    # Create a DataFrame where weights sum to zero within each group
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "weights": [1.0, -1.0, 0.0, 0.0],
        }
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")

    # sum of weights in group 'x' = 0 and group 'y' = 0
    # Expect result to be inf, -inf or NaN; usually division by zero leads to NaN or +/-inf in numpy
    # Check the results for NaN or infinite values, i.e. no exceptions raised
    assert all(
        np.isnan(x) or np.isinf(x) for x in result
    ), "Expected NaN or infinite values when sum of weights per group is zero"
