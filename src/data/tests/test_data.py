import numpy as np
import pandas as pd
import pytest

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


def test_groupweightedaverage_zero_sum_weights():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "weight": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    # Because sum_weights are zero for each group, resulting division yields inf or NaN.
    # Check that result contains only inf or NaN values corresponding to groups.
    assert result.index.equals(pd.Index(["x", "y"], name="group"))
    # The sum of weights is zero, so result should be infinite or NaN (NaN likely due to 0/0).
    assert result.isna().all() or np.isinf(result).all() or (result.isna() | np.isinf(result)).all()