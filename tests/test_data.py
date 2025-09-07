import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_nan_inf_values():
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf, 5.0],
            "B": [2.0, 4.0, np.nan, 8.0, -np.inf],
            "C": [np.inf, 1.0, 3.0, 4.0, np.nan],
        }
    )
    weights = {"A": 1.0, "B": np.nan, "C": np.inf}

    # According to pandas semantics, arithmetic with NaN or Inf will propagate NaNs/Inf.
    # We compute expected results similarly.
    total_weight = sum(weights.values())
    weighted_sum = sum(df[col] * weight for col, weight in weights.items())
    expected = weighted_sum / total_weight

    result = weigthed_average(df, weights)

    # Use pandas testing to handle NaNs and Infs properly
    pd.testing.assert_series_equal(result, expected)
