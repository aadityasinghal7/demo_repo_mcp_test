import math

import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_with_nan_inf():
    df = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 4.0, np.inf], "B": [10.0, np.inf, 30.0, np.nan, 50.0]}
    )

    # Per README, replacing Inf/-Inf by NaN then fillna(0) recommended
    df_clean = df.replace([np.inf, -np.inf], pd.NA).fillna(0)

    weights = {"A": 0.6, "B": 0.4}

    # Compute expected manually:
    # weighted_sum per row = A * 0.6 + B * 0.4
    # total_weight = 1.0
    expected = df_clean["A"] * 0.6 + df_clean["B"] * 0.4

    result = weigthed_average(df_clean, weights)

    pd.testing.assert_series_equal(result, expected)


# This test can be run with:
# pytest tests/test_data.py -v
