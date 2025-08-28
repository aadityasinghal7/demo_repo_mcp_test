import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_zero_total_weight_per_group():
    # Create a DataFrame where the sum of weights per group is zero
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "weights": [0, 0, 0, 0],  # sum per group = 0
        }
    )

    # The expected behavior is to produce inf or NaN or avoid division error
    # Here, we check that the function does not raise an exception and returns appropriate output (NaN)
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")

    # Both groups have zero total weight, so division 0/0 => NaN
    expected = pd.Series([np.nan, np.nan], index=pd.Index(["A", "B"], name="group"))

    pd.testing.assert_series_equal(result, expected)


