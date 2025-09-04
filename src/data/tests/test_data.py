import numpy as np
import pandas as pd

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 1, 1, 2],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected_values = {
        "X": (10 * 1 + 20 * 2) / (1 + 2),  # (10 + 40) / 3 = 50 / 3 = 16.666...
        "Y": (30 * 1 + 40 * 1 + 50 * 2)
        / (1 + 1 + 2),  # (30 + 40 + 100) / 4 = 170 / 4 = 42.5
    }
    expected = pd.Series(expected_values)
    pd.testing.assert_series_equal(result, expected)
