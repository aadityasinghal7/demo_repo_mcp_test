import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_zero_weights_sum():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 0, 0],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series([float("nan"), float("nan")], index=["A", "B"])
    pd.testing.assert_series_equal(result, expected)
