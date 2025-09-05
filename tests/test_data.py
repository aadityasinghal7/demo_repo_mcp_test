import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic():
    # Sample data with groups and known values
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 6],
        }
    )
    # Expected result manually computed:
    # group x: (10*1 + 20*3) / (1+3) = (10 + 60) / 4 = 70 / 4 = 17.5
    # group y: (30*2 + 40*2 + 50*6) / (2+2+6) = (60 + 80 + 300) / 10 = 440 / 10 = 44.0
    expected = pd.Series({"x": 17.5, "y": 44.0})

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
