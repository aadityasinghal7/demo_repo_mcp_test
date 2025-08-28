import numpy as np
import pandas as pd

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 4, 4],
        }
    )
    # For group X:
    # weighted average = (10*1 + 20*3) / (1 + 3) = (10 + 60) / 4 = 17.5
    # For group Y:
    # weighted average = (30*2 + 40*4 + 50*4) / (2 + 4 + 4) = (60 + 160 + 200) / 10 = 42.0
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series({"X": 17.5, "Y": 42.0})
    pd.testing.assert_series_equal(result, expected)