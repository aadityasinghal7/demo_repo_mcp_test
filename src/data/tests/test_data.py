import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 10, 20, 30],
            "weight": [1, 2, 3, 4, 3],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    # expected calculation:
    # Group X: (10*1 + 20*2) / (1 + 2) = (10 + 40) / 3 = 50 / 3 = 16.6666667
    # Group Y: (10*3 + 20*4 + 30*3) / (3 + 4 + 3) = (30 + 80 + 90) / 10 = 200 / 10 = 20.0
    expected = pd.Series({"X": 50 / 3, "Y": 20.0})
    pd.testing.assert_series_equal(
        result.sort_index(), expected.sort_index(), rtol=1e-7, atol=1e-8
    )
