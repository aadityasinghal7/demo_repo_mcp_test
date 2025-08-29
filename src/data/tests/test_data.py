import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 6],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "X": (10 * 1 + 20 * 3) / (1 + 3),
            "Y": (30 * 2 + 40 * 2 + 50 * 6) / (2 + 2 + 6),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())