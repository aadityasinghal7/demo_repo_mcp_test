import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 1, 1, 2],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "A": (10 * 1 + 20 * 2) / (1 + 2),
            "B": (30 * 1 + 40 * 1 + 50 * 2) / (1 + 1 + 2),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())