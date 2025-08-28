import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weights": [1, 2, 1, 0, 2],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")

    expected = pd.Series(
        {
            "X": (10 * 1 + 20 * 2) / (1 + 2),  # (10 + 40) / 3 = 50/3 ≈ 16.6667
            "Y": (30 * 1 + 40 * 0 + 50 * 2) / (1 + 0 + 2),  # (30 + 0 + 100)/3 = 130/3 ≈ 43.3333
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False)