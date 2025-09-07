import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B", "C"],
            "value": [10, 20, 30, 40, 50, 60],
            "weight": [1, 2, 1, 2, 3, 1],
        }
    )
    result = groupweightedaverage(
        data, groupby="group", value="value", weights="weight"
    )

    expected = pd.Series(
        {
            "A": (10 * 1 + 20 * 2) / (1 + 2),
            "B": (30 * 1 + 40 * 2 + 50 * 3) / (1 + 2 + 3),
            "C": (60 * 1) / 1,
        }
    )

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
