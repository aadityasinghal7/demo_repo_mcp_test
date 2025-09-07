import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    # Create a simple DataFrame with groups, values, and weights
    data = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 1, 1],
        }
    )

    # Manually calculate expected weighted averages per group
    # For group X: weighted average = (10*1 + 20*3) / (1+3) = (10 + 60) / 4 = 70 / 4 = 17.5
    # For group Y: weighted average = (30*2 + 40*1 + 50*1) / (2+1+1) = (60 + 40 + 50) / 4 = 150 / 4 = 37.5
    expected = pd.Series({"X": 17.5, "Y": 37.5})

    result = groupweightedaverage(
        data, groupby="group", value="value", weights="weight"
    )

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
