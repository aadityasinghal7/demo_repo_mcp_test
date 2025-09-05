import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    # Create a simple DataFrame with groups and numeric values and weights
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 3, 4, 5],
        }
    )

    # Calculate expected weighted averages manually per group
    # For group X: weighted average = (10*1 + 20*2) / (1 + 2) = (10 + 40) / 3 = 50 / 3 = 16.666...
    # For group Y: weighted average = (30*3 + 40*4 + 50*5) / (3 + 4 + 5) = (90 + 160 + 250) / 12 = 500 / 12 = 41.666...
    expected = pd.Series(
        {
            "X": (10 * 1 + 20 * 2) / (1 + 2),
            "Y": (30 * 3 + 40 * 4 + 50 * 5) / (3 + 4 + 5),
        }
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(
        result.sort_index(), expected.sort_index(), check_names=False
    )
