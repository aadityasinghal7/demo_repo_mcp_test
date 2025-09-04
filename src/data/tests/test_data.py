import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B", "C"],
            "val": [10, 20, 30, 40, 50, 60],
            "weights": [1, 2, 3, 4, 5, 6],
        }
    )
    # Manually compute group weighted averages:
    # For group A: (10*1 + 20*2) / (1+2) = (10 + 40) / 3 = 50/3 ≈ 16.6667
    # For group B: (30*3 + 40*4 + 50*5) / (3+4+5) = (90 + 160 + 250) / 12 = 500/12 ≈ 41.6667
    # For group C: (60*6) / 6 = 360/6 = 60
    expected = pd.Series(
        {
            "A": (10 * 1 + 20 * 2) / (1 + 2),
            "B": (30 * 3 + 40 * 4 + 50 * 5) / (3 + 4 + 5),
            "C": (60 * 6) / 6,
        }
    )
    result = groupweightedaverage(df, groupby="group", value="val", weights="weights")
    pd.testing.assert_series_equal(result, expected, check_names=False)
