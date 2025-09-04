import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average():
    df = pd.DataFrame(
        {
            "A": [2, 0, 0],
            "B": [1, 2, 3],
        }
    )
    weights = {"A": 1.0, "B": 0.0}
    result = weigthed_average(df, weights)
    expected = pd.Series([2.0, 0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_weigthed_average_nan_inf_handling():
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf, 5.0],
            "B": [2.0, 4.0, np.inf, 8.0, np.nan],
            "C": [np.nan, 6.0, 7.0, 8.0, 9.0],
        }
    )
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = weigthed_average(df, weights)

    # Expected: weighted sum / total_weight = (sum of weighted values) / 1.0
    # Manual calculation taking into account pandas semantics:
    # For each row:
    # row 0: 1.0*0.5 + 2.0*0.3 + nan*0.2 = nan -> result is nan
    # row 1: nan*0.5 + 4.0*0.3 + 6.0*0.2 = nan
    # row 2: 3.0*0.5 + inf*0.3 + 7.0*0.2 = inf
    # row 3: inf*0.5 + 8.0*0.3 + 8.0*0.2 = inf
    # row 4: 5.0*0.5 + nan*0.3 + 9.0*0.2 = nan

    expected = pd.Series([np.nan, np.nan, np.inf, np.inf, np.nan])
    pd.testing.assert_series_equal(result, expected, check_names=False)
