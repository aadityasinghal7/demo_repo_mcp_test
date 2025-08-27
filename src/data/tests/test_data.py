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


def test_weigthed_average_with_nan_and_inf():
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf],
            "B": [4.0, 5.0, np.nan, -np.inf],
        }
    )
    weights = {"A": 0.6, "B": 0.4}

    # Using normal float arithmetic with NaN or inf will propagate these values
    result = weigthed_average(df, weights)

    # Check that result contains np.nan or np.inf as expected
    # Row 0: 1*0.6 + 4*0.4 = 0.6 + 1.6 = 2.2
    # Row 1: nan * 0.6 + 5 * 0.4 = nan + 2.0 = nan (propagates nan)
    # Row 2: 3 * 0.6 + nan * 0.4 = 1.8 + nan = nan
    # Row 3: inf * 0.6 + -inf * 0.4 = inf + -inf = nan (inf - inf => nan)
    expected = pd.Series([2.2, np.nan, np.nan, np.nan], dtype=float)

    pd.testing.assert_series_equal(result, expected, check_names=False)
