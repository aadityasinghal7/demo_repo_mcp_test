import numpy as np
import pandas as pd

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


def test_weigthed_average_with_nans_and_infs():
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, np.inf, 4.0],
            "B": [np.inf, 2.0, 3.0, np.nan],
            "C": [1.0, 2.0, np.nan, np.inf],
        }
    )
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    # According to the function:
    # weighted sum = 0.5 * A + 0.3 * B + 0.2 * C
    # total_weight = 1.0
    result = weigthed_average(df, weights)

    # Construct expected manually:
    # Row 0: 0.5*1.0 + 0.3*inf + 0.2*1.0 = inf (due to inf)
    # Row 1: 0.5*nan + 0.3*2.0 + 0.2*2.0 = nan (nan * 0.5 propagates)
    # Row 2: 0.5*inf + 0.3*3.0 + 0.2*nan = nan (nan propagates from 0.2*nan)
    # Row 3: 0.5*4.0 + 0.3*nan + 0.2*inf = nan (nan propagates from 0.3*nan)
    expected = pd.Series([np.inf, np.nan, np.nan, np.nan])

    pd.testing.assert_series_equal(result, expected, check_names=False, check_dtype=False)

