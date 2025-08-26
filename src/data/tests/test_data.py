import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average, groupweightedaverage


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


def test_groupweightedaverage_nan_and_inf_values():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "z", "z"],
            "value": [1.0, np.nan, 3.0, np.inf, -np.inf, 5.0],
            "weight": [1.0, 2.0, np.nan, 4.0, 1.0, np.inf],
        }
    )

    # According to documented assumptions:
    # NaN propagates in arithmetic => weighted sums with NaNs become NaN
    # Inf/-Inf should be replaced before computing, but here test natural behavior
    # We'll test the raw output, expecting NaNs where they propagate
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Manually compute expected results:
    # For group 'x':
    # values = [1.0, NaN]
    # weights = [1.0, 2.0]
    # weighted sum = 1.0*1.0 + NaN*2.0 = NaN
    # sum weights = 1.0 + 2.0 = 3.0
    # weighted average = NaN / 3.0 = NaN

    # For group 'y':
    # values = [3.0, inf]
    # weights = [NaN, 4.0]
    # weighted sum = NaN*3.0 + 4.0*inf = NaN + inf = NaN (inf + NaN = NaN in NumPy/pandas)
    # sum weights = NaN + 4.0 = NaN
    # weighted average = NaN / NaN = NaN

    # For group 'z':
    # values = [-inf, 5.0]
    # weights = [1.0, inf]
    # weighted sum = 1.0*(-inf) + inf*5.0 = -inf + inf = NaN (inf - inf = NaN)
    # sum weights = 1.0 + inf = inf
    # weighted average = NaN / inf = NaN

    expected = pd.Series([np.nan, np.nan, np.nan], index=["x", "y", "z"])

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
        check_dtype=False,
        check_less_precise=False,
        obj="groupweightedaverage with NaN and Inf values",
        check_exact=False,
        rtol=1e-5,
        atol=1e-8,
    )