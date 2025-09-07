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


def test_groupweightedaverage_handling_nan_inf():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C", "C"],
            "value": [1.0, np.nan, np.inf, 4.0, 5.0, -np.inf, 7.0],
            "weight": [1.0, 2.0, 3.0, np.nan, np.inf, 1.0, -np.inf],
        }
    )

    # Expected behavior: calculation done with pandas sum and division which excludes NaN by default
    # inf and -inf can cause sum to be inf/-inf or NaN if mixed, so calculation follows pandas semantics.
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "A": (1.0 * 1.0 + np.nan * 2.0)
            / (
                1.0 + 2.0
            ),  # sum of weight_contrib / sum of weights -> with nan weights or values as zeroed by sum?
            "B": (np.inf * 3.0 + 4.0 * np.nan) / (3.0 + np.nan),
            "C": (5.0 * np.inf + (-np.inf) * 1.0 + 7.0 * (-np.inf))
            / (np.inf + 1.0 + (-np.inf)),
        }
    )

    # Because NaNs are ignored in sum, NaN weights effectively drop those rows.
    # For group A:
    # weight_contrib: [1*1=1, nan*2=nan] sum -> 1
    # weights: [1,2] sum -> 3
    # result A: 1/3 = 0.3333
    # For group B:
    # weight_contrib: [inf*3=inf, 4*nan=nan] sum -> inf
    # weights: [3, nan] sum -> 3
    # result B: inf/3 = inf
    # For group C:
    # weight_contrib: [5*inf=inf, -inf*1= -inf, 7*(-inf) = -inf] sum = inf + (-inf) + (-inf) = NaN (inf -inf = nan)
    # weights: [inf, 1, -inf] sum = inf + 1 + (-inf) = NaN
    # result C = NaN / NaN = NaN

    expected = pd.Series({"A": 1 / 3, "B": np.inf, "C": np.nan})

    pd.testing.assert_series_equal(result, expected)
