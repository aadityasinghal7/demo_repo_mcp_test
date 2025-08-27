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


def test_groupweightedaverage_nan_and_inf_handling():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, np.nan, np.inf, 4.0, 5.0, -np.inf],
            "wt": [1.0, 2.0, 3.0, np.nan, np.inf, -np.inf],
        }
    )

    # Following the repo doc, NaN and Inf propagate/arithmetic as is
    # We will test what groupweightedaverage returns without any pre-cleaning

    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")

    # Compute expected manually:

    # brand A:
    # values = [1.0, NaN], weights = [1.0, 2.0]
    # sum_weight_contrib = 1.0*1.0 + NaN*2.0 = NaN
    # sum_weights = 1.0 + 2.0 = 3.0
    # => NaN / 3.0 = NaN

    # brand B:
    # values = [inf, 4.0], weights = [3.0, NaN]
    # sum_weight_contrib = inf*3.0 + 4.0*NaN = inf + NaN = NaN (because NaN + anything = NaN)
    # sum_weights = 3.0 + NaN = NaN
    # inf/NaN = NaN

    # brand C:
    # values = [5.0, -inf], weights = [inf, -inf]
    # sum_weight_contrib = 5.0*inf + (-inf)*(-inf) = inf + inf*inf?
    # inf*inf = inf
    # sum_weight_contrib = inf + inf = inf
    # sum_weights = inf + (-inf) = NaN (inf + -inf = undefined)
    # inf / NaN = NaN

    expected = pd.Series(
        [np.nan, np.nan, np.nan], index=["A", "B", "C"], dtype="float64"
    )

    pd.testing.assert_series_equal(result, expected)


