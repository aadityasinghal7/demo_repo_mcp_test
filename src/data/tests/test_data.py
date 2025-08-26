import numpy as np
import pandas as pd
import pytest
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
    df = pd.DataFrame({
        "brand": ["A", "A", "B", "B", "C"],
        "value": [1.0, np.nan, np.inf, 4.0, 5.0],
        "wt": [1.0, 2.0, 3.0, np.nan, np.inf]
    })

    # The output for groups should handle NaN and Inf as per pandas arithmetic:
    # - For group A:
    #     value * wt = [1.0*1.0, NaN*2.0] => [1.0, NaN]
    #     sum = NaN (because of NaN)
    #     sum_weights = 1.0 + 2.0 = 3.0
    #     weighted average = NaN / 3.0 = NaN
    # - For group B:
    #     value * wt = [inf*3.0, 4.0*NaN] => [inf, NaN]
    #     sum = inf (sum ignores NaN and inf according to pandas sum behavior is inf)
    #     sum_weights = 3.0 + NaN = NaN (sum with NaN is NaN)
    #     result = inf / NaN = NaN
    # - For group C:
    #     value * wt = 5.0 * inf = inf
    #     sum = inf
    #     sum_weights = inf
    #     result = inf / inf = NaN (by numpy semantics)

    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")

    # Build expected Series with index matching groups
    expected = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.Index(["A", "B", "C"], name="brand"),
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected, check_names=False, check_dtype=False)
