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


def test_groupweightedaverage_division_by_zero():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B"],
            "value": [1.0, 3.0, 10.0, 5.0],
            "wt": [0.0, 0.0, 2.0, 3.0],
        }
    )
    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")
    # For group A, sum of weights is 0 --> division by zero leads to inf or NaN
    # We expect that the result for group A is inf or NaN; check via isna or isinf
    assert "A" in result.index and "B" in result.index

    # Group A sum weights is 0, so result should be inf or NaN (depending on pandas behavior)
    a_val = result.loc["A"]
    b_val = result.loc["B"]

    # Check that group A result is inf or nan
    assert pd.isna(a_val) or np.isinf(a_val), f"Expected NaN or Inf for group with zero weights, got {a_val}"

    # For group B, weighted average = (2*10 + 3*5)/(2+3) = (20+15)/5 = 7.0
    expected_b = 7.0
    assert np.isclose(b_val, expected_b), f"Expected {expected_b} for group B, got {b_val}"