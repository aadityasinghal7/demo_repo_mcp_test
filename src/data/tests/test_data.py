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


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 1, 1],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "x": (10*1 + 20*3) / (1 + 3),
            "y": (30*2 + 40*1 + 50*1) / (2 + 1 + 1),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


def test_weighted_average_zero_total_weight():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights = {"A": 0.0, "B": 0.0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)


def test_groupweightedaverage_zero_total_weight():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 0, 0],
        }
    )
    # When sum of weights per group is zero, division by zero should occur.
    with pytest.raises(ZeroDivisionError):
        _ = groupweightedaverage(df, groupby="group", value="value", weights="weight")


def test_weighted_average_non_numeric_columns():
    # DataFrame contains non-numeric data in one column
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],  # non-numeric
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    # Should raise error because we cannot multiply non-numeric
    with pytest.raises(TypeError):
        weigthed_average(df, weights)

    # Also test non-numeric weights
    df_numeric = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    # weights have non-numeric value
    weights_non_numeric = {"A": 0.5, "B": "invalid"}

    with pytest.raises(TypeError):
        weigthed_average(df_numeric, weights_non_numeric)


def test_weighted_average_nan_inf_handling():
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf, 5.0, -np.inf],
            "B": [2.0, 4.0, np.nan, 8.0, np.inf, -np.inf],
        }
    )
    # weights contain nan and inf values as well
    weights_nan_inf = {"A": 1.0, "B": np.nan}
    weights_inf = {"A": np.inf, "B": 1.0}

    # Case 1: weights with NaN - should propagate NaN or error on addition
    with pytest.raises(TypeError):
        _ = weigthed_average(df, weights_nan_inf)

    # Case 2: weights containing inf - result may contain NaN or inf, but no error
    result_inf_weights = weigthed_average(df, weights_inf)
    # Should be a Series with numeric (may contain inf or nan)
    assert isinstance(result_inf_weights, pd.Series)

    # Now replace inf/-inf with NaN and fill NaNs to simulate cleaning
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    weights_cleaned = {"A": 1.0, "B": 2.0}
    result_cleaned = weigthed_average(df_cleaned, weights_cleaned)
    expected_cleaned = (df_cleaned["A"] * 1.0 + df_cleaned["B"] * 2.0) / (1.0 + 2.0)
    pd.testing.assert_series_equal(result_cleaned, expected_cleaned)