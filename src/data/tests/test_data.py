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
    # Create DataFrame with NaN and infinite values
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf, 5.0],
            "B": [2.0, 4.0, np.nan, 8.0, -np.inf],
        }
    )
    # weights include normal, NaN and infinite values
    weights = {"A": 1.0, "B": np.nan}

    # The weigthed_average function does not internally clean NaN/Inf so:
    # - NaNs in weights or data will propagate to the output
    # - infinite values will propagate as well, resulting in inf or nan outputs

    # In one scenario, weights with NaN will cause the sum of weights to be nan,
    # thus result would be nan series.
    result = weigthed_average(df, weights)
    # because sum(weights.values()) is nan, result should be all nan
    assert result.isna().all(), f"Expected all NaN result when weights contain NaN, got {result}"

    # Replace weights to test with infinite weight values
    weights_inf = {"A": 1.0, "B": np.inf}
    result_inf = weigthed_average(df, weights_inf)

    # The weighted sum will include inf * some values, expect inf or nan in result
    # Because B contains -inf and NaN, multiplied by inf weights, result may be nan or inf
    # We check that the result contains either inf or nan values
    assert result_inf.isin([np.inf, -np.inf]).any() or result_inf.isna().any(), (
        "Expected result to contain inf, -inf or NaN due to infinite weights or data"
    )

    # Now test with cleaned data and weights replacing NaN and inf with 0 as recommended
    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    weights_clean = {"A": 1.0, "B": 0.5}
    result_clean = weigthed_average(df_clean, weights_clean)

    # Expected: numeric values without NaN or inf
    assert not result_clean.isna().any(), "Expected no NaN in result with cleaned inputs"
    assert not result_clean.isin([np.inf, -np.inf]).any(), "Expected no inf in result with cleaned inputs"

    # Compute expected result manually
    total_weight = sum(weights_clean.values())
    expected = (df_clean["A"] * weights_clean["A"] + df_clean["B"] * weights_clean["B"]) / total_weight
    pd.testing.assert_series_equal(result_clean, expected)