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


def test_weigthed_average_edge_cases():
    # Edge case 1: zero total weight
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df1, weights1)

    # Edge case 2: missing column in df that's in weights
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 0.5, "B": 0.5}  # 'B' missing in df
    with pytest.raises(KeyError):
        weigthed_average(df2, weights2)

    # Edge case 3: NaN and Inf values in df
    df3 = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [np.inf, 2, 3],
        }
    )
    weights3 = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df3, weights3)
    expected = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)

    # Edge case 4: weights with NaN or Inf
    df4 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights4_nan = {"A": np.nan, "B": 1.0}
    result_nan = weigthed_average(df4, weights4_nan)
    expected_nan = (df4["A"] * np.nan + df4["B"] * 1.0) / (np.nan + 1.0)
    pd.testing.assert_series_equal(result_nan, expected_nan)

    weights4_inf = {"A": np.inf, "B": 1.0}
    result_inf = weigthed_average(df4, weights4_inf)
    expected_inf = (df4["A"] * np.inf + df4["B"] * 1.0) / (np.inf + 1.0)
    pd.testing.assert_series_equal(result_inf, expected_inf)