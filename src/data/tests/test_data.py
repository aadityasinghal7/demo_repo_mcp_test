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


def test_groupweightedaverage_nan_inf_handling():
    # Create DataFrame with NaN and inf values in value and weights
    df = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "value": [1.0, np.nan, np.inf, 10.0, 5.0, -np.inf],
            "weight": [1.0, 2.0, 3.0, np.nan, np.inf, 1.0],
        }
    )

    # Expected behavior:
    # group g1: value = [1.0, nan], weight = [1.0, 2.0]
    #   weighted sum = 1.0*1.0 + nan*2.0 = nan
    #   sum weights = 1.0 + 2.0 = 3.0
    #   result = nan / 3.0 = nan
    #
    # group g2: value = [inf, 10.0], weight = [3.0, nan]
    #   weighted sum = inf*3.0 + 10.0*nan = inf + nan = nan
    #   sum weights = 3.0 + nan = nan
    #   result = nan / nan = nan
    #
    # group g3: value = [5.0, -inf], weight = [inf, 1.0]
    #   weighted sum = 5.0*inf + (-inf)*1.0 = inf + (-inf) = nan (undefined)
    #   sum weights = inf + 1.0 = inf
    #   result = nan / inf = nan

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Expected Series indexed by group
    expected = pd.Series(
        {
            "g1": np.nan,
            "g2": np.nan,
            "g3": np.nan,
        }
    )

    # Use pandas testing with check_names=False (no names on the returned series) and check_dtype=False
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False, check_dtype=False)
