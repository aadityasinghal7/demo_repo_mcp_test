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


def test_groupweightedaverage_empty_groups():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "d"],  # group 'c' missing entries
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 1, 0, 0, 0],  # group b and d have zero total weight
        }
    )
    # Include group 'c' with no rows by reindexing after grouping
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    # Expected series: for group 'a' normal calculation,
    # for groups 'b' and 'd' division by zero error or NaN 
    # but groupweightedaverage does not raise in this case.
    # We expect division by zero warnings converted to NaN for weight sums of zero
    # For missing group 'c' no entry in result

    expected = pd.Series(
        {
            "a": (10*1 + 20*1) / (1 + 1),
            "b": np.nan,
            "d": np.nan,
        }
    )
    # Add NaNs for zero-weight groups manually for comparison
    result_filled = result.reindex(expected.index)
    # Assert group 'a' is correct
    assert np.isclose(result_filled["a"], expected["a"]), f"Expected {expected['a']} but got {result_filled['a']}"
    # Assert groups with zero weight have NaN result
    assert pd.isna(result_filled["b"]), "Expected NaN for group 'b' with zero total weight"
    assert pd.isna(result_filled["d"]), "Expected NaN for group 'd' with zero total weight"
    # Assert group 'c' is missing (not in result)
    assert "c" not in result.index, "Expected group 'c' to be missing (no entries)"