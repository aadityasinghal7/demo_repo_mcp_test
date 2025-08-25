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


def test_groupweightedaverage_non_numeric_data():
    # Non-numeric in 'value' column
    df_value_non_numeric = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2"],
            "value": [1, "x", 3],  # non-numeric in 'value'
            "weight": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(TypeError):
        groupweightedaverage(df_value_non_numeric, groupby="group", value="value", weights="weight")

    # Non-numeric in 'weights' column
    df_weight_non_numeric = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2"],
            "value": [1.0, 2.0, 3.0],
            "weight": [1.0, "y", 3.0],  # non-numeric in 'weights'
        }
    )
    with pytest.raises(TypeError):
        groupweightedaverage(df_weight_non_numeric, groupby="group", value="value", weights="weight")