import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_and_error_cases():
    # Basic correct weighted average
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 3, 4],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {"x": (10 * 1 + 20 * 2) / (1 + 2), "y": (30 * 3 + 40 * 4) / (3 + 4)}
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

    # Zero weights for a group -> result should be NaN (division by zero)
    df_zero_weights = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [5, 10, 15, 20],
            "weight": [0, 0, 1, 1],
        }
    )
    result_zero = groupweightedaverage(
        df_zero_weights, groupby="group", value="value", weights="weight"
    )
    expected_zero = pd.Series(
        {"a": np.nan, "b": (15 * 1 + 20 * 1) / (1 + 1)}  # sum of weights zero -> NaN
    )
    pd.testing.assert_series_equal(result_zero.sort_index(), expected_zero.sort_index())

    # Missing groups (group with no rows) should not appear in output
    df_missing = pd.DataFrame(
        {"group": ["x", "x", "y"], "value": [1, 2, 3], "weight": [1, 1, 1]}
    )
    # group 'z' does not exist
    result_missing = groupweightedaverage(
        df_missing, groupby="group", value="value", weights="weight"
    )
    assert "z" not in result_missing.index
    expected_missing = pd.Series({"x": (1 * 1 + 2 * 1) / 2, "y": 3 / 1})
    pd.testing.assert_series_equal(
        result_missing.sort_index(), expected_missing.sort_index()
    )

    # Non-numeric values in value or weights columns should raise error
    df_non_numeric = pd.DataFrame(
        {"group": ["x", "x"], "value": [10, "nan"], "weight": [1, 2]}
    )
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_non_numeric, groupby="group", value="value", weights="weight"
        )

    df_non_numeric_weight = pd.DataFrame(
        {"group": ["x", "x"], "value": [10, 20], "weight": [1, "bad"]}
    )
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_non_numeric_weight, groupby="group", value="value", weights="weight"
        )
