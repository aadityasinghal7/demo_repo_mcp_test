import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_zero_weights():
    # Create a DataFrame where one group has zero total weight
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "weight": [
                1,
                -1,
                0,
                0,
            ],  # Group A sums to zero weight (1 + -1), group B sums to zero
        }
    )

    # Call groupweightedaverage and verify no error and expected output for zero weights
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For groups with zero total weight, division by zero could produce inf or NaN
    # Assert the indices are as expected
    assert set(result.index) == {"A", "B"}
    # For group A and B total weights are zero, so result should be inf or NaN
    # Check that result values for groups with zero total weight are either NaN or infinite
    for val in result:
        assert (
            pd.isna(val)
            or val == float("inf")
            or val == float("-inf")
            or pd.isnull(val)
        )


# To run: pytest tests/test_data.py
