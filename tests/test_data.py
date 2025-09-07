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
            ],  # group A weights sum to 0, group B weights sum to 0
        }
    )

    # Calculate groupweightedaverage
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For group A and B where sum of weights is zero, result will be division by zero, producing inf or nan
    # Check that values for groups with zero weight sum are either NaN or inf
    assert pd.isna(result.loc["A"]) or result.loc["A"] in [float("inf"), float("-inf")]
    assert pd.isna(result.loc["B"]) or result.loc["B"] in [float("inf"), float("-inf")]
