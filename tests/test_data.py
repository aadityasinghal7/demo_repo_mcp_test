import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_zero_sum_weights_per_group():
    # Create a DataFrame with groups where the sum of weights is zero
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y"],
            "value": [10, 20, 30, 40],
            "weights": [1, -1, 0, 0],
        }
    )

    # Apply the groupweightedaverage function
    result = groupweightedaverage(df, groupby="group", value="value", weights="weights")

    # For group 'X', weights sum to zero (1 + -1 = 0), expect infinite or NaN results handled gracefully
    # For group 'Y', weights sum to zero (0 + 0 = 0) similarly
    # We expect result containing inf or nan where sum of weights is zero
    # Check that no division by zero error occurs and result entries where weights sum zero are nan or inf
    sum_weights = df["weights"].groupby(df["group"]).sum()
    for grp in sum_weights.index:
        if sum_weights[grp] == 0:
            assert pd.isna(result[grp]) or pd.isinf(result[grp])
        else:
            # In this test case all sums are zero, but keep for completeness
            assert not pd.isna(result[grp]) and not pd.isinf(result[grp])
