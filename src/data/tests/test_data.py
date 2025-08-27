import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_error_handling():
    # Test DataFrame with groups, including a group with zero total weight
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Z", "Z"],
            "value": [1, 2, 3, 4, 5, 6],
            "weight": [0, 0, 1, 2, "a", 1],  # group X will have zero total weight; group Z has non-numeric weight "a"
        }
    )

    # Convert weight column to numeric, coercing errors to NaN to simulate realistic bad input
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Call groupweightedaverage and capture output
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Expected:
    # group X weights sum to 0 + 0 = 0 -> division by zero -> expect inf or nan (will be NaN due to pandas behavior)
    # group Y weights sum 1 + 2 = 3, weighted average = (3*1 + 4*2)/3 = (3 + 8)/3 = 11/3 ≈ 3.6667
    # group Z weights sum = NaN + 1 = 1 (NaN ignored), weighted sum = NaN*5 + 1*6 = 6
    # So group Z average = 6 / 1 = 6.0

    # Check that group X result is NaN (due to division by zero)
    assert pd.isna(result.get("X")), "Expected NaN for group with zero sum of weights (division by zero)"

    # Check group Y weighted average close to expected
    expected_y = (3*1 + 4*2)/3
    assert np.isclose(result.get("Y"), expected_y), f"Expected {expected_y} but got {result.get('Y')} for group Y"

    # Check group Z weighted average = 6.0 ignoring non-numeric weight converted to NaN
    assert np.isclose(result.get("Z"), 6.0), f"Expected 6.0 but got {result.get('Z')} for group Z"

    # Also test missing group: the result should only include groups present, so group "W" is missing
    assert "W" not in result.index, "Result should not contain groups not present in the DataFrame"

    # Confirm the dtype of result is float64 (NaNs force floats)
    assert result.dtype == float, "Result dtype should be float due to NaNs and numeric calculations"

