import numpy as np
import pandas as pd
from src.data.data_processing import weigthed_average


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


def test_groupweightedaverage_non_numeric_input():
    import pandas as pd

    from src.data.data_processing import groupweightedaverage

    # DataFrame with non-numeric value in value column
    df_value_non_numeric = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": [1.0, "non-numeric", 3.0],
            "weight": [1.0, 2.0, 1.0],
        }
    )
    # Expect an exception or error during computation due to non-numeric 'value'
    try:
        groupweightedaverage(
            df_value_non_numeric, groupby="group", value="value", weights="weight"
        )
    except Exception as e:
        assert (
            isinstance(e, (TypeError, ValueError, pd.errors.PerformanceWarning))
            or "could not convert" in str(e).lower()
        )
    else:
        # If no exception, ensure result is NaN or inf where appropriate
        result = groupweightedaverage(
            df_value_non_numeric, groupby="group", value="value", weights="weight"
        )
        assert result.isna().any() or result.isnull().any()

    # DataFrame with non-numeric weight column
    df_weight_non_numeric = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": [1.0, 2.0, 3.0],
            "weight": [1.0, "non-numeric", 1.0],
        }
    )
    # Expect an exception or error during computation due to non-numeric 'weight'
    try:
        groupweightedaverage(
            df_weight_non_numeric, groupby="group", value="value", weights="weight"
        )
    except Exception as e:
        assert (
            isinstance(e, (TypeError, ValueError, pd.errors.PerformanceWarning))
            or "could not convert" in str(e).lower()
        )
    else:
        result = groupweightedaverage(
            df_weight_non_numeric, groupby="group", value="value", weights="weight"
        )
        assert result.isna().any() or result.isnull().any()
