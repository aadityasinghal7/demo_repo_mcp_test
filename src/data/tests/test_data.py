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


def test_groupweightedaverage_non_numeric_values_or_weights():
    df_numeric_weights = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1, 2, 3, 4],
            "weight": ["a", "b", "c", "d"],
        }
    )
    # weights column is non-numeric, should raise or produce NaNs
    result = groupweightedaverage(
        df_numeric_weights, groupby="group", value="value", weights="weight"
    )
    # Result will be NaN because multiplication with strings yields NaN or TypeError, depends on pandas version
    assert result.isna().all() or result.isnull().all()

    df_non_numeric_value = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": ["a", "b", "c", "d"],
            "weight": [1.0, 0.5, 0.2, 0.1],
        }
    )
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_non_numeric_value, groupby="group", value="value", weights="weight"
        )

    # Mixed case: weight column numeric but contains a string value among numbers
    df_mixed_weights = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "weight": [1, 0.5, "bad_weight", 0.1],
        }
    )
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_mixed_weights, groupby="group", value="value", weights="weight"
        )
