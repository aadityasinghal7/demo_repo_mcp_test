import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_non_numeric_columns():
    df = pd.DataFrame(
        {
            "group": ["G1", "G1", "G2"],
            "value": [1.0, 2.0, 3.0],
            "weights": ["a", 1.0, 2.0],  # Non-numeric weights (string in first element)
        }
    )
    # Non-numeric weights should cause an exception or error during calculation
    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby="group", value="value", weights="weights")

    df2 = pd.DataFrame(
        {
            "group": ["G1", "G1", "G2"],
            "value": ["x", 2.0, 3.0],  # Non-numeric value
            "weights": [1.0, 1.0, 2.0],
        }
    )
    with pytest.raises(TypeError):
        groupweightedaverage(df2, groupby="group", value="value", weights="weights")
