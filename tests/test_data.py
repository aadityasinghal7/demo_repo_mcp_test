import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_non_numeric_inputs():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [1, "a", 3, 4],
            "weight": [1, 2, "b", 4],
        }
    )

    # Non-numeric in 'value' column should cause an exception during multiplication
    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # Fix 'value' column and set non-numeric in 'weight' column and check exception again
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby="group", value="value", weights="weight")
