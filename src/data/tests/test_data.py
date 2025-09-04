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


def test_groupweightedaverage_error_handling():
    df = pd.DataFrame(
        {
            "group": ["X", "Y", "X"],
            "value": [1, 2, 3],
            "weight": [0.5, 0.5, 0.5],
        }
    )

    # missing groupby column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="missing_group", value="value", weights="weight"
        )

    # missing value column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="group", value="missing_value", weights="weight"
        )

    # missing weight column
    with pytest.raises(KeyError):
        groupweightedaverage(
            df, groupby="group", value="value", weights="missing_weight"
        )
