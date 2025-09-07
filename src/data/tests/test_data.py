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


def test_groupweightedaverage_non_numeric_columns():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": ["x", "y", "z", "w"],  # non-numeric values
            "weights": ["low", "high", "medium", "low"],  # non-numeric weights
        }
    )

    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby="group", value="value", weights="weights")
