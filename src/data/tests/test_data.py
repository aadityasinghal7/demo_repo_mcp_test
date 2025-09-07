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
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": [1.0, "non-numeric", 3.0],
            "weight": [1.0, 2.0, "non-numeric"],
        }
    )

    # Expect a TypeError or ValueError due to invalid operations on non-numeric data
    with pytest.raises((TypeError, ValueError)):
        groupweightedaverage(df, groupby="group", value="value", weights="weight")
