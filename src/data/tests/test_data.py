import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average, groupweightedaverage


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


def test_groupweightedaverage_basic_functionality():
    # Create a sample DataFrame with groups and values
    data = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Y", "Z"],
            "value": [10, 20, 30, 40, 50, 60],
            "weight": [1, 2, 1, 1, 2, 1],
        }
    )
    # Calculate weighted average manually for each group:
    # For group X:
    # sum_weighted_value = 10*1 + 20*2 = 10 + 40 = 50
    # sum_weights = 1 + 2 = 3
    # weighted avg = 50 / 3 ≈ 16.6667
    #
    # For group Y:
    # sum_weighted_value = 30*1 + 40*1 + 50*2 = 30 + 40 + 100 = 170
    # sum_weights = 1 + 1 + 2 = 4
    # weighted avg = 170 / 4 = 42.5
    #
    # For group Z:
    # sum_weighted_value = 60*1 = 60
    # sum_weights = 1
    # weighted avg = 60 / 1 = 60

    expected = pd.Series(
        data={
            "X": 50 / 3,
            "Y": 170 / 4,
            "Z": 60,
        }
    )

    # Run the function
    result = groupweightedaverage(data, groupby="group", value="value", weights="weight")

    # Using almost equal for float comparison
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), rtol=1e-6)