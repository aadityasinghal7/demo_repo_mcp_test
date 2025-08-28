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
    df = pd.DataFrame(
        {
            "group": ["G1", "G1", "G2", "G2", "G2"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 3, 2, 2, 6],
        }
    )
    # For G1: weighted avg = (10*1 + 20*3) / (1+3) = (10 + 60) / 4 = 70/4 = 17.5
    # For G2: weighted avg = (30*2 + 40*2 + 50*6) / (2+2+6) = (60 + 80 + 300)/10 = 440/10 = 44.0
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series({"G1": 17.5, "G2": 44.0})
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())