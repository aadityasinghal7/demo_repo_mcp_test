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
    df = pd.DataFrame({
        "group": ["X", "X", "Y", "Y", "Y"],
        "value": [10, 20, 30, 40, 50],
        "weight": [1, 3, 1, 2, 3]
    })

    # Expected calculation:
    # group X: (10*1 + 20*3) / (1+3) = (10 + 60)/4 = 70/4 = 17.5
    # group Y: (30*1 + 40*2 + 50*3) / (1+2+3) = (30 + 80 + 150)/6 = 260/6 ≈ 43.3333
    expected = pd.Series({
        "X": 17.5,
        "Y": 260 / 6,
    })

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False, rtol=1e-5)