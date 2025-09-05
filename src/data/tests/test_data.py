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


def test_groupweightedaverage_basic_functionality():
    import pandas as pd

    from src.data.data_processing import groupweightedaverage

    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B", "B"],
            "value": [1.0, 3.0, 10.0, 20.0, 30.0],
            "wt": [1.0, 1.0, 2.0, 1.0, 1.0],
        }
    )
    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")

    expected = pd.Series(
        {
            "A": (1.0 * 1.0 + 3.0 * 1.0) / (1.0 + 1.0),
            "B": (10.0 * 2.0 + 20.0 * 1.0 + 30.0 * 1.0) / (2.0 + 1.0 + 1.0),
        }
    )
    pd.testing.assert_series_equal(result, expected)
