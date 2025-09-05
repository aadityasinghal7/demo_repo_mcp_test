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


def test_groupweightedaverage_zero_weight_sum():
    import pandas as pd

    from src.data.data_processing import groupweightedaverage

    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y"],
            "value": [10, 20, 30, 40],
            "weight": [0, 0, 2, 3],
        }
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")

    # For group 'x', sum of weights is zero, result should be inf or nan or cause no crash.
    # For group 'y', weighted average = (30*2 + 40*3)/ (2+3) = (60+120)/5 = 36
    import numpy as np

    assert np.isinf(result["x"]) or np.isnan(result["x"])
    assert result["y"] == 36.0
