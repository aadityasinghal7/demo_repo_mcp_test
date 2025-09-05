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


def test_weighted_average_zero_total_weight():
    import pandas as pd

    from src.data.data_processing import weigthed_average

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {"A": 0, "B": 0}

    result = weigthed_average(df, weights)
    assert (
        result.isna().all()
        or (result == float("inf")).all()
        or (result == float("-inf")).all()
        or (result.isnull().all())
        or (result.isin([float("inf"), float("-inf")]).all())
    ), "Expected result to handle division by zero gracefully"
