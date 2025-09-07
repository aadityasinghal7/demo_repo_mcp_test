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


def test_weigthed_average_invalid_weights():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Case 1: empty weights dict
    weights_empty = {}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_empty)

    # Case 2: weights sum to zero
    weights_zero = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights_zero)
