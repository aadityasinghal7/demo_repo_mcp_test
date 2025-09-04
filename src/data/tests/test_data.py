import numpy as np
import pandas as pd
import pytest

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


def test_weigthed_average_missing_edge_cases():
    # Case 1: zero total weight
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)

    # Case 2: non-numeric column in df
    df_invalid = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
        }
    )
    weights_valid = {"A": 0.5, "B": 0.5}
    with pytest.raises(TypeError):
        weigthed_average(df_invalid, weights_valid)
