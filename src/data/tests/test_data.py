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


def test_weigthed_average_non_numeric_input():
    df = pd.DataFrame(
        {
            "A": [2, 4, 6],
            "B": [1, 3, 5],
            "C": ["x", "y", "z"],
        }
    )
    weights = {"A": 0.5, "C": 0.5}  # "C" is non-numeric

    with pytest.raises(TypeError):
        # The current implementation does not explicitly check types,
        # but multiplication with strings should raise a TypeError.
        _ = weigthed_average(df, weights)