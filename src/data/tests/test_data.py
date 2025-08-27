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


def test_weigthed_average_non_numeric_columns():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [4, 5, 6]
        }
    )
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    # The 'B' column is non-numeric and should cause an error when multiplied
    with pytest.raises(TypeError):
        weigthed_average(df, weights)