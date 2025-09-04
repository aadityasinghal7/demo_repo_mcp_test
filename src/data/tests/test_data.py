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


def test_weigthed_average_error_handling():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["x", "y", "z"],  # Non-numeric column
        }
    )
    # weights include a non-existent column 'D' and a non-numeric column 'C'
    weights = {"A": 0.5, "C": 0.3, "D": 0.2}

    # We expect that attempting weighted sum with non-numeric or missing columns will raise an error
    with pytest.raises(Exception):
        weigthed_average(df, weights)
