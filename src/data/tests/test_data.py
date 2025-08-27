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
    # Non-numeric column in DataFrame
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [4, 5, 6]
        }
    )
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    # Expect TypeError or ValueError because of non-numeric column multiplication
    with pytest.raises(TypeError):
        weigthed_average(df, weights)

    # Missing column in DataFrame
    df2 = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "C": [4, 5, 6]
        }
    )
    weights2 = {"A": 0.5, "B": 0.3, "C": 0.2}  # B missing in df2
    with pytest.raises(KeyError):
        weigthed_average(df2, weights2)

    # Weights sum to zero
    df3 = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        }
    )
    weights3 = {"A": 0.0, "B": 0.0}
    # This will cause division by zero -> expect ZeroDivisionError or numpy warning/error
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df3, weights3)