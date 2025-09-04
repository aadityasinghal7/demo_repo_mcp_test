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


def test_weigthed_average_edge_cases():
    # Case 1: Zero total weight -> should raise ZeroDivisionError
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights = {"A": 0, "B": 0}
    import pytest

    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)

    # Case 2: Missing columns in DataFrame for given weights -> should raise KeyError
    df = pd.DataFrame({"A": [1, 2]})
    weights = {"A": 0.5, "B": 0.5}
    with pytest.raises(KeyError):
        weigthed_average(df, weights)

    # Case 3: NaN and Inf values in DataFrame columns
    df = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [np.inf, 5, 6],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df.replace([np.inf, -np.inf], np.nan).fillna(0), weights)
    expected = pd.Series(
        [
            (1 * 0.5 + 0 * 0.5) / 1.0,
            (0 * 0.5 + 5 * 0.5) / 1.0,
            (3 * 0.5 + 6 * 0.5) / 1.0,
        ]
    )
    assert result.equals(expected), f"Expected {expected} but got {result}"
