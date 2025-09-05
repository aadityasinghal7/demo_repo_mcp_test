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
    # zero total weight -> division by zero, expect Inf or warning behavior
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights = {"A": 0.0, "B": 0.0}
    result = weigthed_average(df, weights)
    assert result.isin([np.inf, -np.inf]).all() or result.isna().all()

    # missing columns should raise KeyError
    df = pd.DataFrame({"A": [1, 2]})
    weights = {"A": 1.0, "B": 1.0}  # B missing
    try:
        _ = weigthed_average(df, weights)
        assert False, "Expected KeyError for missing column"
    except KeyError:
        pass

    # NaN values propagate accordingly
    df = pd.DataFrame({"A": [1, np.nan], "B": [3, 4]})
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df, weights)
    expected = (df["A"] * 0.5 + df["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)

    # Inf values propagate accordingly
    df = pd.DataFrame({"A": [1, np.inf], "B": [3, 4]})
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df, weights)
    expected = (df["A"] * 0.5 + df["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)
