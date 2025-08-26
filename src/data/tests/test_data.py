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


def test_weigthed_average_edge_cases():
    # Case 1: zero total weight -> expect division by zero (inf or raise)
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0, "B": 0}
    with pytest.raises(ZeroDivisionError):
        _ = weigthed_average(df1, weights1)

    # Case 2: missing columns in df -> KeyError or similar
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 1.0, "B": 2.0}  # 'B' missing
    with pytest.raises(KeyError):
        _ = weigthed_average(df2, weights2)

    # Case 3: NaN values in df columns
    df3 = pd.DataFrame({"A": [1, np.nan], "B": [2, 3]})
    weights3 = {"A": 0.5, "B": 0.5}
    result3 = weigthed_average(df3, weights3)
    expected3 = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    assert result3.equals(expected3), f"Expected {expected3}, got {result3}"

    # Case 4: Inf / -Inf values in df columns
    df4 = pd.DataFrame({"A": [np.inf, 1], "B": [2, -np.inf]})
    weights4 = {"A": 0.5, "B": 0.5}
    result4 = weigthed_average(df4, weights4)
    # Weighted sum: [inf*0.5 + 2*0.5 = inf, 1*0.5 + (-inf)*0.5 = -inf]
    expected4 = pd.Series([np.inf, -np.inf])
    assert result4.equals(expected4), f"Expected {expected4}, got {result4}"