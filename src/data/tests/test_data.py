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
    # Case 1: Zero total weight -> Expect division by zero leads to Inf or NaN
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0.0, "B": 0.0}
    result1 = weigthed_average(df1, weights1)
    assert result1.isna().all() or np.isinf(result1).all()

    # Case 2: Missing columns in df -> KeyError expected
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 1.0, "B": 1.0}  # B missing from df2
    import pytest

    with pytest.raises(KeyError):
        weigthed_average(df2, weights2)

    # Case 3: NaN values propagate in output
    df3 = pd.DataFrame({"A": [1, np.nan], "B": [3, 4]})
    weights3 = {"A": 0.5, "B": 0.5}
    result3 = weigthed_average(df3, weights3)
    expected3 = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result3, expected3)

    # Case 4: Inf values handled correctly (inf * weight) and propagated
    df4 = pd.DataFrame({"A": [1, np.inf], "B": [3, 4]})
    weights4 = {"A": 0.6, "B": 0.4}
    result4 = weigthed_average(df4, weights4)
    expected4 = (df4["A"] * 0.6 + df4["B"] * 0.4) / (0.6 + 0.4)
    pd.testing.assert_series_equal(result4, expected4)
