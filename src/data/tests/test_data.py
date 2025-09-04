def test_weigthed_average_edge_cases():
    import numpy as np
    import pandas as pd

    from src.data.data_processing import weigthed_average

    # Case 1: zero total weight (sum(weights.values()) == 0)
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights1 = {"A": 0.0, "B": 0.0}
    try:
        result1 = weigthed_average(df1, weights1)
    except ZeroDivisionError:
        pass  # expected
    else:
        assert np.all(
            np.isnan(result1)
        ), "Expected NaN series when total weight is zero"

    # Case 2: missing columns in df specified in weights
    df2 = pd.DataFrame({"A": [1, 2]})
    weights2 = {"A": 1.0, "B": 1.0}
    try:
        _ = weigthed_average(df2, weights2)
    except KeyError:
        pass  # expected
    else:
        assert False, "Expected KeyError due to missing column 'B'"

    # Case 3: NaN and Inf values in df
    df3 = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [np.inf, 2, 3],
        }
    )
    weights3 = {"A": 0.5, "B": 0.5}
    result3 = weigthed_average(df3, weights3)

    expected3 = (df3["A"] * 0.5 + df3["B"] * 0.5) / (0.5 + 0.5)
    # Use np.isclose with nan handling; since NaN or Inf propagate
    comparison = (result3.fillna(-9999) == expected3.fillna(-9999)) | (
        result3.isna() & expected3.isna()
    )
    assert comparison.all(), f"Expected {expected3}, but got {result3}"
