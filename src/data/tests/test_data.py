def test_weigthed_average_edge_cases():
    # Case 1: zero total weight -> should raise ZeroDivisionError
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights_zero = {"A": 0.0, "B": 0.0}
    try:
        _ = weigthed_average(df, weights_zero)
        assert False, "Expected ZeroDivisionError due to zero total weight"
    except ZeroDivisionError:
        pass

    # Case 2: missing column in df
    weights_missing = {"A": 0.5, "C": 0.5}  # 'C' not in df
    try:
        _ = weigthed_average(df, weights_missing)
        assert False, "Expected KeyError due to missing column"
    except KeyError:
        pass

    # Case 3: NaN and Inf values in df
    df_nan_inf = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [np.inf, 5, -np.inf],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df_nan_inf, weights)
    # Expected: arithmetic propagates NaN and Inf
    expected = (df_nan_inf["A"] * 0.5 + df_nan_inf["B"] * 0.5) / (0.5 + 0.5)
    pd.testing.assert_series_equal(result, expected)
