def test_weigthed_average_edge_cases():
    # zero total weight
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights = {"A": 0.0, "B": 0.0}
    try:
        result = weigthed_average(df, weights)
    except ZeroDivisionError:
        pass
    else:
        assert not np.isfinite(
            result
        ).any(), "Result should be non-finite for zero total weight"

    # missing column in df
    weights = {"A": 1.0, "C": 1.0}
    try:
        _ = weigthed_average(df, weights)
        assert False, "Expected KeyError for missing column"
    except KeyError:
        pass

    # NaN values
    df_nan = pd.DataFrame(
        {
            "A": [1, np.nan, 3],
            "B": [4, 5, np.nan],
        }
    )
    weights = {"A": 0.5, "B": 0.5}
    result = weigthed_average(df_nan, weights)
    expected = (df_nan["A"] * 0.5 + df_nan["B"] * 0.5) / (0.5 + 0.5)
    assert result.equals(expected), f"Expected {expected}, but got {result}"

    # Inf values (replace as per README)
    df_inf = pd.DataFrame(
        {
            "A": [1, np.inf, 3],
            "B": [4, 5, -np.inf],
        }
    )
    df_inf_clean = df_inf.replace([np.inf, -np.inf], pd.NA).fillna(0)
    result = weigthed_average(df_inf_clean, weights)
    expected = (df_inf_clean["A"] * 0.5 + df_inf_clean["B"] * 0.5) / (0.5 + 0.5)
    assert result.equals(expected), f"Expected {expected}, but got {result}"
