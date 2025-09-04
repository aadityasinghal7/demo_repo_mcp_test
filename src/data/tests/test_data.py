def test_weigthed_average_edge_cases():
    # Zero total weight (sum weights == 0)
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_zero = {"A": 0, "B": 0}
    try:
        result = weigthed_average(df, weights_zero)
        assert result.isna().all() or np.isinf(result).all() or np.isnan(result).all()
    except ZeroDivisionError:
        pass  # expected behavior if division by zero

    # Missing columns in df that are in weights
    weights_missing = {"A": 1, "C": 1}  # 'C' missing in df
    with pytest.raises(KeyError):
        weigthed_average(df, weights_missing)

    # NaN values in data
    df_nan = pd.DataFrame({"A": [np.nan, 2, 3], "B": [4, np.nan, 6]})
    weights = {"A": 0.5, "B": 0.5}
    result_nan = weigthed_average(df_nan, weights)
    expected_nan = pd.Series(
        [
            (np.nan * 0.5 + 4 * 0.5) / 1.0,
            (2 * 0.5 + np.nan * 0.5) / 1.0,
            (3 * 0.5 + 6 * 0.5) / 1.0,
        ]
    )
    pd.testing.assert_series_equal(result_nan, expected_nan)

    # Inf values in data
    df_inf = pd.DataFrame({"A": [np.inf, 2, 3], "B": [4, -np.inf, 6]})
    result_inf = weigthed_average(df_inf, weights)
    expected_inf = pd.Series(
        [
            (np.inf * 0.5 + 4 * 0.5) / 1.0,
            (2 * 0.5 + -np.inf * 0.5) / 1.0,
            (3 * 0.5 + 6 * 0.5) / 1.0,
        ]
    )
    pd.testing.assert_series_equal(result_inf, expected_inf)
