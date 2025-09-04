def test_weigthed_average_edge_cases():
    # Edge case 1: zero total weight
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights1 = {"A": 0.0, "B": 0.0}
    try:
        result1 = weigthed_average(df1, weights1)
        assert all(
            result1.isna() | np.isinf(result1)
        ), "Expected NaN or Inf when total weight is zero"
    except ZeroDivisionError:
        pass  # Acceptable behaviour if ZeroDivisionError raised

    # Edge case 2: missing columns in weights
    df2 = pd.DataFrame({"A": [1, 2, 3]})
    weights2 = {"A": 0.5, "B": 0.5}  # B missing from df2
    with pytest.raises(KeyError):
        weigthed_average(df2, weights2)

    # Edge case 3: NaN and Inf values in df
    df3 = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [np.inf, 5.0, -np.inf],
        }
    )
    weights3 = {"A": 0.6, "B": 0.4}
    result3 = weigthed_average(df3, weights3)
    expected3 = (df3["A"] * 0.6 + df3["B"] * 0.4) / (0.6 + 0.4)
    pd.testing.assert_series_equal(result3, expected3)
