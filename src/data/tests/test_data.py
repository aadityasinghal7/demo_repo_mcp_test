def test_weigthed_average_zero_total_weight():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights = {"A": 0.0, "B": 0.0}
    # Expect ZeroDivisionError due to division by zero in total_weight
    import pytest

    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, weights)
