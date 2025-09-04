def test_weigthed_average_non_numeric_input():
    df = pd.DataFrame(
        {
            "A": [2, 0, 0],
            "B": ["x", "y", "z"],  # non-numeric values in a weighted column
        }
    )
    weights = {"A": 1.0, "B": 1.0}
    import pytest

    with pytest.raises(TypeError):
        _ = weigthed_average(df, weights)
