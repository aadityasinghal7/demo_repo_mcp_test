import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_invalid_weights():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        }
    )

    # Case: total weight is zero
    zero_weights = {"A": 0, "B": 0, "C": 0}
    with pytest.raises(ZeroDivisionError):
        weigthed_average(df, zero_weights)

    # Case: weights contain non-numeric value
    non_numeric_weights = {"A": 0.2, "B": "not_a_number", "C": 0.5}
    with pytest.raises(TypeError):
        weigthed_average(df, non_numeric_weights)

    # Case: weights contain None
    none_weights = {"A": None, "B": 0.3, "C": 0.5}
    with pytest.raises(TypeError):
        weigthed_average(df, none_weights)
