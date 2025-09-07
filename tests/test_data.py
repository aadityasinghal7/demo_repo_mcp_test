import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_non_numeric_data():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, "non-numeric"], "C": [7, 8, 9]})
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}

    with pytest.raises(TypeError):
        weigthed_average(df, weights)
