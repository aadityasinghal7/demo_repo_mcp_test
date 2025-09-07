import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_non_numeric_columns():
    df = pd.DataFrame({"A": ["x", "y", "z"], "B": ["foo", "bar", "baz"]})
    weights = {"A": 0.5, "B": 0.5}
    with pytest.raises(TypeError):
        weigthed_average(df, weights)

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {"A": "weight", "B": 0.5}
    with pytest.raises(TypeError):
        weigthed_average(df, weights)
