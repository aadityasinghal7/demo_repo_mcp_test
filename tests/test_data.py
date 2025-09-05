import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_zero_total_weight():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {"A": 0, "B": 0}

    with pytest.raises(ZeroDivisionError):
        _ = weigthed_average(df, weights)
