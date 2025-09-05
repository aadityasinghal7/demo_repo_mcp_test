import pandas as pd
import pytest

from src.data.data_processing import weigthed_average


def test_weigthed_average_non_numeric_column():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],  # non-numeric column
        }
    )
    weights = {"A": 1.0, "B": 1.0}
    with pytest.raises(TypeError):
        _ = weigthed_average(df, weights)
