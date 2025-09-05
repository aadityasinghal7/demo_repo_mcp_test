import numpy as np
import pandas as pd
from src.data.data_processing import weigthed_average


def test_weigthed_average():
    df = pd.DataFrame(
        {
            "A": [2, 0, 0],
            "B": [1, 2, 3],
        }
    )
    weights = {"A": 1.0, "B": 0.0}
    result = weigthed_average(df, weights)
    expected = pd.Series([2.0, 0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_weigthed_average_invalid_column():
    import pandas as pd

    from src.data.data_processing import weigthed_average

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights = {"A": 0.4, "C": 0.6}  # 'C' does not exist in df

    try:
        weigthed_average(df, weights)
    except KeyError:
        pass
    else:
        assert (
            False
        ), "Expected KeyError when weights reference non-existent DataFrame columns"
