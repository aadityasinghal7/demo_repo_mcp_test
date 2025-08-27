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


def test_weigthed_average_zero_total_weight():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    weights = {"A": 0.0, "B": 0.0}
    try:
        result = weigthed_average(df, weights)
        # Expecting result to be infinite or NaN due to division by zero
        assert result.isnull().all() or np.isinf(result).all()
    except ZeroDivisionError:
        # If function raises ZeroDivisionError, test passes because it handles zero weight improperly
        pass
    except Exception as e:
        # If any other exception occurs, fail the test
        assert False, f"Unexpected exception raised: {e}"