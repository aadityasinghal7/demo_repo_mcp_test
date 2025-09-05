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


def test_weigthed_average_nan_inf_handling():
    import numpy as np
    import pandas as pd

    from src.data.data_processing import weigthed_average

    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, np.inf, 4.0],
            "B": [2.0, 3.0, 4.0, np.nan],
            "C": [np.inf, 1.0, 2.0, 3.0],
        }
    )
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    result = weigthed_average(df, weights)

    # Manual calculation with pandas semantics (inf and nan propagate)
    expected = (
        df["A"] * weights["A"] + df["B"] * weights["B"] + df["C"] * weights["C"]
    ) / sum(weights.values())

    pd.testing.assert_series_equal(result, expected)
