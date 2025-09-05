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


def test_weighted_average_nan_and_inf_values():
    import numpy as np
    import pandas as pd

    from src.data.data_processing import weigthed_average

    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0, np.inf, 5.0],
            "B": [2.0, 4.0, np.nan, 6.0, -np.inf],
            "C": [np.nan, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Following documented assumptions, replace inf and -inf with pd.NA, then fill na with 0
    clean_df = df.replace([np.inf, -np.inf], pd.NA).fillna(0)

    weights = {"A": 0.3, "B": 0.5, "C": 0.2}
    result = weigthed_average(clean_df, weights)

    expected = (
        clean_df["A"] * weights["A"]
        + clean_df["B"] * weights["B"]
        + clean_df["C"] * weights["C"]
    ) / sum(weights.values())

    pd.testing.assert_series_equal(result, expected)
