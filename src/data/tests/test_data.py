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


def test_weigthed_average_nan_and_inf_handling():
    df = pd.DataFrame(
        {
            "A": [1, np.nan, 3, np.inf, 5],
            "B": [2, 4, np.inf, 8, np.nan],
            "C": [np.nan, 1, 2, 3, 4],
        }
    )
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}
    # Following recommended cleaning from README: replace inf with pd.NA, then fillna(0)
    cleaned_df = df.replace([np.inf, -np.inf], pd.NA).fillna(0)

    result = weigthed_average(cleaned_df, weights)
    # Manually compute expected
    # For each row: (A*0.2 + B*0.3 + C*0.5) / (sum weights=1.0)
    expected_values = []
    for idx, row in cleaned_df.iterrows():
        weighted_sum = sum(row[col] * w for col, w in weights.items())
        expected_values.append(weighted_sum / sum(weights.values()))
    expected = pd.Series(expected_values)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"