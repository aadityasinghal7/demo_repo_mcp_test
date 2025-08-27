import numpy as np
import pandas as pd

from src.data.data_processing import weigthed_average, groupweightedaverage


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


def test_weighted_functions_with_nan_inf():
    df = pd.DataFrame({
        "brand": ["A", "A", "B", "B"],
        "value": [1.0, np.nan, np.inf, -np.inf],
        "wt": [1.0, 2.0, np.nan, 3.0],
        "A": [np.nan, 2.0, 3.0, np.inf],
        "B": [1.0, np.inf, -np.inf, 4.0],
    })

    # Preprocess: replace inf/-inf with NaN then fill NaN with 0 as recommended
    clean_df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Test weigthed_average with columns 'A' and 'B' and weights
    weights = {"A": 0.6, "B": 0.4}
    result_weighted_average = weigthed_average(clean_df[["A", "B"]], weights)
    expected_weighted_average = (clean_df["A"] * 0.6 + clean_df["B"] * 0.4) / (0.6 + 0.4)
    assert np.allclose(result_weighted_average, expected_weighted_average), (
        f"Weigthed average with NaN/Inf failed. "
        f"Expected {expected_weighted_average}, got {result_weighted_average}"
    )

    # Test groupweightedaverage with groupby='brand' over 'value' weighted by 'wt'
    result_group_weighted = groupweightedaverage(clean_df, groupby="brand", value="value", weights="wt")
    # Compute expected manually:
    grouped = clean_df.groupby("brand")
    sum_weighted = (grouped["wt"] * grouped["value"]).sum()
    sum_weights = grouped["wt"].sum()
    expected_group_weighted = sum_weighted / sum_weights

    pd.testing.assert_series_equal(result_group_weighted, expected_group_weighted, check_names=False)