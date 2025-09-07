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
            "A": [1.0, 2.0, np.nan, 4.0, np.inf, -np.inf],
            "B": [10.0, np.nan, 30.0, np.inf, 50.0, -np.inf],
        }
    )

    weights = {"A": 0.6, "B": 0.4}

    # According to README:
    # NaN will propagate in arithmetic, Inf/-Inf should be replaced before computing.
    # So test with original dataframe (expect NaN or Inf in output)
    result_orig = weigthed_average(df, weights)

    # Check positions with NaN inputs result in NaN output
    assert pd.isna(result_orig.iloc[2])  # row with A=nan
    assert pd.isna(result_orig.iloc[1])  # row with B=nan propagating to weighted sum

    # Check Inf/-Inf propagate without raising:
    assert np.isinf(result_orig.iloc[4])  # row with A=inf
    assert np.isinf(result_orig.iloc[5])  # row with A=-inf and B=-inf

    # Now replace Inf and NaN as recommended and test calculation is finite and numeric
    df_clean = df.replace([np.inf, -np.inf], pd.NA).fillna(0)

    result_clean = weigthed_average(df_clean, weights)

    # There should be no NaN or Inf in this result
    assert not result_clean.isna().any()
    assert np.all(np.isfinite(result_clean))

    # Manually compute weighted sums for first row as check
    expected_first = df_clean.loc[0, "A"] * 0.6 + df_clean.loc[0, "B"] * 0.4
    assert result_clean.iloc[0] == pytest.approx(expected_first)

    # Check that results for previously NaN/Inf rows are numeric after clean
    assert isinstance(result_clean.iloc[2], float)
    assert isinstance(result_clean.iloc[4], float)
