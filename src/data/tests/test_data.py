import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_non_numeric_and_edge_cases():
    # Setup test data with:
    # - numeric values
    # - non-numeric strings in value and weights columns
    # - nan and infinite values
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C", "D", "D", "E"],
            "value": [1.0, 2.0, np.nan, 4.0, 5.0, np.inf, "x", 7.0, 8.0],
            "wt": [1.0, "y", 2.0, np.nan, np.inf, 1.0, 1.0, "z", 1.0],
        }
    )

    # Coerce value and weights to numeric, forcing non-numeric to NaN as recommended
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["wt"] = pd.to_numeric(df["wt"], errors="coerce")

    # Expected behavior:
    # For groups with valid numeric values and weights, weighted average computed ignoring NaN propagation.
    # NaNs or infinities cause results to propagate or NaN as per pandas standard arithmetic.
    # Calculate manually for valid groups:
    # Group A: rows 0,1 => value: [1.0, 2.0], wt: [1.0, NaN] -> weighted sum = 1*1 + 2*NaN = NaN; sum weights = 1 + NaN = NaN; result should be NaN
    # Group B: rows 2,3 => value: [NaN,4.0], wt: [2.0,NaN] weighted sum = NaN*2 + 4*NaN = NaN, sum weights = 2 + NaN = NaN -> NaN
    # Group C: rows 4,5 => value: [5, inf], wt: [inf, 1] weighted sum = 5*inf + inf*1 = inf + inf = inf; sum weights = inf + 1 = inf; inf / inf = NaN (pandas returns NaN)
    # Group D: rows 6,7 => value: [NaN,7], wt: [1, NaN] weighted sum= NaN*1 + 7*NaN = NaN; sum weights=1 + NaN = NaN -> NaN
    # Group E: row 8 => value:8, wt:1 => 8/1=8.0

    result = groupweightedaverage(df, groupby="group", value="value", weights="wt")

    expected = pd.Series(
        {
            "A": np.nan,
            "B": np.nan,
            "C": np.nan,
            "D": np.nan,
            "E": 8.0,
        }
    )

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False)
