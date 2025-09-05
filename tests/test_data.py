import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_with_nan_and_inf():
    df = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Z", "Z"],
            "value": [1.0, np.nan, np.inf, 4.0, -np.inf, 6.0],
            "wt": [1.0, 2.0, 3.0, np.inf, -np.inf, np.nan],
        }
    )

    # Expected behavior: the calculations involving NaN or Inf will propagate as per pandas.
    # We manually compute expected values ignoring problematic groups with infinite sums:
    # Group X:
    #  weighted sum = 1.0*1.0 + nan*2.0 = nan
    #  sum weights = 1.0 + 2.0 = 3.0
    # => nan/3.0 = nan
    #
    # Group Y:
    # sum weight contrib = inf*3.0 + 4.0*inf = inf + inf = inf
    # sum weights = 3.0 + inf = inf
    # => inf / inf => nan
    #
    # Group Z:
    # sum weight contrib = -inf * -inf + 6.0 * nan = inf + nan = nan
    # sum weights = -inf + nan = nan
    # result = nan

    expected = pd.Series(
        [np.nan, np.nan, np.nan], index=pd.Index(["X", "Y", "Z"], name="group")
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="wt")

    pd.testing.assert_series_equal(
        result, expected, check_names=True, check_exact=False, check_dtype=False
    )


# Note: This test ensures groupweightedaverage returns results consistent with pandas
# Nan and Inf propagate as per pandas groupby sum behavior. Users should pre-clean data as recommended.
