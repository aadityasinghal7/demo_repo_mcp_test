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


def test_groupweightedaverage_nan_and_inf_handling():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "z", "z"],
            "value": [1.0, np.nan, np.inf, 4.0, 5.0, np.nan],
            "weight": [1.0, 2.0, 3.0, np.inf, np.nan, 1.0],
        }
    )

    # Expected behavior: calculations propagate NaN and inf as per arithmetic rules in pandas
    # Calculate manually the expected result for each group:
    # group x:
    # weight_contrib = [1.0*1.0, 2.0*np.nan] => [1.0, nan] sum => nan
    # sum_weight = [1.0, 2.0] sum => 3.0
    # result 'x' = nan / 3.0 = nan
    #
    # group y:
    # weight_contrib = [3.0*inf, inf*4.0] => [inf, inf] sum => inf
    # sum_weight = [3.0, inf] sum => inf
    # result 'y' = inf / inf = nan (inf/inf is undefined)
    #
    # group z:
    # weight_contrib = [nan*5.0, 1.0*np.nan] => [nan, nan] sum => nan
    # sum_weight = [nan, 1.0] sum => nan (NaN + 1.0 = NaN in sum by default)
    # result 'z' = nan / nan = nan

    expected = pd.Series(
        data=[np.nan, np.nan, np.nan],
        index=pd.Index(["x", "y", "z"], name="group"),
        name=None,
    )

    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    pd.testing.assert_series_equal(result, expected)
