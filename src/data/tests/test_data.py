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


def test_groupweightedaverage_edge_cases():
    # Setup dataframe with groups including an all-zero weight group,
    # group with NaN/Inf values, and missing group key
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", None, "D"],
            "value": [1.0, 3.0, np.nan, 4.0, 5.0, 7.0, 10.0],
            "wt": [1.0, 1.0, 0.0, np.inf, 0.0, 2.0, 0.0],
        }
    )
    # Calling groupweightedaverage:
    # group A: weights 1.0,1.0 -> sum weights = 2.0, weighted sum = 1*1 + 3*1 =4 => 2.0
    # group B: weights 0.0, inf (infinite weights treated as inf)
    #   This will cause sum weights to be inf and weighted sum to be inf*value (NaN for first)
    #   The result should be NaN or appropriate handling by pandas division
    # group C: weights all zero, so sum weights = 0, expect inf or NaN from division. Pandas produces NaN.
    # group None (missing key): treated as group nan by groupby, should be included
    # group D: weight zero, expect NaN as sum weights = 0
    result = groupweightedaverage(df, groupby="group", value="value", weights="wt")

    # Expected behavior:
    # For group A:
    expected_A = (1.0*1.0 + 3.0*1.0) / (1.0 + 1.0)  # 4 / 2 = 2.0
    # For group B: weighted sum = (0.0 * NaN) + (inf * 4.0) = inf, sum weights = 0 + inf = inf, result=inf/inf->NaN
    # For group C: weighted sum = 5*0 = 0, sum weights=0 -> division 0/0 -> NaN
    # For group None: weighted sum=7*2=14, sum weights=2, result=7.0
    # For group D: weighted sum=10*0=0, sum weights=0-> NaN

    # Build expected Series with index matching group labels
    expected = pd.Series(
        {
            "A": expected_A,
            "B": np.nan,
            "C": np.nan,
            None: 7.0,
            "D": np.nan,
        }
    )

    # Compare result and expected, using pandas testing for NaN-aware comparison
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_names=False)