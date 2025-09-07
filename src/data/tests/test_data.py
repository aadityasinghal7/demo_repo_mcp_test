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
    # Construct a DataFrame with NaN and inf values in value and weights
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, np.nan, 2.0, np.inf, -np.inf, 3.0],
            "weight": [1.0, 2.0, np.nan, 1.0, 1.0, np.inf],
        }
    )

    # According to repo notes, inf/-inf should be replaced by pd.NA and fillna(0)
    # Emulate recommended cleaning step to see if groupweightedaverage handles cleaned data as expected
    df_clean = df.replace([np.inf, -np.inf], pd.NA).fillna(0)

    # Compute groupweightedaverage on cleaned data
    result = groupweightedaverage(
        df_clean, groupby="group", value="value", weights="weight"
    )

    # Calculate expected values manually:
    # group A: values=[1.0,0.0], weights=[1.0,2.0] -> weighted sum=1*1+0*2=1; sum weights=3 -> 1/3
    # group B: values=[2.0,0.0], weights=[0.0,1.0] -> weighted sum=2*0+0*1=0; sum weights=1 -> 0/1=0
    # group C: values=[0.0,3.0], weights=[1.0,0.0] -> weighted sum=0*1+3*0=0; sum weights=1 -> 0/1=0
    expected = pd.Series(
        [1.0 / 3, 0.0, 0.0], index=pd.Index(["A", "B", "C"], name="group")
    )

    # Use almost equal because floating point division
    pd.testing.assert_series_equal(
        result, expected, check_names=False, check_dtype=False
    )
