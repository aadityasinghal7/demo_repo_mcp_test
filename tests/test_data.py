import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_functionality():
    # Create a DataFrame with known groups, values, weights, NaN, Inf, and zero weights
    data = pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y", "Z", "Z", "Z"],
            "value": [10, np.nan, 5, np.inf, 1, 2, 3],
            "weight": [1, 0, 2, 0, 0, 3, 4],
        }
    )

    # Expected calculation:
    # Group X: weights = [1,0], values = [10, nan] -> sum(weight*value) = 1*10 + 0*nan = 10, sum(weights) = 1+0=1 => 10/1=10
    # Group Y: weights = [2,0], values = [5, inf] -> sum(weight*value) = 2*5 + 0*inf =10, sum(weights)=2+0=2 => 10/2=5
    # Group Z: weights = [0,3,4], values = [1,2,3] -> sum(weight*value) = 0*1 + 3*2 +4*3=0+6+12=18, sum(weights)=0+3+4=7 => 18/7 ~ 2.5714

    expected = pd.Series({"X": 10, "Y": 5, "Z": 18 / 7})

    # Run the function
    result = groupweightedaverage(
        data, groupby="group", value="value", weights="weight"
    )

    # Use pandas testing for equality with some tolerance for floating point
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


if __name__ == "__main__":
    pytest.main([__file__])
