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


def test_groupweightedaverage_nan_and_non_numeric_values():
    import numpy as np
    import pandas as pd

    from src.data.data_processing import groupweightedaverage

    # Data with NaN in value and non-numeric in weights
    df_nan_value = pd.DataFrame(
        {"brand": ["A", "A", "B"], "value": [1.0, np.nan, 3.0], "wt": [1.0, 2.0, 3.0]}
    )
    df_non_numeric_weights = pd.DataFrame(
        {"brand": ["A", "A", "B"], "value": [1.0, 2.0, 3.0], "wt": [1.0, "two", 3.0]}
    )

    # With NaN values, result should propagate NaN for group "A"
    result_nan = groupweightedaverage(
        df_nan_value, groupby="brand", value="value", weights="wt"
    )
    assert pd.isna(result_nan.loc["A"])
    assert not pd.isna(result_nan.loc["B"])

    # Non-numeric weights should raise an error
    with pytest.raises(TypeError):
        groupweightedaverage(
            df_non_numeric_weights, groupby="brand", value="value", weights="wt"
        )
