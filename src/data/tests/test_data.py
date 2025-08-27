import numpy as np
import pandas as pd

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_nan_inf_handling():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, np.nan, np.inf, 4.0, -np.inf, 6.0],
            "wt": [1.0, 2.0, 3.0, 1.0, 2.0, np.nan],
        }
    )

    # Manually calculate expected result using pandas semantics:
    # For group A:
    # weighted sum = sum(weight * value) = 1*1.0 + 2*nan + 3*inf = 1.0 + nan + inf -> nan + inf = nan
    # but pandas sum() ignores NaNs, inf they are additive, yet inf propagates
    # However, pandas sum() skips NaNs but includes inf. So sum_weight_contrib for A is:
    # 1*1.0 + 3*inf = 1 + inf = inf
    # sum_weights = sum(weights): 1 + 2 + 3 = 6 (na is no.)
    # So result = inf / 6 = inf

    # For group B:
    # weighted sum = 1*4.0 + 2*(-inf) + nan*6 = 4.0 + (-inf) + nan -> nan ignored by sum(), so 4 + (-inf) = -inf
    # sum_weights = 1 + 2 + nan = 3 (NaN ignored)
    # result = -inf / 3 = -inf

    expected = pd.Series([np.inf, -np.inf], index=["A", "B"])

    result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")

    pd.testing.assert_series_equal(result, expected, check_names=False)