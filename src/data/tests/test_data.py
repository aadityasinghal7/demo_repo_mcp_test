import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_non_numeric_columns():
    df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B"],
            "value": [1.0, "non-numeric", 3.0, 4.0],
            "wt": [1.0, 2.0, "non-numeric", 1.0],
        }
    )
    # Expecting an exception due to non-numeric columns being used in arithmetic
    with pytest.raises(TypeError):
        groupweightedaverage(df, groupby="brand", value="value", weights="wt")