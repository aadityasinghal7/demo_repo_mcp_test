from src.data.data_processing import groupweightedaverage


def test_groupweightedaverage_basic_functionality():
    df = pd.DataFrame(
        {
            "group": ["x", "x", "y", "y", "y"],
            "value": [10, 20, 30, 40, 50],
            "weight": [1, 2, 3, 4, 5],
        }
    )
    result = groupweightedaverage(df, groupby="group", value="value", weights="weight")
    expected = pd.Series(
        {
            "x": (10 * 1 + 20 * 2) / (1 + 2),
            "y": (30 * 3 + 40 * 4 + 50 * 5) / (3 + 4 + 5),
        }
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())
