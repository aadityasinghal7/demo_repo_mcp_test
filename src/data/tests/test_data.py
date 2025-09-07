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


def test_cli_argument_style_group():
    """Test CLI with --style group for correct execution and output without error."""
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout.strip()
    # Output should not be empty and should be a list or series-like string representation
    assert output != ""
    # The output should contain the brand groups A, B, C keys in some form, since groupweightedaverage uses brand groups
    assert any(brand in output for brand in ["A", "B", "C"])
