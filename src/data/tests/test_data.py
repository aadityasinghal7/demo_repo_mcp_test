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


def test_cli_argument_style_weighted_avg():
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    # The output should be a pandas Series printed; verify it contains numeric values and index labels A, B, C
    # For example output like: A    <float>\nB    <float>\nC    <float>\ndtype: float64
    assert re.search(r"^\s*A\s+", output, flags=re.MULTILINE)
    assert re.search(r"^\s*B\s+", output, flags=re.MULTILINE)
    assert re.search(r"^\s*C\s+", output, flags=re.MULTILINE)
    # also check that output contains floats and dtype line
    assert re.search(r"dtype:\s*float64", output, flags=re.MULTILINE)
