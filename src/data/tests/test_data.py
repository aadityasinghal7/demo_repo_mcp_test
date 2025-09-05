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


def test_cli_script_execution_weighted_avg_style(tmp_path, capsys):
    import subprocess
    import sys

    script_path = "src/data/data_processing.py"
    result = subprocess.run(
        [sys.executable, script_path, "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Check output is not empty
    output = result.stdout.strip()
    assert output != ""
    # Try to parse output as floats, either a single float or lines of floats (Series)
    lines = output.splitlines()
    # output should contain valid floats
    for line in lines:
        float(line)  # raise if conversion fails
