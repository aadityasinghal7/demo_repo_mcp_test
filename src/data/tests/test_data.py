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


def test_cli_script_execution_group_style(tmp_path, capsys):
    import subprocess
    import sys

    script_path = tmp_path / "data_processing.py"
    # Copy the actual script content for execution in tmp_path
    from pathlib import Path

    original_script = Path(__file__).parent.parent / "data_processing.py"
    script_path.write_text(original_script.read_text())

    result = subprocess.run(
        [sys.executable, str(script_path), "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()

    # Output should be a printed Series indexed by brand with numeric values
    # Parse output via pandas (expect 3 brands A,B,C)
    from io import StringIO

    import pandas as pd

    try:
        # The output is a printed Series, usually like
        # brand
        # A    some_val
        # B    some_val
        # C    some_val
        # dtype: float64
        # We can parse it as a string table with whitespace sep.
        parsed = pd.read_csv(StringIO(output), sep=r"\s+", index_col=0)
    except Exception:
        # If printout is a single line (e.g. dict like), fallback to eval
        parsed = pd.Series(eval(output))

    # Validate expected brands present
    assert set(parsed.index) == {"A", "B", "C"}
    # Validate all values are floats
    assert all(isinstance(x, (float, int)) for x in parsed.values)
    # Validate no NaNs or infs
    assert not parsed.isnull().any()
    assert (~parsed.isin([float("inf"), float("-inf")])).all()
