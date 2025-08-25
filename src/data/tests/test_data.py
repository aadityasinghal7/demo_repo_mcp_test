import subprocess
import sys
import re
import pandas as pd
import numpy as np
import pytest


def test_cli_argument_style_weighted_avg_extra_inputs():
    # Run the data_processing.py script as a subprocess with --style weighted_avg
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Capture stdout and try to parse the printed Series output
    output = result.stdout.strip()

    # The script prints a pandas Series, e.g.:
    # 0     X.XXXX
    # 1     Y.YYYY
    # ...
    # We parse lines of the form "<index>   <value>"
    series_dict = {}
    for line in output.split("\n"):
        m = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if m:
            idx = int(m.group(1))
            val_str = m.group(2).strip()
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            series_dict[idx] = val

    # Recreate DataFrame with same random seed and same logic from data_processing.py
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    N_DATA = 100
    data = pd.DataFrame(
        {
            "brand": rng.choice(["A", "B", "C"], size=N_DATA),
            "A": rng.standard_normal(N_DATA),
            "B": rng.standard_normal(N_DATA) * 50 + 20,
            "C": rng.standard_normal(N_DATA) * 100 + 1000,
        }
    )
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}

    # Use the weigthed_average function from src.data.data_processing to get expected result
    from src.data.data_processing import weigthed_average

    expected_series = weigthed_average(data, weights)

    # Assert the keys and values match the expected Series output from weigthed_average
    # The Series index is from 0 to len-1
    # Confirm the parsed keys match expected index
    assert set(series_dict.keys()) == set(expected_series.index), "Output indices do not match expected."

    # Convert parsed output to a float array in index order to compare
    output_values = np.array([series_dict[i] for i in sorted(series_dict.keys())], dtype=float)
    expected_values = expected_series.values

    # Use np.allclose for float comparison
    assert np.allclose(
        output_values, expected_values
    ), f"CLI output does not match expected weighted average.\nOutput: {output_values}\nExpected: {expected_values}"