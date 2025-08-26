import subprocess
import sys
import re
import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
def test_cli_weighted_avg_mode(tmp_path):
    """
    Integration test for command line interface running data_processing.py with --style weighted_avg option
    to verify output correctness.
    """
    script_path = "src/data/data_processing.py"

    # Run the script with --style weighted_avg
    result = subprocess.run(
        [sys.executable, script_path, "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Capture output, which should be a pandas Series printed as strings like "0    val\n1    val\n..."
    output = result.stdout.strip()
    # Parse the output into a Series
    # The output format is the Python default print of a pandas Series: index + spaces + value per line
    lines = output.splitlines()
    indices = []
    values = []
    for line in lines:
        # split by whitespace separation to separate index and value
        parts = line.strip().split()
        if len(parts) >= 2:
            indices.append(parts[0])
            values.append(float(parts[1]))
    # convert to pd.Series indexed by the parsed indices
    series_out = pd.Series(values, index=indices)

    # Now replicate the exact calculation in the test from data_processing.py to check correctness

    # Create the same random data as in data_processing.py
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

    # Compute weighted average as data_processing.py does with weights
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}
    weighted_sum = sum(data[col] * w for col, w in weights.items())
    expected_series = weighted_sum / sum(weights.values())

    # The output series and expected series should have all close values
    pd.testing.assert_series_equal(series_out.astype(float), expected_series, check_names=False, atol=1e-10)