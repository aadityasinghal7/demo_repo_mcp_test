import subprocess
import sys
import textwrap
import re
import pandas as pd
import numpy as np
import pytest


def test_cli_invocation(tmp_path):
    """Integration test for CLI correctness for both styles producing expected output."""

    # Create a minimal script path
    module_path = "src/data/data_processing.py"

    # Run the CLI with --style weighted_avg
    result_weighted_avg = subprocess.run(
        [sys.executable, module_path, "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_weighted_avg = result_weighted_avg.stdout.strip()

    # Run the CLI with --style group
    result_group = subprocess.run(
        [sys.executable, module_path, "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_group = result_group.stdout.strip()

    # The outputs should be parseable floats / Series output.

    # For weighted_avg style:
    # Since the random seed and data generation are fixed, recompute expected output here:
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
    expected_weighted_avg = (
        data["A"] * 0.2 + data["B"] * 0.3 + data["C"] * 0.5
    ) / (0.2 + 0.3 + 0.5)

    # The CLI prints the resulting pd.Series, which will print as floats separated by newlines
    # or index + floats. We expect the output to be a sequence of floats, one per row.

    # Parse CLI output for weighted_avg: They should be floats, one per line, matching expected
    try:
        cli_values = np.array([float(line) for line in output_weighted_avg.splitlines()])
    except Exception as e:
        pytest.fail(f"Failed to parse weighted_avg output as floats:\n{output_weighted_avg}\n{e}")

    # Assert shape matches expected
    assert cli_values.shape == expected_weighted_avg.shape

    # Assert the printed values closely match the expected (allow small tolerance)
    np.testing.assert_allclose(cli_values, expected_weighted_avg.values, rtol=1e-5, atol=1e-8)


    # For group style:
    # The CLI prints the Series returned by groupweightedaverage:
    # This Series is indexed by 'brand' with the weighted average of 'A' grouped by 'brand' weighted by 'B'

    # We expect output like:
    # brand
    # A    ...
    # B    ...
    # C    ...
    # dtype: float64

    # Let's parse it: It will be several lines, last line dtype, first line maybe empty or 'brand'
    lines = output_group.splitlines()
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    # The CLI prints a pd.Series with index brand

    # The last line should start with 'dtype: '
    assert lines[-1].startswith("dtype:")

    # The first line may be the name of the Series index (brand)
    # The subsequent lines should be "{brand} {value}" pairs separated by whitespace

    # Extract brand:value pairs skipping the possible index line and dtype line
    pairs_lines = lines
    # If first line equal 'brand' (the index name), drop it
    if pairs_lines[0] == "brand":
        pairs_lines = pairs_lines[1:]

    # Drop last dtype line
    pairs_lines = pairs_lines[:-1]

    parsed = {}
    for line in pairs_lines:
        # Each line like: "A    0.123456"
        parts = line.split()
        assert len(parts) == 2, f"Unexpected line format: {line}"
        brand, val_str = parts
        parsed[brand] = float(val_str)

    # Expected results from the Python function:
    expected_group = (data["B"] * data["A"]).groupby(data["brand"]).sum() / data["B"].groupby(data["brand"]).sum()

    # Compare keys
    assert set(parsed.keys()) == set(expected_group.index)

    # Check values close match
    for brand, val in parsed.items():
        expected_val = expected_group.loc[brand]
        assert abs(val - expected_val) < 1e-5, f"Brand {brand} mismatch: {val} vs {expected_val}"