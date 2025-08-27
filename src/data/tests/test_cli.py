import subprocess
import sys
import re

import pytest


@pytest.mark.integration
def test_cli_argument_styles():
    # Test --style weighted_avg
    result_weighted_avg = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Output should be a pandas Series-like printout (e.g. floats or index / values)
    out_avg = result_weighted_avg.stdout.strip()
    # It should print something non-empty that contains numeric values
    assert out_avg
    assert re.search(r"[-+]?\d*\.\d+|\d+", out_avg), "Output should contain numeric values"

    # Test --style group
    result_group = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    out_group = result_group.stdout.strip()
    assert out_group
    # For 'group', output is a Series indexed by brand like 'A    0.xyz\nB 1.23\nC 2.3'
    # Check it contains brand letters A, B, C and floats
    assert all(b in out_group for b in ["A", "B", "C"])
    assert re.search(r"[-+]?\d*\.\d+|\d+", out_group), "Output should contain numeric values"

    # No crashes => subprocess.run with check=True succeeded without raising

