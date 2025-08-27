import subprocess
import re
import pytest
import pandas as pd


@pytest.mark.integration
def test_cli_integration():
    """
    Run the CLI entrypoint with both --style options and validate outputs.
    Since the outputs are random-generated weighted averages, we test parsing and type.
    """

    cmd = ["python", "src/data/data_processing.py", "--style", "weighted_avg"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    # It should be a single number (float)
    val = float(output)

    cmd = ["python", "src/data/data_processing.py", "--style", "group"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()

    # The group output should be a printed pandas Series, e.g.:
    # brand
    # A    <float>
    # B    <float>
    # C    <float>
    # dtype: float64
    # Check at least it has those lines and that values parse as floats
    lines = output.splitlines()
    assert lines[0] == "brand"
    assert lines[-1] == "dtype: float64"
    # Middle lines like 'A    <float>'
    pattern = re.compile(r"^([ABC])\s+(-?\d+(\.\d+)?([eE][-+]?\d+)?)$")
    brands = {"A", "B", "C"}
    seen = set()
    for line in lines[1:-1]:
        m = pattern.match(line)
        assert m is not None, f"Invalid line format: {line}"
        brand = m.group(1)
        val = float(m.group(2))
        seen.add(brand)
    assert seen == brands

