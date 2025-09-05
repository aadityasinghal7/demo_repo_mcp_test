import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("style", ["weighted_avg", "group"])
def test_cli_main_functionality(style):
    # Run the CLI with the given style
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", style],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    # Output should not be empty
    assert output, f"Output is empty for style {style}"
    # Output should parse as pandas Series (lines of floats or a float)
    lines = output.splitlines()
    # The output is printed Series, so it can be multiple lines with index and values
    # attempt to parse float values in lines after index
    # For weighted_avg style, output is a single series with index A,B,C or maybe single line with values
    # For group style, output is grouped by brand, so index like A,B,C - so we expect at least some lines
    for line in lines:
        # line could be like "A    0.123456"
        # split and parse float on the right side
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                float(parts[-1])
            except ValueError:
                pytest.fail(f"Output line does not contain float value: {line}")
        else:
            pytest.fail(f"Output line does not have expected format: {line}")


# Note: No extra setup is needed; this runs the script as CLI and checks output correctness.
