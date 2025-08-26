import subprocess
import sys
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("style", ["weighted_avg", "group"])
def test_cli_script_run(style):
    """
    Integration test verifying that the CLI runs correctly with --style weighted_avg and --style group arguments.

    It calls the data_processing.py script using subprocess and checks for successful execution and expected output.
    """
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", style],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI exited with error: {result.stderr}"
    # Since output depends on random data, just check that output is non-empty and is valid printed Series output
    output = result.stdout.strip()
    assert output, "CLI did not print any output"
    # Check output contains expected characters from pandas.Series repr (index and values)
    # Usually pandas Series print as index:value pairs, so check for presence of numeric values and index patterns
    assert any(char.isdigit() for char in output), "Output does not contain numeric values"


