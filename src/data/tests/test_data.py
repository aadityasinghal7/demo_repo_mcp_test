import subprocess
import sys
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("style", ["weighted_avg", "group"])
def test_cli_integration(style):
    # Run the data_processing module as a script with the given style argument
    result = subprocess.run(
        [sys.executable, "-m", "src.data.data_processing", "--style", style],
        capture_output=True,
        text=True,
        check=False,
    )
    # The exit code should be zero, indicating no error
    assert result.returncode == 0, (
        f"CLI with style '{style}' exited with code {result.returncode}. "
        f"stderr: {result.stderr}"
    )
    # Output should not be empty
    assert result.stdout.strip(), f"CLI output is empty for style '{style}'"