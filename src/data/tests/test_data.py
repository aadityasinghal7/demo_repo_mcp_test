import subprocess
import sys
import pytest

@pytest.mark.integration
def test_cli_script_runs():
    script_path = "src/data/data_processing.py"

    # Run with --style weighted_avg
    result_weighted = subprocess.run(
        [sys.executable, script_path, "--style", "weighted_avg"],
        capture_output=True,
        text=True,
    )
    assert result_weighted.returncode == 0
    assert result_weighted.stdout.strip() != ""

    # Run with --style group
    result_group = subprocess.run(
        [sys.executable, script_path, "--style", "group"],
        capture_output=True,
        text=True,
    )
    assert result_group.returncode == 0
    assert result_group.stdout.strip() != ""