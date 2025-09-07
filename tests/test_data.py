import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_cli_execution_paths():
    # Define the path to the CLI script
    script_path = Path(__file__).parent.parent / "src" / "data" / "data_processing.py"

    # Test with --style weighted_avg
    result_weighted_avg = subprocess.run(
        [sys.executable, str(script_path), "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_weighted_avg = result_weighted_avg.stdout.strip()
    # Assert output is not empty and no error output
    assert output_weighted_avg != ""
    assert result_weighted_avg.stderr == ""

    # Test with --style group
    result_group = subprocess.run(
        [sys.executable, str(script_path), "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_group = result_group.stdout.strip()
    # Assert output is not empty and no error output
    assert output_group != ""
    assert result_group.stderr == ""
