import re
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_cli_integration():
    script_path = "src/data/data_processing.py"
    # Run with style=weighted_avg
    result_weighted_avg = subprocess.run(
        [sys.executable, script_path, "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_weighted_avg = result_weighted_avg.stdout.strip()
    # The output should be a Series-like printed with numeric values; check for numeric content
    assert re.search(
        r"\d", output_weighted_avg
    ), "Output weighted_avg does not contain numbers"

    # Run with style=group
    result_group = subprocess.run(
        [sys.executable, script_path, "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_group = result_group.stdout.strip()
    # Likewise check for result output with numeric content and group keys (brands A, B, C)
    assert re.search(
        r"[ABC]", output_group
    ), "Output group does not contain expected group keys"
    assert re.search(r"\d", output_group), "Output group does not contain numbers"
