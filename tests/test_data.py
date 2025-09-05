import subprocess
import sys

import pytest


def test_cli_invocation_with_invalid_style():
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "invalid_style"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "invalid choice" in result.stdout
