import subprocess
import sys
import os
import pytest

@pytest.mark.integration
def test_command_line_interface_run():
    """Integration test to run data_processing.py CLI with both styles to verify end-to-end execution without errors."""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_processing.py"))
    for style in ["weighted_avg", "group"]:
        result = subprocess.run(
            [sys.executable, script_path, "--style", style],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI exited with error for style {style}: {result.stderr}"
        assert result.stdout.strip(), f"No output produced for style {style}"

