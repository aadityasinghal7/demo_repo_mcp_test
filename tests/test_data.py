import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_cli_script_runs_correctly():
    script_path = Path(__file__).parent.parent / "src" / "data" / "data_processing.py"
    for style in ["weighted_avg", "group"]:
        result = subprocess.run(
            [sys.executable, str(script_path), "--style", style],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI exited with non-zero for style={style}"
        assert result.stdout.strip(), f"No output produced for style={style}"
