import subprocess
import sys
import pytest
from src.data import data_processing


def test_cli_main_functionality(monkeypatch, capsys):
    import os

    # Helper to run main with given args, capturing output or errors
    def run_main_with_args(args):
        # Save original argv
        orig_argv = sys.argv[:]
        sys.argv = [orig_argv[0]] + args
        try:
            data_processing.__main__ = None  # Clear cached main module if any
            # Re-import to run the __main__ code with new sys.argv
            from importlib import reload

            reload(data_processing)
        finally:
            sys.argv = orig_argv

    # Test --style weighted_avg runs without exceptions and outputs values
    run_main_with_args(["--style", "weighted_avg"])
    out = capsys.readouterr().out
    # The output should be a series printed (some float numbers)
    assert out.strip(), "Output for weighted_avg should not be empty"
    # Check output is float numbers printed (e.g. comma-separated or similar)
    floats = [float(x) for x in out.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit() or (
        x.startswith('-') and x[1:].replace('.', '', 1).isdigit())]
    assert floats, "Output should contain float numbers for weighted_avg"

    # Test --style group runs without exceptions and outputs values
    run_main_with_args(["--style", "group"])
    out = capsys.readouterr().out
    assert out.strip(), "Output for group should not be empty"
    # Try to parse output floats again
    floats = [float(x) for x in out.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit() or (
        x.startswith('-') and x[1:].replace('.', '', 1).isdigit())]
    assert floats, "Output should contain float numbers for group"

    # Test no --style argument uses default weighted_avg
    run_main_with_args([])
    out = capsys.readouterr().out
    assert out.strip(), "Output for default style should not be empty"

    # Test invalid --style argument yields SystemExit (argparse error)
    import sys as sys_mod

    def run_invalid():
        sys.argv = [sys.argv[0], "--style", "invalid_style"]
        from importlib import reload

        reload(data_processing)

    with pytest.raises(SystemExit):
        run_invalid()