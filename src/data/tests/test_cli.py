import subprocess
import sys
import pandas as pd
import numpy as np
import pytest
from src.data import data_processing


def test_cli_invocation_args(monkeypatch, tmp_path):
    # Create a temporary Python script to run data_processing.py with --style weighted_avg
    script_path = tmp_path / "data_processing.py"
    with open(script_path, "w") as f:
        f.write(
            """
import sys
sys.path.insert(0, "")
from src.data.data_processing import weigthed_average
import numpy as np
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style", type=str, default="weighted_avg", choices=["weighted_avg", "group"]
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N_DATA = 100
    data = pd.DataFrame(
        {
            "brand": rng.choice(["A", "B", "C"], size=N_DATA),
            "A": rng.standard_normal(N_DATA),
            "B": rng.standard_normal(N_DATA) * 50 + 20,
            "C": rng.standard_normal(N_DATA) * 100 + 1000,
        }
    )
    if args.style == "weighted_avg":
        weighted_avg = weigthed_average(data, weights={"A": 0.2, "B": 0.3, "C": 0.5})
    elif args.style == "group":
        from src.data.data_processing import groupweightedaverage
        weighted_avg = groupweightedaverage(
            data, groupby="brand", value="A", weights="B"
        )
    print(weighted_avg)
"""
        )

    # Run the actual data_processing.py via subprocess with --style weighted_avg
    import os

    # Identify the actual script location in the repo for running
    dp_path = "src/data/data_processing.py"
    result = subprocess.run(
        [sys.executable, dp_path, "--style", "weighted_avg"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    output = result.stdout.strip()

    # We can also compute the expected result in Python using the same code for weighted_avg style
    rng = np.random.default_rng(42)
    N_DATA = 100
    data = pd.DataFrame(
        {
            "brand": rng.choice(["A", "B", "C"], size=N_DATA),
            "A": rng.standard_normal(N_DATA),
            "B": rng.standard_normal(N_DATA) * 50 + 20,
            "C": rng.standard_normal(N_DATA) * 100 + 1000,
        }
    )
    expected_series = data_processing.weigthed_average(data, weights={"A": 0.2, "B": 0.3, "C": 0.5})

    # The CLI prints the Series, so it should be like expected_series.to_string()
    expected_output = expected_series.to_string()

    assert output == expected_output, "CLI output does not match expected weighted average output"