import subprocess
import re
import pytest
import pandas as pd


def test_cli_argument_style_group_extra_inputs():
    # Run the data_processing.py script with --style group
    result = subprocess.run(
        ["python", "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()

    # The output should be a printed pandas Series indexed by brand (A, B, C)
    # Parse the output back into a Series to verify correctness
    # The output of print(Series) will be something like:
    #
    # A    <value>
    # B    <value>
    # C    <value>
    # dtype: float64
    #
    # So we parse accordingly.

    lines = output.splitlines()
    series_data = {}
    dtype_line_found = False
    for line in lines:
        if line.strip().startswith("dtype:"):
            dtype_line_found = True
            break
        match = re.match(r"^\s*([A-Z])\s+([-+]?\d*\.\d+|\d+([eE][-+]?\d+)?)\s*$", line)
        if match:
            key = match.group(1)
            val = float(match.group(2))
            series_data[key] = val
    assert dtype_line_found, "Output does not contain dtype line, output:\n" + output

    # Prepare the same dataframe and calculation to compare with
    import numpy as np

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
    # According to the script, for --style group:
    # weighted_avg = groupweightedaverage(data, groupby="brand", value="A", weights="B")
    weight_contrib = data["B"] * data["A"]
    sum_weight_contrib = weight_contrib.groupby(data["brand"]).sum()
    sum_weights = data["B"].groupby(data["brand"]).sum()
    expected_series = sum_weight_contrib / sum_weights

    # Check all keys and values are close
    assert set(series_data.keys()) == set(expected_series.index)
    for key in series_data:
        assert abs(series_data[key] - expected_series[key]) < 1e-12, f"Value mismatch for brand {key}: expected {expected_series[key]}, got {series_data[key]}"