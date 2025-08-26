import subprocess
import pandas as pd
import numpy as np
import ast
import sys
import io

import pytest


@pytest.mark.integration
def test_cli_group_mode(tmp_path):
    # Run the data_processing.py script with --style group option
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()

    # The output is a printed pandas Series. Parse it safely.
    # It should look like:
    # brand
    # A    x.xxx
    # B    y.yyy
    # C    z.zzz
    # dtype: float64

    # Convert the string back to a pandas Series for comparison
    # Replace possible spaces and parse with pandas read_csv from a string buffer
    from io import StringIO

    # Prepare the text to parse into a DataFrame then Series
    # Remove the last dtype line
    lines = output.splitlines()
    if lines[-1].startswith("dtype:"):
        lines = lines[:-1]
    # The first line is the name of the index
    index_name = lines[0].strip()
    # The rest are index value and float values (separated by spaces)
    data_lines = lines[1:]
    data_text = "\n".join(data_lines)

    # Now parse it as two columns using whitespace separation
    df = pd.read_csv(StringIO(data_text), sep=r"\s+", header=None, names=[index_name, "val"])

    actual_series = pd.Series(df.val.values, index=df[index_name])
    actual_series.index.name = index_name

    # Reproduce the internal calculation to compare results
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

    # Import the tested function from data_processing.py
    # Since the function is not imported yet, re-import it here
    from src.data.data_processing import groupweightedaverage

    expected_series = groupweightedaverage(data, groupby="brand", value="A", weights="B")

    # Check that the output matches expected values to a reasonable decimal precision
    pd.testing.assert_series_equal(actual_series, expected_series, check_names=True, rtol=1e-5, atol=1e-8)