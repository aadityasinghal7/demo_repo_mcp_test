import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.data import data_processing


@pytest.mark.integration
@pytest.mark.parametrize("style", ["weighted_avg", "group"])
def test_cli_script_execution(style):
    # Run the script with the given --style argument
    result = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", style],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    output = result.stdout.strip()

    # Parse the output which should be printed result (Series)
    # Try to evaluate output as a pandas Series or numpy array-like
    # The output is expected to be a pandas Series string representation
    # We will parse the output assuming it's valid repr of a pandas Series or ndarray

    # Simple attempt: interpret output lines as pandas Series repr:
    # But since it is just print(weighted_avg) which is a Series,
    # it will look like:
    # Either a float series (e.g. group weighted avg) with index,
    # or a float series with no index (weighted_avg).
    # We'll parse with regex or use eval after safe replacement

    # For 'weighted_avg' style, output is a pandas Series without index:
    # e.g. 0    0.12345
    #       1    -0.2345
    #       dtype: float64

    # For 'group' style, output is a smaller series indexed by brand

    # We parse it line-by-line to build the series
    lines = output.splitlines()
    # remove empty lines and dtype line
    lines = [
        line for line in lines if line.strip() and not line.strip().startswith("dtype")
    ]

    # Weighted_avg output indexed by 0..N or unnamed index (int),
    # Group output indexed by brand name (A, B, C)

    index = []
    values = []

    for line in lines:
        m = re.match(r"^(.*?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", line.strip())
        if not m:
            # Try other pattern: for example if index may be string without spaces, rest value
            m2 = re.match(r"^(\S+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", line.strip())
            if not m2:
                pytest.fail(f"Output line could not be parsed: {line}")
            else:
                index.append(m2.group(1))
                values.append(float(m2.group(2)))
        else:
            index.append(m.group(1))
            values.append(float(m.group(2)))

    # Convert values to np.array for validation
    values = np.array(values)

    # Validate output according to style
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

    if style == "weighted_avg":
        expected = data_processing.weigthed_average(
            data, weights={"A": 0.2, "B": 0.3, "C": 0.5}
        )
        # Check output length matches expected length
        assert len(values) == len(expected)
        # Check values close
        assert np.allclose(values, expected.to_numpy(), rtol=1e-5, atol=1e-8)
        # Check index are numeric strings matching indices
        assert all(str(i) == idx for i, idx in enumerate(index))
    elif style == "group":
        expected = data_processing.groupweightedaverage(
            data, groupby="brand", value="A", weights="B"
        )
        # length and index keys match
        assert len(values) == len(expected)
        # Values close to expected
        assert np.allclose(values, expected.to_numpy(), rtol=1e-5, atol=1e-8)
        # index matches expected.index as strings
        assert sorted(index) == sorted(expected.index.astype(str))
