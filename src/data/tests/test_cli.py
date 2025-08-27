import subprocess
import sys
import pandas as pd
import numpy as np
import pytest


@pytest.mark.integration
def test_cli_script_invocation(tmp_path):
    # Run with --style weighted_avg
    result_weighted_avg = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_weighted_avg = result_weighted_avg.stdout.strip()

    # Run with --style group
    result_group = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_group = result_group.stdout.strip()

    # Recompute expected weighted_avg using the same RNG seed and logic
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
    expected_weighted_avg = (
        data["A"] * 0.2 + data["B"] * 0.3 + data["C"] * 0.5
    ) / (0.2 + 0.3 + 0.5)
    expected_weighted_avg_str = repr(expected_weighted_avg)

    # Check printed output matches expected weighted_avg
    # The output is printed repr(series), so we can allow minor float differences by parsing back and comparing
    output_weighted_avg_series = pd.eval(output_weighted_avg)
    pd.testing.assert_series_equal(
        expected_weighted_avg, output_weighted_avg_series, check_names=False, atol=1e-10
    )

    # Recompute expected groupweightedaverage
    sum_weight_contrib = (data["B"] * data["A"]).groupby(data["brand"]).sum()
    sum_weights = data["B"].groupby(data["brand"]).sum()
    expected_group = sum_weight_contrib / sum_weights

    output_group_series = pd.eval(output_group)
    # pd.eval for Series objects restores Series with index, test close equality
    pd.testing.assert_series_equal(
        expected_group, output_group_series, check_names=False, atol=1e-10
    )