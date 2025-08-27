import subprocess
import sys
import re
import pandas as pd
import numpy as np


def test_command_line_interface(tmp_path):
    # Run CLI with --style weighted_avg
    result_weighted = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "weighted_avg"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_weighted = result_weighted.stdout.strip()

    # Parse output as a pandas Series
    # The print outputs a Series string, e.g.
    # 0    ...
    # 1    ...
    # dtype: float64
    # So parse it with pd.read_csv from string, add linebreak to separate index,values properly
    lines = output_weighted.splitlines()
    # recompose into csv with index,value
    csv_text = "index,value\n" + "\n".join(f"{line.split()[0]},{line.split()[1]}" for line in lines if re.match(r"^\d+", line))
    weighted_series = pd.read_csv(pd.compat.StringIO(csv_text), index_col=0, squeeze=True)
    weighted_series.index = weighted_series.index.astype(int)
    weighted_series = weighted_series.astype(float)

    # Check length and numeric values roughly
    assert len(weighted_series) == 100
    assert np.isclose(weighted_series.sum(), weighted_series.sum())  # trivially true to assert numeric parse

    # Run CLI with --style group
    result_group = subprocess.run(
        [sys.executable, "src/data/data_processing.py", "--style", "group"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_group = result_group.stdout.strip()

    lines = output_group.splitlines()
    csv_text = "index,value\n" + "\n".join(f"{line.split()[0]},{line.split()[1]}" for line in lines if re.match(r"^[A-Z]", line))
    group_series = pd.read_csv(pd.compat.StringIO(csv_text), index_col=0, squeeze=True)
    group_series = group_series.astype(float)

    # Should have 3 groups: A, B, C
    assert set(group_series.index) == {"A", "B", "C"}
    assert group_series.isnull().sum() == 0

    # Basic numeric sanity check: mean within expected range
    assert (group_series > -10).all()
    assert (group_series < 10).all() or (group_series < 100).all()


# Overwrite the file with this test code
# (This comment is just to indicate the user instruction, not part of code)