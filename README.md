# Data processing utilities (pandas)

Utilities for computing weighted averages used across data pipelines.

## Install
- Python 3.12
- Create a venv and install deps from the repo root:
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - pip install -r requirements.txt

## Usage

Weighted average of selected columns:
```python
import pandas as pd
from src.data.data_processing import weigthed_average

df = pd.DataFrame({"A": [2, 0, 0], "B": [1, 2, 3]})
weights = {"A": 1.0, "B": 0.0}
result = weigthed_average(df, weights)  # pd.Series([2.0, 0.0, 0.0])
```

Group-wise weighted average:
```python
import pandas as pd
from src.data.data_processing import groupweightedaverage

df = pd.DataFrame(
    {"brand": ["A","A","B"], "value": [1.0, 3.0, 10.0], "wt": [1.0, 1.0, 2.0]}
)
# average(value) weighted by wt, per brand
result = groupweightedaverage(df, groupby="brand", value="value", weights="wt")
# returns a Series indexed by brand with the weighted means
```

Run the demo:
- python src/data/data_processing.py --style weighted_avg
- python src/data/data_processing.py --style group

## Behavior and assumptions
- Inputs must be numeric for the columns used in calculations.
- NaN: standard pandas/NumPy semantics apply (propagate in arithmetic). Pre-clean as needed:
  - df = df.fillna(0) or df = df.dropna(subset=[...])
- Inf/-Inf: recommended to replace before computing:
  - df = df.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
- Strings or non-numerics in weighted columns will raise; cast or filter them out:
  - df["A"] = pd.to_numeric(df["A"], errors="coerce")

## Testing
- From the repo root:
  - pytest
- The project uses a src/ layout. Run pytest from the repository root so imports like from src.data... resolve.
