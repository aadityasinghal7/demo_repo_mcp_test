```python
import pytest
from math import gcd
from your_module import gcd_lcm  # replace 'your_module' with the actual module name

@pytest.mark.parametrize("a, b, expected_gcd, expected_lcm", [
    (12, 18, 6, 36),
    (7, 5, 1, 35),
    (21, 14, 7, 42),
    (0, 5, 5, 0),
    (0, 0, 0, 0),
])
def test_gcd_lcm(a, b, expected_gcd, expected_lcm):
    assert gcd_lcm(a, b) == (expected_gcd, expected_lcm)
```