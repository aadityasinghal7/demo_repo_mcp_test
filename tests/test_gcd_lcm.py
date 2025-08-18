```python
import pytest
from your_module import gcd_and_lcm  # replace 'your_module' with the actual module name

def test_gcd_and_lcm():
    assert gcd_and_lcm([12, 18, 24]) == (6, 72)
    assert gcd_and_lcm([5, 10, 15]) == (5, 30)
    assert gcd_and_lcm([7]) == (7, 7)
    assert gcd_and_lcm([1, 1, 1]) == (1, 1)
    assert gcd_and_lcm([2, 3, 4, 5]) == (1, 60)
    with pytest.raises(ValueError):
        gcd_and_lcm([])
    with pytest.raises(ValueError):
        gcd_and_lcm([2, -3, 4])
```