```python
import pytest
from your_module import gcd_and_lcm  # replace your_module with the actual module name

def test_gcd_and_lcm():
    # Test with typical positive integers
    assert gcd_and_lcm([12, 18, 24]) == (6, 72)
    # Test with single element list
    assert gcd_and_lcm([7]) == (7, 7)
    # Test with prime numbers
    assert gcd_and_lcm([5, 7, 11]) == (1, 385)
    # Test with repeated numbers
    assert gcd_and_lcm([4, 4, 4]) == (4, 4)
    # Test that empty list raises error
    with pytest.raises(ValueError):
        gcd_and_lcm([])
    # Test that zero or negative values raise error
    with pytest.raises(ValueError):
        gcd_and_lcm([3, 0, 5])
    with pytest.raises(ValueError):
        gcd_and_lcm([3, -1, 5])
```