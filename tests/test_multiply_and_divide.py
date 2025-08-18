```python
import pytest

def multiply_and_divide(a: int, b: int, c: int) -> float:
    return divide(multiply(a, b), c)

def test_multiply_and_divide():
    assert multiply_and_divide(2, 3, 3) == 2.0
    assert multiply_and_divide(4, 5, 2) == 10.0
    assert multiply_and_divide(0, 10, 1) == 0.0
    assert multiply_and_divide(6, 7, 0) == 0  # division by zero case
```