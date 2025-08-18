def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

import math
from typing import List, Tuple

def gcd_and_lcm(numbers: List[int]) -> Tuple[int, int]:
    """
    Compute the GCD (Greatest Common Divisor) and LCM (Least Common Multiple)
    of a list of integers.

    Args:
        numbers (List[int]): A list of positive integers

    Returns:
        Tuple[int, int]: (gcd, lcm)
    """
    if not numbers:
        raise ValueError("List of numbers cannot be empty")

    # Initialize with the first number
    current_gcd = numbers[0]
    current_lcm = numbers[0]

    for num in numbers[1:]:
        if num <= 0:
            raise ValueError("All numbers must be positive integers")
        current_gcd = math.gcd(current_gcd, num)
        current_lcm = current_lcm * num // math.gcd(current_lcm, num)

    return current_gcd, current_lcm
