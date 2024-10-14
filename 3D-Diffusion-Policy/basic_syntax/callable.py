from typing import Callable
import torch
import numpy

""" 
    The general syntax for Callable 
    
    Callable[[ArgType1, ArgType2, ...], ReturnType]
    
        - The list inside [] specifies the types of the function's arguments.
        - After the comma, you specify the return type of the function. 
"""

def multiply_by_two(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2

def apply_function_to_tensor(tensor: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return func(tensor)
"""
    func: Callable[[torch.Tensor], torch.Tensor] means that "func" should be a callable that takes a "torch.Tensor" 
    as an argument and return a "torch.Tensor"
    
    torch.Tensor를 argument로 받고 torch.Tensor를 Return 해주는 함수여야함. 
"""

tensor = torch.tensor([1, 2, 3])
result = apply_function_to_tensor(tensor, multiply_by_two)
print(result)

"""
    assert condition, "Optional error message"
    
    The "assert" statement is used to test if a condition is "True".
    If the condition is "False", it raises an "AssertionError" and display an error message.
"""
x = -1
assert x > 0, "X should be positive"