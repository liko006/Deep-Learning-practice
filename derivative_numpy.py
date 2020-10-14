import numpy as np
from typing import Callable

def deriv(func: Callable[[np.ndarray], np.ndarray], 
          input_: np.ndarray, 
          delta: float = 0.001) -> np.ndarray:
    '''
    배열 input의 각 요소에 대해 함수 func의 도함숫 값 계산
    '''
    
    return (func(input_ + delta) - func(input_ - delta)) / ( 2 * delta)
