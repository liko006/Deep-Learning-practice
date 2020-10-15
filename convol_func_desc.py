import numpy as np
from typing import List, Callable

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    
    '''
    두 함수를 연쇄(chain)적으로 평가
    '''
    
    assert len(chain) == 2, \
    "인자 chain의 길이는 2여야 함"
    
    f1 = chain[0]
    f2 = chain[1]
    
    return f2(f1(x))
