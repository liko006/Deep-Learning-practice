import numpy as np
from typing import List, Callable

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def chain_deriv_3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    '''
    세 함수로 구성된 합성함수의 도함수를 계산하기 위해 연쇄법칙을 사용함
    '''
    
    assert len(chain) == 3, \
    "인자 chain의 길이는 3이어야 함"
    
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    # f1(x)
    f1_of_x = f1(input_range)
    
    # f2(f1(x))
    f2_of_x = f2(f1_of_x)
    
    # df3du
    df3du = deriv(f3, f2_of_x)
    
    # df2du
    df2du = deriv(f2, f1_of_x)
    
    # df1dx
    df1dx = deriv(f1, input_range)
    
    # 각 점끼리 값을 곱함
    return df1dx * df2du * df3du
