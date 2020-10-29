import numpy as np
from typing import List, Callable

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def matrix_function_forward_sum(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> float:
    '''
    두 개의 np.ndarray X와 W를 입력받으며 sigma 함수를 포함하는 합성함수의 순방향 계산
    '''
    assert X.shape[1] == W.shape[0]
    
    # 행렬곱
    N = np.dot(X, W)
    
    # 행렬곱 계산 결과를 sigma에 전달
    S = simga(N)
    
    # 행렬 요소의 합을 구함
    L = np.sum(S)
    
    return L
