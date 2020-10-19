import numpy as np

def matrix_forward_extra(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> np.ndarray:
    '''
    행렬곱이 포함된 함수와 또 다른 함수의 합성함수에 대한 순방향 계산을 수행
    '''
    
    assert X.shape[1] == W.shape[0]
    
    # 행렬곱
    N = np.dot(X,W)
    
    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)
    
    return S
