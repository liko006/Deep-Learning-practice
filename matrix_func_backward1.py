import numpy as np

def matrix_function_backward_1(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> np.ndarray:
    '''
    첫 번째 요소에 대한 행렬함수의 도함수 계산
    '''
    
    assert X.shape[1] == W.shape[0]
    
    # 행렬곱
    N = np.dot(X,W)
    
    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)
    
    # 역방향 계산
    dSdN = deriv(sigma, N)
    
    # dNdX
    dNdX = np.transpose(W, (1,0))
    
    # 계산한 값을 모두 곱함. 여기서는 dNdX의 모양이 1*1 이므로 순서는 무관함
    return np.dot(dSdN, dNdX)
