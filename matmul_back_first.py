import numpy as np

def matmul_backward_first(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    '''
    첫 번째 인자에 대한 행렬곱의 역방향 계산 수행
    '''
    
    # 역방향 계산
    dNdX = np.transpose(W, (1,0))
    
    return dNdX
