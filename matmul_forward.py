import numpy as np

def matmul_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    '''
    순방향 계산을 행렬곱으로 계산
    '''
    
    assert X.shape[1] == W.shape[0], \
    '''
    행렬곱을 계산하려면 첫 번째 배열의 열의 개수와
    두번째 배열의 행의 개수가 일치해야 한다.
    그러나 지금은 첫 번째 배열의 열의개수 가 {0}이고
    두 번째 배열의 행의 개수가 {1}이다.
    '''.format(X.shape[1], W.shape[0])
    
    # 행렬곱 연산
    N = np.dot(X,W)
    
    return N
