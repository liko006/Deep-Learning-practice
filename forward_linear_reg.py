import numpy as np
from typing import List, Callable, Dict, Tuple

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def forward_linear_regression(X_batch: np.ndarray, 
                              y_batch: np.ndarray, 
                              weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    선형회귀의 순방향 계산 과정
    '''
    
    # X와 y의 배치 크기가 같은지 확인
    assert X_batch.shape[0] == y_batch.shape[0]
    
    # 행렬곱 계산이 가능한지 확인
    assert X_batch.shape[1] == weights['W'].shape[0]
    
    # B의 모양이 1x1인지 확인
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1
    
    # 순방향 계산 수행
    N = np.dot(X_batch, weights['W'])
    
    P = N + weights['B']
    
    loss = np.mean(np.power(y_batch - P, 2))
    
    # 순방향 계산 과정의 중간값 저장
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P 
    forward_info['y'] = y_batch
    
    return loss, forward_info
