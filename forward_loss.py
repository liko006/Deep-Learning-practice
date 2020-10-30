import numpy as np
from typing import List, Callable, Dict, Tuple

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def forward_loss(X: np.ndarray,
                 y: np.ndarray, 
                 weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    '''
    신경망 모델의 순방향 계산 및 손실값을 단계별로 수행
    '''
    
    M1 = np.dot(X, weights['W1'])
    
    N1 = M1 + weights['B1']
    
    O1 = sigmoid(N1)
    
    M2 = np.dot(O1, weights['W2'])
    
    N2 = M2 + weights['B2']
    
    loss = np.mean(np.power(y - P, 2))
    
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['N2'] = N2
    forward_info['P'] = P
    forward_info['y'] = y
    
    return forward_info, loss
