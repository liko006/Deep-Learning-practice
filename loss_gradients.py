import numpy as np
from typing import List, Callable, Dict, Tuple

# np.ndarray를 인자로 받고 np.ndarray를 반환하는 함수
Array_Function = Callable[[np.ndarray], np.ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    선형회귀 모형의 dLdW, dLdB 계산
    '''
    batch_size = forward_info['X'].shape[0]
    
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    
    dPdN = np.ones_like(forward_info['N'])
    
    dPdB = np.ones_like(forward_info['B'])
    
    dLdN = dLdP * dPdN
    
    dNdW = np.transpose(forward_info['X'], (1,0))
    
    # 여기서 행렬곱을 수행함
    # dNdW가 왼쪽에 와야 함
    dLdW = np.dot(dNdW, dLdN)
    
    # 배치 크기에 해당하는 차원에 따라 합을 계산함
    dLdB = (dLdP * dPdB).sum(axis=0)
    
    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    
    return loss_gradients
