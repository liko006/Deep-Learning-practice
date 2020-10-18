import numpy as np

def leaky_relu(x: np.ndarray) -> np.ndarray:
    '''
    np.ndarray 배열의 각 요소에 'Leaky ReLU' 함수를 적용한다.
    '''
    return np.maximum(0.2 * x, x)
