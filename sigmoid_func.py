import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    '''
    입력으로 받은 np.ndarray의 각 요소에 대한 sigmoid 함숫값을 계산한다.
    '''
    return 1 / (1 + np.exp(-x))
