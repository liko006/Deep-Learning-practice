def multi_inputs_add(x: np.ndarray, y: np.ndarray, sigma: Array_Function) -> float:
    '''
    두 개의 입력을 받아 값을 더하는 함수의 순방향 계산
    '''
    assert x.shape == y.shape
    
    a = x + y
    return sigma(a)
