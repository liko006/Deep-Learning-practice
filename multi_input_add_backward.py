def multi_inputs_add_backward(x: np.ndarray, y: np.ndarray, sigma: Array_Function) -> float:
    '''
    두 개의 입력을 받는 함수의 두 입력에 대한 각각의 도함수 계산
    '''
    
    # 정방향 계산 수행
    a = x + y
    
    # 도함수 계산
    dsda = deriv(sigma, a)
    
    dadx, dady = 1, 1
    
    return dsda * dadx, dsda * dady
