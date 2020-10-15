def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    '''
    두 함수로 구성된 합성함수의 도함수를 계산하기 위해 연쇄법칙을 사용함
    (f2(f1(x)))' = f2'(f1(x)) * f1'(x)
    '''
    
    assert len(chain) == 2, \
    "인자 chain의 길이는 2여야 함"
    
    assert input_range.ndim == 1, \
    "input_range는 1차원 ndarray여야 함"
    
    f1 = chain[0]
    f2 = chain[1]
    
    # df1/dx
    f1_of_x = f1(input_range)
    
    # df1/du
    df1dx = deriv(f1, input_range)
    
    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))
    
    # 각 점끼리 값을 곱함
    return df1dx * df2du
