import numpy as np

""" 
    numpy.repeat(a, repeats, axis=None)
    
    a = array
    repeats = The number of repetitions for each element
"""
x = np.repeat(2, 7) #[2 2 2 2 2 2 2]

y = np.repeat([1, 2], [3, 4]) # [1 1 1 2 2 2 2]

y_1 = np.array([[1, 2], [3, 4]])
""" 
    np.repeat(y_1, 3) 
    axis = None 일 경우 : shape에 상관없이 flatten 해서 처리 
    [1 1 1 2 2 2 3 3 3 4 4 4] <- (12, 0)
    
    np.repeat(y_1, 3, axis = 0) 
    axis = 0 일 경우 : 가장 바깥쪽 괄호 기준으로 각 요소 추가
    [[1, 2],
     [1, 2],
     [1, 2],
     [3, 4],
     [3, 4],
     [3, 4]]
    
    np.repeat(y_1, 3, axis = 1)
    axis = 1 일 경우 : 그 다음 바깥쪽 기준으로 각 요소 추가
    [[1 1 1 2 2 2]
     [3 3 3 4 4 4]]
"""

z = np.arange(27).reshape((3, 3, 3))

print(np.repeat(z, 3))
print(np.repeat(z, 3, axis=0))
print(np.repeat(z, 3, axis=1))
print(np.repeat(z, 3, axis=2))



