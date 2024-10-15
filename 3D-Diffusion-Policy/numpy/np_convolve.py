import numpy as np
import matplotlib.pylab as plt

n = np.arange(10)
x = np.heaviside(n-3, 1) - np.heaviside(n-6, 1)

""" 
    np.convolve : Convolution operation
    
    np.convolve(a, v, mode='full')
    
    a = First one-dimensional input array
    v = Second one-dimensional input array
    
    mode 
     - full : M + N - 1 
     - same : max(M, N) 
     - valid : max(M, N) - min(M, N) + 1
     
    return : Discrete, linear convolution of a and v
"""

Convolution = np.convolve(x, x, 'same')
plt.stem(n, x)
plt.show()

plt.stem(n, Convolution)
plt.show()