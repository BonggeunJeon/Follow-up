import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt



data = genfromtxt('D:\\09. Practice\Seminar-Report\\3D-Diffusion-Policy\\EMA\\2017_seoul_temperature-2C.csv', delimiter=',', skip_header=1)

days = data[:, 0]
temp = data[:, 1:]



# Average
means = np.repeat(np.mean(temp), len(temp)) # np.repeat이 뭐야? 
# np.mean(temp)를 len(temp) 만큼 반복

# Moving Average (2개 이상의 연속된 데이터 값의 평균을 계속적으로 계산하고 평균을 냄)
""" 
    X'_1 = (x_1 + X_2 + X_3) / 3 
    x'_2 = (x_2 + x_3 + x_4) / 3
    x'_3 = (x_3 + x_4 + x_5) / 3
    
    x'_k = (x_k + x_k-1 + x_k-2) / 3
    
    x'_k = x'_k-1 + (x_k - x_k-n) / n
    
    이동 평균을 이용하면 보다 최근의 자료를 반영하며 데이터의 경향을 알 수 있음 (노이즈 제거 효과도 있음)
    
"""
moving_avg = np.convolve(temp.flatten(), np.ones(30), 'same') / 30 # np.convolve가 뭐야?

# Weighted Moving Average (가중평균이란, 각 데이터에 중요도, 영향도, 빈도에 따라 가중치를 곱하여 구해지는 평균)
# 고등학교 1학년 성적 20%, 2학년 30%, 3학년 50% 
""" 
    가중이동평균 = 가중평균 + 이동평균
    
    x' = nx_k + (n-1)x_k-1 + ... + 2x_k-(n-2) + x_k-(n-1) / n + (n-1) + ... + 2 + 1
    
    최근의 값에 비중을 더 줌
"""
weighted_moving_avg = np.convolve(temp.flatten(), np.arange(1, 31), 'same')/np.sum(np.arange(1, 31)) # np.arrange & np.sum이 뭐야?

# Exponentially Weighted Moving Average 
""" 
    지수가중이동평균 = 오래된 데이터일 수록 가중치를 더 극적으로 감소시킴
    
    x' = a^3x_k-3 + a^2(1-a)x_k-2 + a(1-a)x_k-1 + (1-a)x_k
    
    x' = ax'_k-1 + (1-a)x_k
    
    지수가중이동평균을 이용하면 경향성을 훨씬 눈에 띄게 볼 수 있음. 
    데이터가 급격하게 튀는 경우가 있더라도 크게 영향을 받지 않음.  
"""
alpha_05 = 0.5
alpha_09 = 0.9
alpha_099 = 0.99

EWMA_05 = []
EWMA_09 = []
EWMA_099 = []

for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_05 * Y + (1-alpha_05) * temp[i]
    EWMA_05.append(Y[0])
    
for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_09 * Y + (1-alpha_09) * temp[i]
    EWMA_09.append(Y[0])
    
for i in range(len(days)):
    if i == 0:
        Y = temp[0]
    else:
        Y = alpha_09 * Y + (1-alpha_099) * temp[i]
    EWMA_099.append(Y[0])


plt.plot(days, temp, 'ro', markersize=3, label='original')
plt.plot(days, means, color="#FF6B33", markersize=3, label='mean')
plt.plot(days, moving_avg, color='#C205B9', markersize=3, label='moving average')
plt.plot(days, weighted_moving_avg, color='#39C205', markersize=3, label="weighted moving average")

plt.plot(days, EWMA_05, color="#D1F529", markersize=3, label='EWMA 0.5')
plt.plot(days, EWMA_09, color="#000000", markersize=3, label='EWMA 0.9')
plt.plot(days, EWMA_099, color="#253F85", markersize=3, label='EWMA 0.99')

plt.xlabel('days')
plt.ylabel('temperature')
plt.legend()

plt.show()