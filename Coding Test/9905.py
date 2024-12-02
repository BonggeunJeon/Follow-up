
number_of_case = int(input())

for case in range(number_of_case):
    
    number = int(input())
    
    dp = [0] * (number+1)
    
    for n in range(1, number+1):
        if n == 1:
            dp[n] = 1
        elif n == 2:
            dp[n] = 2
        elif n == 3:
            dp[n] = 4
        elif n >= 4:
            dp[n] = dp[n-3] + dp[n-2] + dp[n-1]
            
    print(dp[n])
            
    
    