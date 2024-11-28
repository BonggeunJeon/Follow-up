"""
N = int(input())

numbers = [0] + list(map(int, input().split()))

maximum = []

 
    1. list out of range
    2. memorization
    3. Timeout -> optimization

sums = []
for i in range(1, len(numbers)):
    sums.append(numbers[0])
    for j in range(i, len(numbers)):
        sums.append(numbers[j])
    #maximum.append(max(sums))
    print(sums)
    sums.clear()
    

#print(maximum)
"""

# The main idea of this problem is the fact that we find out the maximum prefix sum
# For this, we need to use memorization which is the idea of Dynamic Programming
# Otherwise, we have timeout. 
# I was supposed to simply calculate prefix sum from the correspoding index to end
# But if I do that, we have so many time complexity. It's time consuming. 

# However, if we calcualte prefix sum until the previsous prefix sum is minus value,
# We can use memorization as well as we can just only one for loop statement. 

lenNums = int(input())
nums = list(map(int, input().split(' ')))

def maxSubArray(nums) -> int:
    sums = [nums[0]]
    for i in range(1, lenNums):
        sums.append(nums[i] + (sums[i-1] if sums[i-1] > 0 else 0 ))
    
    print(sums)
    return max(sums)
  
print(maxSubArray(nums))
