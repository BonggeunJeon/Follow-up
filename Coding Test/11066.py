number_of_cases = int(input())

def merging_files():
    """
    sums = [[0 for col in range(len(size_of_files))] for row in range(len(size_of_files))]
    
    for i in range(0, len(size_of_files)):
       for j in range(0, len(size_of_files)):
          sums[i][j] = size_of_files[i] if i == j else 0
    
    
    for i in range(0, len(size_of_files)):
        for j in range(i+1, len(size_of_files)):
            sums[i][j] = size_of_files[j] + sums[i][j-1]
        print(sums[i])
    """    
    # The second approach
    
    """
    total_merging_value = size_of_files[0] + size_of_files[1] 
    size_of_files.insert(1, size_of_files[0] + size_of_files[1])
    del size_of_files[0] 
    
    while len(size_of_files) > 1:
        count = 1
        try:
            index = size_of_files.index(min(size_of_files))
            minimum = min(size_of_files[index-1] + size_of_files[index], size_of_files[index] + size_of_files[index+1])
            size_of_files.insert(index+1, minimum)
            total_merging_value += minimum
            del size_of_files[index]
            count += 1
        except IndexError:
            print(count)
            break
    
    
    print(len(size_of_files), size_of_files) 
    """
    # The third approach
    N, A = int(input()), [0] + list(map(int, input().split()))
    
    S = [0 for _ in range(N+1)]
    for i in range(1, N+1):
        S[i] = S[i-1] + A[i]
 
   
    DP = [[0 for i in range(N+1)] for _ in range(N+1)]
    for i in range(2, N+1): 
        for j in range(1, N+2-i):  
            DP[j][j+i-1] = min([DP[j][j+k] + DP[j+k+1][j+i-1] for k in range(i-1)]) + (S[j+i-1] - S[j-1])
    
    return DP[1][N]

for case in range(number_of_cases):
    print(merging_files())
    



