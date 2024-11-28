number_of_cases = int(input())

def merging_files(size_of_files):
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
        try:
            index = size_of_files.index(min(size_of_files))
            minimum = min(size_of_files[index-1] + size_of_files[index], size_of_files[index] + size_of_files[index+1])
            size_of_files.insert(index+1, minimum)
            total_merging_value += minimum
            del size_of_files[index]
        except IndexError:
            print("Sibal")
            break
    
    return total_merging_value + sum(size_of_files)
    """
    

for case in range(number_of_cases):
    number_of_files = int(input())
    size_of_files = list(map(int, input().split()))
    
    merging_files(size_of_files)
    



