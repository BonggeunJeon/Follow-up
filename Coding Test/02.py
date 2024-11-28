def merging_files(lst, sub_merge):
    
    sorted_list = sorted(lst)
    
    if len(sorted_list) > 2:
        sub_merge.append(sorted_list[0] + sorted_list[1]) 
        sorted_list.append(sorted_list[0] + sorted_list[1])
        del sorted_list[0:2]
        
        merging_files(sorted_list, sub_merge)
    
    if len(sorted_list) > 1:
        return sum(sub_merge, sub_merge[-1] + sub_merge[-2])
    

number_of_cases = int(input())
outputs = []

for case in range(number_of_cases):
    number_of_chapters = int(input())
    
    if number_of_chapters < 3 or number_of_chapters > 500:
        break
        #raise ValueError("The number of chapters have to be more than 3 and less then 500")
    
    sizes_of_chapters = input().split()
    
    for i in range(len(sizes_of_chapters)):
        sizes_of_chapters[i] = int(sizes_of_chapters[i])
    
    if sum(sizes_of_chapters) > 10000:
        break
        #raise ValueError("The total size of chapters cannot be exceed more than 10,000")
    
    sub = []
    sub.clear()
    outputs.append(merging_files(sizes_of_chapters, sub))
    
for i in range(len(outputs)):
    print(outputs[i])