def idiot(a, b):
    
    string_a = str(a)
    string_b = str(b)
    char_zero = '0'
    if char_zero in string_a or char_zero in string_b:
        pass
    
    if len(string_a) > 3 or len(string_b) > 3:
        pass
    
    if string_a == string_b:
        pass
    
    else:
        a = int(string_a[::-1])
        b = int(string_b[::-1])
        
        return max(a, b)

a, b = map(int, input().split())

print(idiot(a, b))
